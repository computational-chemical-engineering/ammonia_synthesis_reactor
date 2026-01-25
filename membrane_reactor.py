import math
import os
import warnings
import importlib
import numpy as np
import json
import scipy as sp
import scipy.sparse.linalg as sla
from scipy.sparse import csc_array

from pymrm import non_uniform_grid, construct_coefficient_matrix, construct_grad, construct_div, construct_convflux_upwind, interp_cntr_to_stagg_tvd, upwind, update_csc_array_indices, interp_cntr_to_stagg, interp_stagg_to_cntr, compute_boundary_values, construct_boundary_value_matrices, construct_interface_matrices, NumJac, clip_approach
from gas_mixture_correlations import GasMixtureCorrelations
from ammonia_synthesis_kinetics import AmmoniaSynthesisKinetics
import defaults  # Import the defaults module

class TestKinetics:
    def __init__(self, T=None, p=None):
        """Initialize test kinetics with optional temperature and pressure arrays."""
        self.k_f = 0.1
        self.set_T_and_p(T, p)
    def __call__(self, p):
        """Return species source terms given partial pressures.

        Args:
            p (ndarray): Partial pressures shaped like (..., num_species).

        Returns:
            ndarray: Reaction rates (stoichiometric sources) with same leading shape.
        """
        c = p/(8.31*self.T)
        #3H2 + N2 <=> 2NH3
        return self.k_f * c[..., [0]] * c[..., [1]] *np.array([-3.0,-1.0,2.0]).reshape((1,1,-1))  # Example reaction rate for testing
        #return self.k_f *  c[..., [0]] * np.array([-1.0,0.0,0.5]).reshape((1,1,-1)) + self.k_f *  c[..., [1]] * np.array([0.0,-1.0,0.5]).reshape((1,1,-1))  # Example reaction rate for testing
        #return self.k_f * c[..., [1]] * np.array([0.0,-1.0,2.0]).reshape((1,1,-1))  # Example reaction rate for testing
    def set_T_and_p(self, T=None, p=None):
        """Set (and broadcast) temperature and pressure fields used in rate calc."""
        if (T is not None):
            self.T = T[..., np.newaxis]
        if (p is not None):
            self.p = p[..., np.newaxis]
class MembraneReactor:
    """2D axisymmetric membrane reactor model.

    Simulates coupled mass, momentum, and (optionally) energy transport with
    catalytic reaction and selective permeation between retentate and permeate
    regions on non-uniform grids. Supports continuation and pseudo‑transient
    strategies for robust steady-state convergence.
    """
    def __init__(self, config_file=None, c=None, p=None, T=None, **kwargs):
        """Construct reactor, load defaults, apply overrides, allocate fields.

        Args:
            config_file (str|None): Optional JSON file overriding defaults.
            c (ndarray|None): Initial concentration field (z,r,s).
            p (ndarray|None): Initial pressure field (z,r).
            T (ndarray|None): Initial temperature field (z,r).
            **kwargs: Explicit overrides of default parameters.

        Side Effects:
            Creates spatial discretization, initializes kinetics & Jacobian
            matrices, performs a non-reactive pressure/velocity initialization.
        """

        # Step 1: Load defaults from the Python module
        importlib.reload(defaults)
        param_dict = defaults.DEFAULTS.copy()  # Copy to avoid modifying the original

        # Step 2: Load user-supplied settings from a JSON file (if provided)
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                user_config = json.load(f)
            param_dict.update(user_config)  # Override defaults with user values

        # Step 3: Override with explicitly provided kwargs
        param_dict.update(kwargs)

        # Assign attributes
        for key, value in param_dict.items():
            setattr(self, key, value)
        self.param_dict = param_dict
        
        # Step 4: Load user-supplied settings from a CSV file (if provided)
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                user_config = json.load(f)
            param_dict.update(user_config)  # Override defaults with user values

        # Step 5: Override with explicitly provided kwargs
        param_dict.update(kwargs)

        # Assign attributes
        for key, value in param_dict.items():
            setattr(self, key, value)

        self.init_derived_parameters()
        self.init_fields(c, p, T)
        _, T_ret = self.split_perm_and_ret(self.T)
        _, p_ret = self.split_perm_and_ret(self.c_p[...,-1])

        self.kinetics = AmmoniaSynthesisKinetics(self.species, T=T_ret, p=p_ret, rho_b=self.rho_b, rho_c=self.rho_c)
        #self.kinetics = TestKinetics(T=T_ret, p=p_ret)
        self.init_jac()

        self.factor_norm_c = (self.c_p[...,:-1].size)**(-1.0/self.ord_norm)
        self.factor_norm_p = (self.c_p[...,-1].size)**(-1.0/self.ord_norm)

    def init_derived_parameters(self):
        """Compute geometry-dependent counts, densities, permeabilities & inlets.

        Populates:
            num_r_perm / num_r_ret, membrane permeability
            array (self.perm), inlet flux distributions, Reynolds/Schmidt groups.
        """
        self.y_ret_in = np.asarray(self.y_ret_in, dtype=np.float64).reshape((1,1,-1))  # Convert to numpy array
        self.y_perm_in = np.asarray(self.y_perm_in, dtype=np.float64).reshape((1,1,-1))  # Convert to numpy array
        self.y_ret_init = np.asarray(self.y_ret_init, dtype=np.float64).reshape((1,1,-1))  # Convert to numpy array
        self.y_perm_init = np.asarray(self.y_perm_init, dtype=np.float64).reshape((1,1,-1))  # Convert to numpy array

        self.num_c = len(self.species)
        self.rho_b = self.rho_c*self.Dcat*(1-self.eps) #[Kgcat/m3R] bed density    
        
        self.correlation = GasMixtureCorrelations(self.species, self.database)    
        
        self.num_r_perm = np.round(self.r_max_perm / self.r_max *self.num_r).astype('int') + 3
        self.num_r_ret = self.num_r - self.num_r_perm
        if (self.num_r_ret < 2):
            self.num_r_ret = 2
            self.num_r_perm = self.num_r - self.num_r_ret

        self.create_spatial_discretization()

        # membrane permeabilities
        perm_dict= {'NH3': self.Perm_NH3, 'H2': self.Perm_NH3 / self.Sel_am_hy, 'N2': self.Perm_NH3 / self.Sel_am_ni}
        perm = np.zeros((1, self.num_c))
        for i, species in enumerate(self.species):
            perm[0, i] = perm_dict[species]
        perm = np.broadcast_to(perm, (self.num_z, self.num_c))
        fltr = (self.z_c <= self.Lsealing) 
        if (any(fltr)):
            perm = perm.copy()
            perm[fltr,:] = 0
        self.perm = perm

        # Bed and Catalyst Density
        self.rho_b = self.rho_c * self.Dcat * (1 - self.eps)

        # reactor geometry
        #self.Pm = np.pi * self.r_min  # Membrane circumference [m]
        #self.Vtot = self.Ab * self.L  # Total reactor volume (excluding permeate) [m³]

        # Reactor Flow Conditions
        r_f_ret = self.r_f_perm/self.r_max_perm
        self.flux_perm_in = (self.F_perm_in / (np.pi * self.r_max_perm**2)) * (2.0-r_f_ret[1:]*r_f_ret[1:]-r_f_ret[0:-1]*r_f_ret[0:-1]).reshape((1,-1,1))*self.y_perm_in
        self.flux_ret_in = np.broadcast_to(np.asarray(self.F_ret_in / (np.pi * (self.r_max**2 - self.r_min**2))).reshape((1,-1,1))*self.y_ret_in, shape= (1,self.num_r_ret, self.num_c))
              
        rho_g = self.correlation.density(self.y_ret_init, self.T_ret_init, self.p_ret_out)  # Gas density [kg/m³]
        viscosity = self.correlation.viscosity(self.y_ret_init, self.T_ret_init)  # Gas viscosity [Pa s]
        D = self.correlation.diffusion(self.y_ret_init, self.T_ret_init, self.p_ret_out)  # Gas viscosity [Pa s]        
        # Reynolds and Schmidt Numbers
        
        mass_flux = self.correlation.molecular_weight(self.flux_perm_in)  # Molecular weight [kg/mol]
        self.Re = mass_flux  * 2.0 * self.r_max / viscosity
        self.Sc = viscosity / rho_g / np.mean(D)
        self.ReSc = self.Re * self.Sc
        self.rL = self.r_max / self.L
        self.Lr = self.L / self.r_max

    def create_spatial_discretization(self):
        # Time-Stepping
        #self.num_time_steps = int(np.round(self.L / self.u0 / self.dt * 5))
        # create grid
        # self.r_f_ret = np.linspace(self.r_min, self.r_max, self.num_r+1)
        self.r_f_ret = non_uniform_grid(self.r_min, self.r_max, self.num_r_ret+1, (self.r_max - self.r_min)/np.maximum(self.num_r_ret-10, 0.8*self.num_r_ret), 1.2)
        self.r_c_ret = 0.5 * (self.r_f_ret[0:-1] + self.r_f_ret[1:])
        # self.r_f_perm = np.linspace(self.r_min_perm, self.r_max_perm, self.num_r_perm+1)
        self.r_f_perm = non_uniform_grid(self.r_min_perm, self.r_max_perm, self.num_r_perm+1, (self.r_max_perm - self.r_min_perm)/np.maximum(self.num_r_perm-10, 0.8*self.num_r_perm), 1.0/1.2)
        self.r_c_perm = 0.5 * (self.r_f_perm[0:-1] + self.r_f_perm[1:])
        # uniform until sealing, then non-uniform
        num_z_sealing = int(np.round(self.Lsealing / self.L * self.num_z))
        z_f_uniform = np.linspace(0, self.Lsealing, num_z_sealing+1)     
        z_f_non_uniform = non_uniform_grid(self.Lsealing, self.L, self.num_z+1-num_z_sealing, (self.L - self.Lsealing)/np.maximum(self.num_z-num_z_sealing-8, 0.8*(self.num_z-num_z_sealing)), 1.2)
        self.z_f = np.concatenate((z_f_uniform, z_f_non_uniform[1:]), axis=0)
        #self.z_f = np.linspace(0, self.L, self.num_z+1)
        #self.z_f = non_uniform_grid(0, self.L, self.num_z+1, self.L / np.maximum(self.num_z-10, 0.8*self.num_z), 1.2)
        self.z_c = 0.5 * (self.z_f[0:-1] + self.z_f[1:])

    def init_fields(self, c=None, p=None, T=None):
        """
        Initialize the concentration field.
        """
        shape_c_p = (self.num_z, self.num_r, self.num_c+1)
        shape_c = (self.num_z, self.num_r, self.num_c)
        shape_p = (self.num_z, self.num_r)
        shape_T = (self.num_z, self.num_r)
        
        self.c_p = np.empty(shape_c_p)
        if c is None:
            c = self.c_p[..., :-1]
            c_ret = c[:, self.num_r_perm:, :]
            c_ret_0 = self.correlation.molar_density(self.y_ret_init, self.T_ret_init, self.p_ret_out)*self.y_ret_init
            c_ret[...] = np.broadcast_to(c_ret_0, c_ret.shape)
            c_perm = c[:, :self.num_r_perm, :]
            c_perm_0 = self.correlation.molar_density(self.y_perm_init, self.T_perm_init, self.p_perm_out)*self.y_perm_init
            c_perm[...] = np.broadcast_to(c_perm_0.reshape((1,1,-1)), c_perm.shape)
        else:
            self.c_p[...,:-1] = np.broadcast_to(np.array(c), shape_c).copy()
            c_ret = self.c[:, self.num_r_perm:, :]
            c_perm = self.c[:, :self.num_r_perm, :]
        
        self.c_ret_ax = interp_cntr_to_stagg(c_ret, x_f=self.z_f, x_c=self.z_c, axis=0)
        self.c_ret_rad = interp_cntr_to_stagg(c_ret, x_f=self.r_f_ret, x_c=self.r_c_ret, axis=1)
        self.c_perm_ax = interp_cntr_to_stagg(c_perm, x_f=self.z_f, x_c=self.z_c, axis=0)
        self.c_perm_rad = interp_cntr_to_stagg(c_perm, x_f=self.r_f_perm, x_c=self.r_c_perm, axis=1)

        self.g_react_source = np.zeros(c_ret.shape)
        
        if p is None:
            p = self.c_p[:, :, -1]
            p_ret = p[:, self.num_r_perm:]
            p_ret[:,:] = np.broadcast_to(np.array(self.p_ret_out).reshape((1,1)), p_ret.shape)
            p_perm = p[:, :self.num_r_perm]
            p_perm[:,:] = np.broadcast_to(np.array(self.p_perm_out).reshape((1,1)), p_perm.shape)            
        else:
            self.c_p[...,-1] = np.broadcast_to(np.array(p), shape_p).copy()
            
        if T is None:
            self.T = np.empty(shape_T)
            T_ret = self.T[:, self.num_r_perm:]
            T_ret[:,:] = np.broadcast_to(np.array(self.T_ret_init).reshape((1,1)), T_ret.shape)         
            T_perm = self.T[:, :self.num_r_perm]
            T_perm[:,:] = np.broadcast_to(np.array(self.T_perm_init).reshape((1,1)), T_perm.shape)
        else:
            self.T = np.broadcast_to(np.array(T), shape_T).copy()
            
        c = self.c_p[...,:-1]
        c_perm = np.sum(c[:, 0:self.num_r_perm, :], axis=-1)
        c_perm_ax = interp_cntr_to_stagg(c_perm, self.z_f, self.z_c, axis=0)
        c_ret = np.sum(c[:, self.num_r_perm:, :], axis=-1)
        c_ret_ax = interp_cntr_to_stagg(c_ret, self.z_f, self.z_c, axis=0)
        self.u_perm_ax = np.sum(self.flux_perm_in[0,:,:],axis=-1)/c_perm_ax
        self.u_ret_ax = np.sum(self.flux_ret_in[0,:,:],axis=-1)/c_ret_ax
        if (self.is_counter_current):
            self.u_ret_ax = -self.u_ret_ax
        self.u_perm_rad = np.zeros((self.num_z, self.num_r_perm+1))
        self.u_ret_rad = np.zeros((self.num_z, self.num_r_ret+1))
        self.div_u = np.zeros((self.num_z, self.num_r))

        self.cnt_c_p = 0
        self.cnt_T = 0

        return self.c_p, self.T

    def init_jac(self):
        """
        Initialize the Jacobian matrices for the system. 
        This includes setting up the accumulation, convection, and diffusion terms.
        """
        shape_c = (self.num_z, self.num_r, self.num_c)
        shape_c_ret = (self.num_z, self.num_r_ret, self.num_c)
        shape_c_perm = (self.num_z, self.num_r_perm, self.num_c)
        shape_p = (self.num_z, self.num_r)
        shape_p_ret = (self.num_z, self.num_r_ret)
        shape_p_perm = (self.num_z, self.num_r_perm)
       
        bc_none = {'a': 0, 'b': 0, 'd': 0}      
        bc_dirichlet_hom = {'a': 0, 'b':1 , 'd' : 0}
        bc_neumann_hom = {'a': 1, 'b': 0, 'd': 0}
        bc_dirichlet = {'a': 0, 'b':1 , 'd' : 1}
        bc_neumann = {'a': 1, 'b':0 , 'd' : 1}
        if (self.is_counter_current):
            bc_ret_ax = (bc_neumann_hom, bc_none)
            bc_p_ret_ax = (bc_dirichlet, bc_none)
            bc_T_ret_ax = (bc_neumann_hom, bc_dirichlet)
        else:
            bc_ret_ax=(bc_none, bc_neumann_hom)  
            bc_p_ret_ax = (bc_none, bc_dirichlet)  
            bc_T_ret_ax = (bc_dirichlet, bc_neumann_hom)
        
        jac_c_accum_perm = construct_coefficient_matrix(1.0, shape_c_perm)
        jac_c_accum_ret = construct_coefficient_matrix(self.eps, shape_c_ret)
        self.div_c_perm_ax  = construct_div(shape_c_perm, self.z_f, nu=0, axis=0)
        self.div_c_perm_rad = construct_div(shape_c_perm, self.r_f_perm, nu=self.nu, axis=1)
        self.div_c_ret_ax  = construct_div(shape_c_ret, self.z_f, nu=0, axis=0)
        self.div_c_ret_rad = construct_div(shape_c_ret, self.r_f_ret, nu=self.nu, axis=1)
        self.grad_c_perm_ax, self.grad_bc_c_perm_ax   = construct_grad(shape_c_perm, self.z_f, self.z_c, bc=(bc_none, bc_neumann_hom), axis=0)
        self.grad_c_perm_rad, _ = construct_grad(shape_c_perm, self.r_f_perm, self.r_c_perm, bc=(bc_neumann_hom, bc_none), axis=1)
        self.grad_c_ret_ax, self.grad_bc_c_ret_ax   = construct_grad(shape_c_ret, self.z_f, self.z_c, bc_ret_ax, axis=0)
        self.grad_c_ret_rad, _ = construct_grad(shape_c_ret, self.r_f_ret, self.r_c_ret, bc=(bc_none, bc_neumann_hom), axis=1)
        self.c_matrix_perm_mem, _ = construct_boundary_value_matrices(shape_c_perm, self.r_f_perm, self.r_c_perm, bc=None, bound_id=1, axis=1)
        self.c_matrix_ret_mem, _ = construct_boundary_value_matrices(shape_c_ret, self.r_f_ret, self.r_c_ret, bc=None, bound_id=0, axis=1)

        self.div_p_perm_ax  = construct_div(shape_p_perm, self.z_f, nu=0, axis=0)
        self.div_p_perm_rad = construct_div(shape_p_perm, self.r_f_perm, nu=self.nu, axis=1)
        self.div_p_ret_ax  = construct_div(shape_p_ret, self.z_f, nu=0, axis=0)
        self.div_p_ret_rad = construct_div(shape_p_ret, self.r_f_ret, nu=self.nu, axis=1)
        self.grad_p_perm_ax, self.grad_bc_p_perm_ax   = construct_grad(shape_p_perm, self.z_f, self.z_c, bc=(bc_none, bc_dirichlet), axis=0)
        self.grad_bc_p_perm_ax *= self.p_perm_out
        self.grad_p_perm_rad, _ = construct_grad(shape_p_perm, self.r_f_perm, self.r_c_perm, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
        self.grad_p_ret_ax, self.grad_bc_p_ret_ax = construct_grad(shape_p_ret, self.z_f, self.z_c, bc=bc_p_ret_ax, axis=0)
        self.grad_bc_p_ret_ax *= self.p_ret_out
        self.grad_p_ret_rad, _ = construct_grad(shape_p_ret, self.r_f_ret, self.r_c_ret, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
        
        self.jac_T_accum = construct_coefficient_matrix(1.0, shape_p)
        self.grad_T_perm_ax, self.grad_bc_T_perm_ax   = construct_grad(shape_p_perm, self.z_f, self.z_c, bc=(bc_dirichlet, bc_neumann_hom), axis=0)
        self.grad_bc_T_perm_ax *= self.T_perm_in
        self.grad_T_perm_rad, _, self.grad_bc_T_perm_rad = construct_grad(shape_p_perm, self.r_f_perm, self.r_c_perm, bc=(bc_neumann_hom, bc_dirichlet), axis=1, shapes_d = (None, (self.num_z, 1)))
        self.grad_T_ret_ax, self.grad_bc_T_ret_ax   = construct_grad(shape_p_ret, self.z_f, self.z_c, bc_T_ret_ax, axis=0)
        self.grad_bc_T_ret_ax *= self.T_ret_in
        self.grad_T_ret_rad, self.grad_bc_T_ret_rad, _ = construct_grad(shape_p_ret, self.r_f_ret, self.r_c_ret, bc=(bc_dirichlet, bc_neumann_hom), axis=1, shapes_d = ((self.num_z, 1), None))
        
        
        # shift retentate-side matrix indices to (later) build a monolithic spatial discretization
        offset = (0, self.num_r_perm, 0)
        offset_p = (0, self.num_r_perm)
        jac_c_accum_perm = update_csc_array_indices(jac_c_accum_perm, shape_c_perm, shape_c)
        jac_c_accum_ret = update_csc_array_indices(jac_c_accum_ret, shape_c_ret, shape_c, offset=offset)
        self.jac_c_accum = jac_c_accum_perm + jac_c_accum_ret
        self.div_c_ret_ax    = update_csc_array_indices(self.div_c_ret_ax, (shape_c_ret,None), (shape_c,None), offset=(offset,None))
        self.div_c_ret_rad   = update_csc_array_indices(self.div_c_ret_rad, (shape_c_ret,None), (shape_c,None), offset=(offset,None))
        self.div_c_perm_ax    = update_csc_array_indices(self.div_c_perm_ax, (shape_c_perm,None), (shape_c,None))
        self.div_c_perm_rad   = update_csc_array_indices(self.div_c_perm_rad, (shape_c_perm,None), (shape_c,None))
        self.grad_c_ret_ax   = update_csc_array_indices(self.grad_c_ret_ax, (None, shape_c_ret), (None, shape_c), offset=(None, offset))
        self.grad_c_ret_rad  = update_csc_array_indices(self.grad_c_ret_rad, (None, shape_c_ret), (None, shape_c), offset=(None, offset))
        self.grad_c_perm_ax   = update_csc_array_indices(self.grad_c_perm_ax, (None, shape_c_perm), (None, shape_c))
        self.grad_c_perm_rad  = update_csc_array_indices(self.grad_c_perm_rad, (None, shape_c_perm), (None, shape_c))
        self.c_matrix_ret_mem = update_csc_array_indices(self.c_matrix_ret_mem, (None, shape_c_ret), (None, shape_c), offset=(None, offset))
        self.c_matrix_perm_mem = update_csc_array_indices(self.c_matrix_perm_mem, (None, shape_c_perm), (None, shape_c))

        self.div_p_ret_ax    = update_csc_array_indices(self.div_p_ret_ax, (shape_p_ret,None), (shape_p,None), offset=(offset_p,None))
        self.div_p_ret_rad   = update_csc_array_indices(self.div_p_ret_rad, (shape_p_ret,None), (shape_p,None), offset=(offset_p,None))
        self.div_p_perm_ax    = update_csc_array_indices(self.div_p_perm_ax, (shape_p_perm,None), (shape_p,None))
        self.div_p_perm_rad   = update_csc_array_indices(self.div_p_perm_rad, (shape_p_perm,None), (shape_p,None))
        self.grad_p_ret_ax   = update_csc_array_indices(self.grad_p_ret_ax, (None, shape_p_ret), (None, shape_p), offset=(None, offset_p))
        self.grad_p_ret_rad  = update_csc_array_indices(self.grad_p_ret_rad, (None, shape_p_ret), (None, shape_p), offset=(None, offset_p))
        self.grad_p_perm_ax   = update_csc_array_indices(self.grad_p_perm_ax, (None, shape_p_perm), (None, shape_p))
        self.grad_p_perm_rad  = update_csc_array_indices(self.grad_p_perm_rad, (None, shape_p_perm), (None, shape_p))

        self.grad_T_ret_ax   = update_csc_array_indices(self.grad_T_ret_ax, (None, shape_p_ret), (None, shape_p), offset=(None, offset_p))
        self.grad_T_ret_rad  = update_csc_array_indices(self.grad_T_ret_rad, (None, shape_p_ret), (None, shape_p), offset=(None, offset_p))
        self.grad_T_perm_ax   = update_csc_array_indices(self.grad_T_perm_ax, (None, shape_p_perm), (None, shape_p))
        self.grad_T_perm_rad  = update_csc_array_indices(self.grad_T_perm_rad, (None, shape_p_perm), (None, shape_p))
        self.construct_darcy_matrices()

        self.numjac = NumJac(shape_c_ret)
        self.numjac_p = NumJac(shape_p + (1,))
        self.sum_c = construct_coefficient_matrix(np.array([[[1.0]]]), shape=(shape_p+(1,), shape_c))
        self.g_c_in = self.div_c_perm_ax[:,0:self.flux_perm_in.size] @ self.flux_perm_in.ravel()
        if (self.is_counter_current):
            self.g_c_in -= self.div_c_ret_ax[:,-self.flux_ret_in.size:] @ self.flux_ret_in.ravel()
        else:
            self.g_c_in += self.div_c_ret_ax[:,0:self.flux_ret_in.size] @ self.flux_ret_in.ravel()

        self._jac = None
            
    def split_perm_and_ret(self, c):
        """Split full field into permeate and retentate views.

        Args:
            c (ndarray): Field.

        Returns:
            tuple: (c_perm, c_ret) views/slices.
        """
        c = np.asarray(c)
        if c.ndim > 1 and c.shape[1] == self.num_r:
            c_perm = c[:, 0:self.num_r_perm, ...]
            c_ret = c[:, self.num_r_perm:self.num_r, ...]
        else:
            c_perm = c
            c_ret = c
        return c_perm, c_ret

    def construct_darcy_matrices(self, c=None, T=None, p=None):
        """Assemble permeability-weighted matrices for velocity (Darcy / Ergun).

        Returns:
            csc_matrix|None: Pressure Jacobian contribution if compute_jac.
        """
        if c is None:
            c = self.c_p[...,:-1]
        if T is None:
            T = self.T
        if p is None:
            p = self.c_p[...,-1]
        c_perm, c_ret = self.split_perm_and_ret(c)
        T_perm, T_ret = self.split_perm_and_ret(T)
        p_perm, p_ret = self.split_perm_and_ret(p)

        viscosity = self.correlation.viscosity(c_perm, T_perm)
        k_field =  0.25*(self.r_max_perm**2-self.r_c_perm**2).reshape((1, -1))/viscosity
        k_field_perm_ax = interp_cntr_to_stagg(k_field, x_f=self.z_f, x_c=self.z_c, axis=0)
        self.k_matrix_perm_ax = construct_coefficient_matrix(k_field_perm_ax, (self.num_z, self.num_r_perm), axis=0)       
        k_field   = 1e1*self.r_max_perm**2/viscosity
        k_field_perm_rad = interp_cntr_to_stagg(k_field, x_f=self.r_f_perm, x_c=self.r_c_perm, axis=1)
        self.k_matrix_perm_rad = construct_coefficient_matrix(k_field_perm_rad, (self.num_z, self.num_r_perm), axis=1)
        
        viscosity = self.correlation.viscosity(c_ret, T_ret)
        rho = self.correlation.density(c_ret, T_ret, p_ret)
        u_ax_abs = np.abs(interp_stagg_to_cntr(self.u_ret_ax, self.z_f, self.z_c, axis=0))
        beta_0 = 150.0 * (1-self.eps)**2*viscosity / (self.eps**3 * self.dp**2) 
        beta_1 = 1.75 * rho *(1-self.eps) * np.abs(u_ax_abs) / (self.eps**3 * self.dp)
        beta_inv = 1.0/(beta_0 + beta_1)
        shape_p_ret = (self.num_z, self.num_r_ret)
        k_field_ret_ax = interp_cntr_to_stagg(beta_inv, x_f=self.z_f, x_c=self.z_c, axis=0)
        self.k_matrix_ret_ax = construct_coefficient_matrix(k_field_ret_ax, shape_p_ret, axis=0)
        k_field_ret_rad = interp_cntr_to_stagg(beta_inv, x_f=self.r_f_ret, x_c=self.r_c_ret, axis=1)
        self.k_matrix_ret_rad = construct_coefficient_matrix(k_field_ret_rad, shape_p_ret, axis=1)

    def construct_jac_darcy(self):
        """Assemble permeability-weighted matrices for velocity (Darcy / Ergun).

        Returns:
            csc_matrix|None: Pressure Jacobian contribution if compute_jac.
        """
        shape_p_perm = (self.num_z, self.num_r_perm, 1)
        shape_c_perm = (self.num_z, self.num_r_perm, self.num_c)
        ck_matrix = construct_coefficient_matrix(self.c_perm_ax, shape=(shape_c_perm, shape_p_perm), axis=0) @ self.k_matrix_perm_ax
        jac_darcy = self.div_c_perm_ax @ ((-ck_matrix) @ self.grad_p_perm_ax)
        ck_matrix = construct_coefficient_matrix(self.c_perm_rad, shape=(shape_c_perm, shape_p_perm), axis=1) @ self.k_matrix_perm_rad
        jac_darcy += self.div_c_perm_rad @ ((-ck_matrix) @ self.grad_p_perm_rad)
        shape_p_ret = (self.num_z, self.num_r_ret, 1)
        shape_c_ret = (self.num_z, self.num_r_ret, self.num_c)
        ck_matrix = construct_coefficient_matrix(self.c_ret_ax, shape=(shape_c_ret, shape_p_ret), axis=0) @ self.k_matrix_ret_ax
        jac_darcy += self.div_c_ret_ax @ ((-ck_matrix) @ self.grad_p_ret_ax)
        ck_matrix = construct_coefficient_matrix(self.c_ret_rad, shape=(shape_c_ret, shape_p_ret), axis=1) @ self.k_matrix_ret_rad
        jac_darcy += self.div_c_ret_rad @ ((-ck_matrix) @ self.grad_p_ret_rad)
        return jac_darcy

    def construct_g_diff(self, c = None, T = None, p = None, compute_jac = False):
        """
        Update the transport coefficients based on the current concentration field.

        Parameters:
        - c (numpy.ndarray): Current concentration field.
        """
        shape_c_ret = (self.num_z, self.num_r_ret, self.num_c)
        shape_c_perm = (self.num_z, self.num_r_perm, self.num_c)
        if c is None:
            c = self.c_p[...,:-1]
        if T is None:
            T = self.T
        if p is None:
            p = self.c_p[...,-1]    
        c_perm, c_ret = self.split_perm_and_ret(c)
        T_perm, T_ret = self.split_perm_and_ret(T)
        p_perm, p_ret = self.split_perm_and_ret(p)
    
       # Retentate side
        g = np.empty(c.shape)
        g_vect = g.reshape((-1,1))
        if (compute_jac or not hasattr(self, 'jac_c_diff')):
            y_ret = c_ret/np.sum(c_ret, axis=-1, keepdims=True)  # Mole fractions
            diff_field_ret = self.correlation.diffusion(y_ret, T_ret, p_ret)
            diff_field_ret_ax = interp_cntr_to_stagg(diff_field_ret, x_f=self.z_f, x_c=self.z_c, axis=0)
            diff_matrix_ret_ax = construct_coefficient_matrix(diff_field_ret_ax, shape_c_ret, axis=0)
            diff_field_ret_rad = interp_cntr_to_stagg(diff_field_ret, x_f=self.r_f_ret, x_c=self.r_c_ret, axis=1)
            diff_matrix_ret_rad = construct_coefficient_matrix(diff_field_ret_rad, shape_c_ret, axis=1)

            y_perm = c_perm/np.sum(c_perm, axis=-1, keepdims=True)  # Mole fractions
            diff_field_perm = self.correlation.diffusion(y_perm, T_perm, p_perm)
            diff_field_perm_ax = interp_cntr_to_stagg(diff_field_perm, x_f=self.z_f, x_c=self.z_c, axis=0)
            diff_matrix_perm_ax = construct_coefficient_matrix(diff_field_perm_ax, shape_c_perm, axis=0)
            diff_field_perm_rad = interp_cntr_to_stagg(diff_field_perm, x_f=self.r_f_perm, x_c=self.r_c_perm, axis=1)
            diff_matrix_perm_rad = construct_coefficient_matrix(diff_field_perm_rad, shape_c_perm, axis=1)
            
            # test: axial dispersion zero
            diff_matrix_perm_ax *= 0
            diff_matrix_ret_ax *= 0
            
            self.jac_c_diff = self.div_c_ret_ax @ (-diff_matrix_ret_ax) @ self.grad_c_ret_ax + self.div_c_ret_rad @ (-diff_matrix_ret_rad) @ self.grad_c_ret_rad + self.div_c_perm_ax @ (-diff_matrix_perm_ax) @ self.grad_c_perm_ax + self.div_c_perm_rad @ (-diff_matrix_perm_rad) @ self.grad_c_perm_rad 
            self.g_bc_c_diff = self.div_c_ret_ax @ ((-diff_matrix_ret_ax) @ self.grad_bc_c_ret_ax) +self.div_c_perm_ax @ ((-diff_matrix_perm_ax) @ self.grad_bc_c_perm_ax)

            #jac_ic_c_diff_perm = self.div_c_perm_rad @ ((-diff_matrix_perm_rad) @ self.grad_bc_c_perm_rad)
            #jac_ic_c_diff_ret = self.div_c_ret_rad @ ((-diff_matrix_ret_rad) @ self.grad_bc_c_ret_rad)
            #jac_c_diff = self.div_c_ret_ax @ self.grad_c_ret_ax + self.div_c_ret_rad @ self.grad_c_ret_rad + self.div_c_perm_ax @ self.grad_c_perm_ax + self.div_c_perm_rad @ self.grad_c_perm_rad 
            #jac_bc_c_diff = self.div_c_ret_ax @ self.grad_bc_c_ret_ax + self.div_c_ret_rad @ self.grad_bc_c_ret_rad +self.div_c_perm_ax @ self.grad_bc_c_perm_ax + self.div_c_perm_rad @ self.grad_bc_c_perm_rad
            if (T_perm.ndim > 1):
                T_perm_mem,_ = compute_boundary_values(T_perm, self.r_f_perm, self.r_c_perm, bc=None, axis=1, bound_id = 1)
                T_ret_mem,_ = compute_boundary_values(T_ret, self.r_f_ret, self.r_c_ret, bc=None, axis=1, bound_id = 0)
                P_matrix_perm_mem = construct_coefficient_matrix(self.Rg*T_perm_mem[:,0,np.newaxis]*self.perm)
                P_matrix_ret_mem = construct_coefficient_matrix(self.Rg*T_ret_mem[:,0,np.newaxis]*self.perm)
            else:
                P_perm_mem = np.broadcast_to((self.Rg*T_perm*self.perm).reshape((1,-1)), (self.num_z, self.num_c))
                P_ret_mem = np.broadcast_to((self.Rg*T_ret*self.perm).reshape((1,-1)), (self.num_z, self.num_c))
                P_matrix_perm_mem = construct_coefficient_matrix(P_perm_mem)
                P_matrix_ret_mem = construct_coefficient_matrix(P_ret_mem)

            flux_matrix_perm_mem = P_matrix_perm_mem @ self.c_matrix_perm_mem - P_matrix_ret_mem @ self.c_matrix_ret_mem
            factor_geom = (self.r_f_perm[-1]/self.r_f_ret[0])**self.nu
            flux_matrix_ret_mem = factor_geom * flux_matrix_perm_mem
            flux_matrix_perm_mem = update_csc_array_indices(flux_matrix_perm_mem, ((self.num_z, 1, self.num_c), None), ((self.num_z, self.num_r_perm+1, self.num_c), None), offset = ((0, self.num_r_perm,0), None))
            flux_matrix_ret_mem = update_csc_array_indices(flux_matrix_ret_mem, ((self.num_z, 1, self.num_c), None), ((self.num_z, self.num_r_ret+1, self.num_c),None))
            self.jac_c_diff += self.div_c_ret_rad @ flux_matrix_ret_mem + self.div_c_perm_rad @ flux_matrix_perm_mem

        g_vect[...] = self.g_bc_c_diff + self.jac_c_diff @ c.reshape((-1,1))
        return g, self.jac_c_diff
    
    def update_velocity_fields(self, p=None):
        """Update staggered velocity arrays from current pressure gradients.

        Returns:
            tuple: (u_perm_ax, u_perm_rad, u_ret_ax, u_ret_rad)
        """
        vel_matrix_perm_ax = (-self.k_matrix_perm_ax) @ self.grad_p_perm_ax
        vel_bc_perm_ax = (-self.k_matrix_perm_ax) @ self.grad_bc_p_perm_ax
        vel_matrix_perm_rad = (-self.k_matrix_perm_rad) @ self.grad_p_perm_rad
        vel_matrix_ret_ax = (-self.k_matrix_ret_ax) @ self.grad_p_ret_ax
        vel_bc_ret_out = -(self.k_matrix_ret_ax @ self.grad_bc_p_ret_ax)
        vel_matrix_ret_rad = (-self.k_matrix_ret_rad) @ self.grad_p_ret_rad
        if (p is None):
            p_vec = self.c_p[...,-1].reshape((-1,1))
        else:
            p_vec = p.reshape((-1,1))
        self.u_perm_ax.reshape((-1,1))[...] = vel_matrix_perm_ax @ p_vec + vel_bc_perm_ax
        self.u_perm_rad.reshape((-1,1))[...] = vel_matrix_perm_rad @ p_vec
        self.u_ret_ax.reshape((-1,1))[...] = vel_matrix_ret_ax @ p_vec + vel_bc_ret_out
        self.u_ret_rad.reshape((-1,1))[...] = vel_matrix_ret_rad @ p_vec

        self.u_perm_ax[0,:] = self.u_perm_ax[1,:] - (self.z_f[1]-self.z_f[0])/(self.z_f[2]-self.z_f[1])*(self.u_perm_ax[2,:]-self.u_perm_ax[1,:])
        if (self.is_counter_current):
            self.u_ret_ax[-1,:] = self.u_ret_ax[-2,:] - (self.z_f[-2]-self.z_f[-1])/(self.z_f[-3]-self.z_f[-2])*(self.u_ret_ax[-3,:]-self.u_ret_ax[-2,:])
        else:
            self.u_ret_ax[0,:] = self.u_ret_ax[1,:] - (self.z_f[1]-self.z_f[0])/(self.z_f[2]-self.z_f[1])*(self.u_ret_ax[2,:]-self.u_ret_ax[1,:])        

        self.div_u = (self.div_p_perm_ax @ self.u_perm_ax.ravel() + self.div_p_perm_rad @ self.u_perm_rad.ravel() 
                      + self.div_p_ret_ax @ self.u_ret_ax.ravel() + self.div_p_ret_rad @ self.u_ret_rad.ravel()).reshape(self.T.shape)

        return self.u_perm_ax, self.u_perm_rad, self.u_ret_ax, self.u_ret_rad
    
    def construct_g_conv(self, c=None, compute_jac = False):
        """Assemble convective species transport residual using upwind/TVD.

        Args:
            c (ndarray|None): Concentration field (optional, for shape only).
            compute_jac (bool): If True, compute Jacobian sparsity pattern.

        Returns:
            tuple: (g, jac) residual and Jacobian (or None if compute_jac=False)
        """
        if c is None:
            c = self.c
        bc_neumann_hom = {'a': 1, 'b': 0, 'd': 0}
        bc_none = {'a': 0, 'b': 0, 'd': 0}
        if (self.is_counter_current):
            bc_ret_ax = (bc_neumann_hom, bc_none)
            is_inflow = self.u_ret_ax[0,:] > 0 
            if (np.any(is_inflow)):
                b_out = (is_inflow*1.0).reshape((1,-1,1))
                a_out = (1.0-b_out)
                d_out = b_out*self.p_ret_out/(self.Rg*self.T_ret_in)*np.array([[[0.0,1.0,0.0]]])
                bc_ret_ax = ({'a':a_out, 'b':b_out, 'd':d_out},bc_none)
        else:
            bc_ret_ax = (bc_none, bc_neumann_hom)
            is_inflow = self.u_ret_ax[-1,:] < 0
            if (np.any(is_inflow)):
                b_out = (is_inflow*1.0).reshape((1,-1,1))
                a_out = (1.0-b_out)
                d_out = b_out*self.p_ret_out/(self.Rg*self.T_ret_in)*np.array([[[0.0,1.0,0.0]]])
                bc_ret_ax = (bc_none, {'a':a_out, 'b':b_out, 'd':d_out})
        
        g = np.empty(c.shape)
        g_vect = g.ravel()
        
        c_perm = c[:, 0:self.num_r_perm, :]
        u_perm_ax = self.u_perm_ax[...,np.newaxis]
        u_perm_rad = self.u_perm_rad[...,np.newaxis]
        self.c_perm_ax,_ = interp_cntr_to_stagg_tvd(c_perm, self.z_f, self.z_c, bc = (bc_none, bc_neumann_hom), v = u_perm_ax, tvd_limiter = upwind, axis=0)
        flux_perm_ax = u_perm_ax * self.c_perm_ax
        g_vect[:] = self.div_c_perm_ax @ flux_perm_ax.ravel()
        self.c_perm_rad,_ = interp_cntr_to_stagg_tvd(c_perm, self.r_f_perm, self.r_c_perm, bc = (bc_neumann_hom, bc_neumann_hom), v = u_perm_rad, tvd_limiter = upwind, axis=1)
        flux_perm_rad = u_perm_rad * self.c_perm_rad
        g_vect[:] += self.div_c_perm_rad @ flux_perm_rad.ravel()
        
        c_ret = c[:, self.num_r_perm:, :]
        u_ret_ax = self.u_ret_ax[...,np.newaxis]
        u_ret_rad = self.u_ret_rad[...,np.newaxis]
        self.c_ret_ax,_ = interp_cntr_to_stagg_tvd(c_ret, self.z_f, self.z_c, bc = bc_ret_ax, v = u_ret_ax, tvd_limiter = upwind, axis=0)
        flux_ret_ax = u_ret_ax * self.c_ret_ax
        g_vect[:] += self.div_c_ret_ax @ flux_ret_ax.ravel()
        
        self.c_ret_rad,_ = interp_cntr_to_stagg_tvd(c_ret, self.r_f_ret, self.r_c_ret, bc = (bc_neumann_hom, bc_neumann_hom), v = u_ret_rad, tvd_limiter = upwind, axis=1)
        flux_ret_rad = u_ret_rad * self.c_ret_rad
        g_vect[:] += self.div_c_ret_rad @ flux_ret_rad.ravel()        
        
        if compute_jac:
            conv_matrix_perm_ax, _ = construct_convflux_upwind(c_perm.shape, self.z_f, self.z_c, bc = (bc_none, bc_neumann_hom), v = u_perm_ax, axis=0)
            jac_perm = self.div_c_perm_ax @ conv_matrix_perm_ax
            conv_matrix_perm_rad, _ = construct_convflux_upwind(c_perm.shape, self.r_f_perm, self.r_c_perm, bc = (bc_neumann_hom, bc_neumann_hom), v = u_perm_rad, axis=1)
            jac_perm += self.div_c_perm_rad @ conv_matrix_perm_rad
            conv_matrix_ret_ax, _ = construct_convflux_upwind(c_ret.shape, self.z_f, self.z_c, bc = bc_ret_ax, v = u_ret_ax, axis=0)
            jac_ret = self.div_c_ret_ax @ conv_matrix_ret_ax
            conv_matrix_ret_rad, _ = construct_convflux_upwind(c_ret.shape, self.r_f_ret, self.r_c_ret, bc = (bc_neumann_hom, bc_neumann_hom), v= u_ret_rad, axis=1)
            jac_ret += self.div_c_ret_rad @ conv_matrix_ret_rad
            jac_perm = update_csc_array_indices(jac_perm, (None, c_perm.shape), (None, c.shape))
            jac_ret = update_csc_array_indices(jac_ret, (None, c_ret.shape), (None, c.shape), offset=(None, (0,self.num_r_perm,0)))
            jac = jac_perm + jac_ret
            return g, jac
        else:
            return g, None
           
    def construct_g_c(self, c=None, c_old=None, compute_jac=False, kinetics_as_source = False):
        """
        Construct the residual vector g and the Jacobian matrix for the system.

        Parameters:
        - c (numpy.ndarray): Current concentration field.
        - c_old (numpy.ndarray): Previous concentration field.

        Returns:
        - g (numpy.ndarray): Residual vector.
        - Jac (scipy.sparse.csc_matrix): Jacobian matrix.
        """
        if (c is None):
            c = self.c

        c_ret = c[:, self.num_r_perm:, :]
        c_tot_ret = np.sum(c_ret, axis=-1, keepdims=True)
        #_, c_tot_ret =  self.split_perm_and_ret(c_tot)
        _, p_ret =  self.split_perm_and_ret(self.p)
        
        g_conv, jac_conv = self.construct_g_conv(c, compute_jac=compute_jac)
        g_diff, jac_diff = self.construct_g_diff(c, compute_jac=compute_jac)
        g = self.g_c_in.reshape(c.shape) + g_conv + g_diff
        if (c_old is not None):
            g += (self.jac_c_accum @ ((c-c_old).reshape((-1, 1))/self.dt)).reshape(c.shape)
        #p_over_c_tot = p_ret[..., np.newaxis]/c_tot_ret[..., np.newaxis]
        p_over_c_tot = p_ret[..., np.newaxis]/c_tot_ret
        if compute_jac:
            self._jac = jac_conv + jac_diff
            if (c_old is not None):
                self._jac += (1.0/self.dt)*self.jac_c_accum
            if not kinetics_as_source:
                g_react, jac_react = self.numjac(lambda c: self.factor_react*self.kinetics(c*p_over_c_tot), c_ret)
                shape_c_ret = c_ret.shape
                offset = (0, self.num_r_perm, 0)
                jac_react = update_csc_array_indices(jac_react, shape_c_ret, c.shape, offset=offset)
                self._jac -= jac_react
        elif not kinetics_as_source:
            g_react =  self.factor_react*self.kinetics(c_ret*p_over_c_tot)
        g_ret = g[:, self.num_r_perm:, :]
        if not kinetics_as_source:
            self.g_react_source = g_react
            g_ret[...] -= g_react
        else:
            g_ret[...] -= self.g_react_source
        return g, self._jac

    def construct_g_c_p(self, c_p=None, T=None, c_old=None, compute_jac=False, kinetics_as_source = False):
        """
        Construct the residual vector g and the Jacobian matrix for the system.

        Parameters:
        - c_p (numpy.ndarray): Current concentration field.
        - c_old (numpy.ndarray): Previous concentration field.

        Returns:
        - g (numpy.ndarray): Residual vector.
        - Jac (scipy.sparse.csc_matrix): Jacobian matrix.
        """
        if (c_p is None):
            c_p = self.c_p
        if (T is None):
            T = self.T
        c = c_p[...,:-1]
        p = c_p[...,-1]
        shape_c_p = c_p.shape
        g = np.empty(shape_c_p)
        c_sum = np.sum(c, axis=-1)
        y = c/c_sum[...,np.newaxis]
        
        c_ret = c[:, self.num_r_perm:, :]
        c_tot_ret = np.sum(c_ret, axis=-1, keepdims=True)
        #_, c_tot_ret =  self.split_perm_and_ret(c_tot)
        _, p_ret =  self.split_perm_and_ret(p)
        p_over_c_tot = p_ret[..., np.newaxis]/c_tot_ret

        g_conv, jac_conv = self.construct_g_conv(c, compute_jac=compute_jac)
        g_diff, jac_diff = self.construct_g_diff(c, compute_jac=compute_jac)
        if compute_jac:
            jac_cc = jac_conv + jac_diff
            if (c_old is not None):
                jac_cc += (1.0/self.dt)*self.jac_c_accum
            if not kinetics_as_source:
                g_react, jac_react = self.numjac(lambda c: self.factor_react*self.kinetics(c*p_over_c_tot), c_ret)
                shape_c_ret = c_ret.shape
                offset = (0, self.num_r_perm, 0)
                jac_react = update_csc_array_indices(jac_react, shape_c_ret, c.shape, offset=offset)
                jac_cc -= jac_react
            c_tot, dc_tot_dp_mat = self.numjac_p(lambda p: self.correlation.molar_density(y, T, p), p)
            y_mat = construct_coefficient_matrix(y, shape=(c.shape, p.shape+(1,)))
            jac_darcy = self.construct_jac_darcy()
            jac_pp = self.sum_c @ jac_darcy
            jac_pp += (self.sum_c @ jac_cc @ y_mat) @ dc_tot_dp_mat
            jac_cp = jac_darcy
            jac_pc = self.sum_c @ jac_cc
            jac_pc = jac_pc - (jac_pc @ y_mat) @ self.sum_c
            jac_pp += self.penalty_p * dc_tot_dp_mat
            jac_pc -= self.penalty_p * self.sum_c
            shape_c = c.shape
            shape_p = p.shape + (1,)
            offset = (0,)*(c.ndim-1) + (shape_c[-1],)
            jac_cc = update_csc_array_indices(jac_cc, shape_c, shape_c_p)
            jac_pp = update_csc_array_indices(jac_pp, shape_p, shape_c_p, offset=offset)
            jac_cp = update_csc_array_indices(jac_cp, (shape_c, shape_p), shape_c_p, offset=(None, offset))
            jac_pc = update_csc_array_indices(jac_pc, (shape_p, shape_c), shape_c_p, offset=(offset, None))
            self._jac = jac_cc + jac_pp + jac_cp + jac_pc
        else:
            c_tot =self.correlation.molar_density(y, T, p)
            if not kinetics_as_source:
                g_react =  self.factor_react*self.kinetics(c_ret*p_over_c_tot)
        
        g_c = self.g_c_in.reshape(c.shape) + g_conv + g_diff
        if (c_old is not None):
            g_c += (self.jac_c_accum @ ((c-c_old).reshape((-1, 1))/self.dt)).reshape(c.shape)        
        g_ret = g_c[:, self.num_r_perm:, :]
        if not kinetics_as_source:
            self.g_react_source = g_react
            g_ret[...] -= g_react
        else:
            g_ret[...] -= self.g_react_source
        g_p = np.sum(g_c, axis=-1) + self.penalty_p * (c_tot - c_sum)
        g[...,:-1] = g_c
        g[...,-1] = g_p
        return g, self._jac

    def construct_g_T_conv(self, T, compute_jac = False):
        """Assemble convective energy residual (includes -T∇·u term).

        Args:
            T (ndarray): Temperature field (z,r).
            compute_jac (bool): If True, compute Jacobian sparsity pattern.

        Returns:
            tuple: (g, jac) residual and Jacobian (or None if compute_jac=False)
        """
        bc_ret_dirichlet = {'a': 0, 'b': 1, 'd': self.T_ret_in}
        bc_neumann_hom = {'a': 1, 'b': 0, 'd': 0}
        if (self.is_counter_current):
            bc_ret_ax = (bc_neumann_hom, bc_ret_dirichlet)
            is_inflow = self.u_ret_ax[0,:] > 0 
            if (np.any(is_inflow)):
                b_out = (is_inflow*1.0).reshape((1,-1))
                a_out = (1.0-b_out)
                d_out = b_out*self.T_ret_in
                bc_ret_ax = ({'a':a_out, 'b':b_out, 'd':d_out}, bc_ret_dirichlet)
        else:
            bc_ret_ax = (bc_ret_dirichlet, bc_neumann_hom)
            is_inflow = self.u_ret_ax[-1,:] < 0
            if (np.any(is_inflow)):
                b_out = (is_inflow*1.0).reshape((1,-1))
                a_out = (1.0-b_out)
                d_out = b_out*self.T_ret_in
                bc_ret_ax = (bc_ret_dirichlet, {'a':a_out, 'b':b_out, 'd':d_out})
        
        bc_perm_dirichlet = {'a': 0, 'b': 1, 'd': self.T_perm_in}
        
        g = np.empty(T.shape)
        g_vect = g.ravel()
        
        T_perm = T[:, 0:self.num_r_perm]
        self.T_perm_ax,_ = interp_cntr_to_stagg_tvd(T_perm, self.z_f, self.z_c, bc = (bc_perm_dirichlet, bc_neumann_hom), v = self.u_perm_ax, tvd_limiter = upwind, axis=0)
        flux_perm_ax = self.u_perm_ax * self.T_perm_ax
        g_vect[:] = self.div_p_perm_ax @ flux_perm_ax.ravel()
        self.T_perm_rad,_ = interp_cntr_to_stagg_tvd(T_perm, self.r_f_perm, self.r_c_perm, bc = (bc_neumann_hom, bc_neumann_hom), v = self.u_perm_rad, tvd_limiter = upwind, axis=1)
        flux_perm_rad = self.u_perm_rad * self.T_perm_rad
        g_vect[:] += self.div_p_perm_rad @ flux_perm_rad.ravel()
        
        T_ret = T[:, self.num_r_perm:]
        self.T_ret_ax,_ = interp_cntr_to_stagg_tvd(T_ret, self.z_f, self.z_c, bc = bc_ret_ax, v = self.u_ret_ax, tvd_limiter = upwind, axis=0)
        flux_ret_ax = self.u_ret_ax * self.T_ret_ax
        g_vect[:] += self.div_p_ret_ax @ flux_ret_ax.ravel()
        self.T_ret_rad,_ = interp_cntr_to_stagg_tvd(T_ret, self.r_f_ret, self.r_c_ret, bc = (bc_neumann_hom, bc_neumann_hom), v = self.u_ret_rad, tvd_limiter = upwind, axis=1)
        flux_ret_rad = self.u_ret_rad * self.T_ret_rad
        g_vect[:] += self.div_p_ret_rad @ flux_ret_rad.ravel()      
        
        g_vect[:] -= (T*self.div_u).ravel()
                
        if compute_jac:
            conv_matrix_perm_ax, _ = construct_convflux_upwind(T_perm.shape, self.z_f, self.z_c, bc = (bc_perm_dirichlet, bc_neumann_hom), v = self.u_perm_ax, axis=0)
            jac_perm = self.div_p_perm_ax @ conv_matrix_perm_ax
            conv_matrix_perm_rad, _ = construct_convflux_upwind(T_perm.shape, self.r_f_perm, self.r_c_perm, bc = (bc_neumann_hom, bc_neumann_hom), v = self.u_perm_rad, axis=1)
            jac_perm += self.div_p_perm_rad @ conv_matrix_perm_rad
            conv_matrix_ret_ax, _ = construct_convflux_upwind(T_ret.shape, self.z_f, self.z_c, bc = bc_ret_ax, v = self.u_ret_ax, axis=0)
            jac_ret = self.div_p_ret_ax @ conv_matrix_ret_ax
            conv_matrix_ret_rad, _ = construct_convflux_upwind(T_ret.shape, self.r_f_ret, self.r_c_ret, bc = (bc_neumann_hom, bc_neumann_hom), v= self.u_ret_rad, axis=1)
            jac_ret += self.div_p_ret_rad @ conv_matrix_ret_rad
            jac_perm = update_csc_array_indices(jac_perm, (None, T_perm.shape), (None, T.shape))
            jac_ret = update_csc_array_indices(jac_ret, (None, T_ret.shape), (None, T.shape), offset=(None, (0,self.num_r_perm,0)))
            jac = jac_perm + jac_ret - construct_coefficient_matrix(self.div_u)
            return g, jac
        else:
            return g, None
        
    def construct_g_T_cond(self, T):
        """Assemble conductive + membrane interfacial heat transfer residual.

        Returns:
            tuple: (g_cond, jac_cond, cp_inv_matrix)
        """
        g = np.empty(T.shape)
        g_vect = g.reshape((-1,1))
        
        y = self.c/np.sum(self.c, axis=-1, keepdims=True)  # Mole fractions
        lmbda = self.correlation.thermal_conductivity(y, T)
        cp = self.correlation.specific_heat(self.c, T)

        lmbda_perm = lmbda[:, 0:self.num_r_perm]
        lmbda_perm_ax = interp_cntr_to_stagg(lmbda_perm, self.z_f, self.z_c, axis=0)
        lmbda_perm_ax_mat = construct_coefficient_matrix(lmbda_perm_ax)
        jac_cond = self.div_p_perm_ax @ (-lmbda_perm_ax_mat @ self.grad_T_perm_ax)
        g_cond_bc = self.div_p_perm_ax @ (-lmbda_perm_ax_mat @ self.grad_bc_T_perm_ax)
        
        lmbda_perm_rad = interp_cntr_to_stagg(lmbda_perm, self.r_f_perm, self.r_c_perm, axis=1)
        lmbda_perm_rad_mat = construct_coefficient_matrix(lmbda_perm_rad)
        jac_cond += self.div_p_perm_rad @ (-lmbda_perm_rad_mat @ self.grad_T_perm_rad)
        
        lmbda_ret = lmbda[:, self.num_r_perm:]
        lmbda_ret_ax = interp_cntr_to_stagg(lmbda_ret, self.z_f, self.z_c, axis=0)
        lmbda_ret_ax_mat = construct_coefficient_matrix(lmbda_ret_ax)
        jac_cond += self.div_p_ret_ax @ (-lmbda_ret_ax_mat @ self.grad_T_ret_ax)
        g_cond_bc += self.div_p_ret_ax @ (-lmbda_ret_ax_mat @ self.grad_bc_T_ret_ax)
        
        lmbda_ret_rad = interp_cntr_to_stagg(lmbda_ret, self.r_f_ret, self.r_c_ret, axis=1)
        lmbda_ret_rad_mat = construct_coefficient_matrix(lmbda_ret_rad)
        jac_cond += self.div_p_ret_rad @ (-lmbda_ret_rad_mat @ self.grad_T_ret_rad)

        #heat transfer through membrane
        
        bc_neumann_hom = {'a': 1, 'b': 0, 'd': 0}
        T_perm = self.T[:, 0:self.num_r_perm]
        T_ret  = self.T[:, self.num_r_perm:]
        _, _ ,T_perm_i, _ = compute_boundary_values(T_perm, self.r_f_perm, self.r_c_perm, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
        T_ret_i,_,_,_ = compute_boundary_values(T_ret, self.r_f_ret, self.r_c_ret, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
        y_perm = y[:, 0:self.num_r_perm,:]
        y_ret  = y[:, self.num_r_perm:,:]
        _, _ ,y_perm_i, _ = compute_boundary_values(y_perm, self.r_f_perm, self.r_c_perm, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
        y_ret_i,_,_,_ = compute_boundary_values(y_ret, self.r_f_ret, self.r_c_ret, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
        c_perm = self.c[:, 0:self.num_r_perm,:]
        c_ret  = self.c[:, self.num_r_perm:,:]
        _, _ ,c_perm_i, _ = compute_boundary_values(c_perm, self.r_f_perm, self.r_c_perm, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
        c_ret_i,_,_,_ = compute_boundary_values(c_ret, self.r_f_ret, self.r_c_ret, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
        _, _ ,u_perm_ax_i, _ = compute_boundary_values(self.u_perm_ax, self.r_f_perm, self.r_c_perm, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
        u_perm_i = interp_stagg_to_cntr(u_perm_ax_i, self.z_f, self.z_c, axis=0)
        u_ret_ax_i,_,_,_ = compute_boundary_values(self.u_ret_ax, self.r_f_ret, self.r_c_ret, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
        u_ret_i = interp_stagg_to_cntr(u_ret_ax_i, self.z_f, self.z_c, axis=0)


        visc_ret = self.correlation.viscosity(y_ret_i, T_ret_i)
        rho_ret = self.correlation.molecular_weight(c_ret_i)
        cp_ret = self.correlation.specific_heat(c_ret_i, T_ret_i)
        Re_ret = rho_ret*self.dp*np.abs(u_ret_i)/visc_ret
        Pr_ret = visc_ret * cp_ret / lmbda_ret_rad[:,[0]]
        Nu_ret = self.Nu_ret(Re_ret, Pr_ret)
        h_ret = Nu_ret*lmbda_ret_rad[:,[0]]/self.dp

        d_tube = 1.0*self.r_f_perm[-1]
        visc_perm = self.correlation.viscosity(y_perm_i, T_perm_i)
        rho_perm = self.correlation.molecular_weight(c_perm_i)
        cp_perm = self.correlation.specific_heat(c_perm_i, T_perm_i)
        Re_perm = rho_perm*d_tube*np.abs(u_perm_i)/visc_perm
        Pr_perm = visc_perm * cp_perm / lmbda_perm_rad[:,[-1]]
        Nu_perm = self.Nu_perm(Re_perm, Pr_perm)
        h_perm = Nu_perm*lmbda_perm_rad[:,[-1]]/d_tube

        if self.nu==1:
            resist_mem = (self.r_f_perm[-1]*np.log(self.r_f_ret[0]/self.r_f_ret[-1])) / self.lambda_mem
            factor_geom = (self.r_f_ret[0]/self.r_f_perm[-1])
            U = 1.0/(1.0/h_ret + resist_mem + 1.0/(factor_geom*h_perm))
        else:
            resist_mem = (self.r_f_perm[-1]-self.r_f_ret[0]) /  self.lambda_mem
            U = 1.0/(1.0/h_ret + resist_mem + 1.0/h_perm)
            factor_geom = 1.0

        ic_1 = {'a':(lmbda_perm_rad[:,[-1]],0), 'b':(U,U)}
        ic_2 = {'a':(0,factor_geom*lmbda_ret_rad[:,[0]]), 'b':(-U,U)}
        interf_mat_perm, _, interf_mat_ret, _ = construct_interface_matrices((T_perm.shape, T_ret.shape), (self.r_f_perm, self.r_f_ret), ic=(ic_1, ic_2), axis=1)
        jac_cond_ic_perm = self.div_p_perm_rad @ (-lmbda_perm_rad_mat @ self.grad_bc_T_perm_rad)
        jac_cond_ic_ret = self.div_p_ret_rad @ (-lmbda_ret_rad_mat @ self.grad_bc_T_ret_rad)
                
        jac_cond += jac_cond_ic_perm @ interf_mat_perm + jac_cond_ic_ret @ interf_mat_ret
    
        cp_inv_mat = construct_coefficient_matrix(1.0/cp)
        g_vect[:] = cp_inv_mat @ (jac_cond @ T.reshape((-1,1)) + g_cond_bc)
        jac_cond = cp_inv_mat @ jac_cond

        return g, jac_cond, cp_inv_mat

    def construct_g_T(self, T=None, T_old=None, compute_jac=False):
        """Combine accumulation, convection, conduction for temperature residual."""
        if (T is None):
            T = self.T   
        g_conv, jac_conv = self.construct_g_T_conv(T, compute_jac=compute_jac)
        g_cond, jac_cond, cp_inv_mat = self.construct_g_T_cond(T)
        g =  g_conv + g_cond
        if T_old is not None:
            g += (self.jac_T_accum @ ((T-T_old).reshape((-1, 1))/self.dt)).reshape(T.shape)

        if (compute_jac):
            self._jac_T = (1.0/self.dt)*self.jac_T_accum + jac_conv + jac_cond
        return g, self._jac_T, cp_inv_mat
    
    def solve_temperature(self, T_old = None):
        """Solve energy equation (if non-isothermal) including reaction heat.

        Returns:
            tuple: (updated_c_tot, success_flag)
        """
        c_p = self.c_p
        T = self.T
        c = c_p[...,:-1]
        p = c_p[...,-1]
        T_vec = T.ravel()
        c_ret = c[:, self.num_r_perm:,:]
        p_ret = p[:, self.num_r_perm:]
        T_ret = T[:,self.num_r_perm:]        
        g_T, jac_T, cp_inv_mat = self.construct_g_T(T, T_old=T_old, compute_jac=True)
        g_T_ret = g_T[:,self.num_r_perm:]
        p_partial = p[:,self.num_r_perm:,np.newaxis]*c_ret/np.sum(c_ret, axis=-1, keepdims=True)
        rates =  self.factor_react*self.kinetics(p_partial)
        enthalpies= self.correlation.species_enthalpies(T_ret)
        dH_react = np.sum(rates * enthalpies, axis=-1)
        g_T_ret[...] += dH_react*cp_inv_mat.data.reshape(T.shape)[:,self.num_r_perm:]
        dT = -sla.spsolve(jac_T, g_T.reshape((-1,1)))
        self.cnt_T += 1
        
        success = (np.linalg.norm(dT, ord=np.inf) < 500) and (np.all(T_vec+dT)>0.0)
        if success:
            T_vec[...] += dT
        self.kinetics.set_T_and_p(T=T_ret)
        return success

    def solve_c_p(self, c_old = None, verbose = 0):
        """Newton/line-search solve for species concentrations.

        Returns:
            tuple: (y, c_tot, g_norm, success_flag)
        """
        success = True
        c_p = self.c_p
        c = c_p[...,:-1]
        p = c_p[...,-1]
        T = self.T
        ord = self.ord_norm
        factor_norm_c = self.factor_norm_c
        factor_norm_p = self.factor_norm_p
        c_p_vec = c_p.ravel()
        for k in range(self.num_concentration_iterations):
            g, jac= self.construct_g_c_p(c_p, c_old = c_old, compute_jac=True)
            g_norm = np.linalg.norm(g[...,:-1].ravel(), ord=ord) * factor_norm_c
            g_p_norm = np.linalg.norm(g[...,-1].ravel(), ord=ord) * factor_norm_p
            if k==0:
                g_norm_init = g_norm
                g_p_norm_init = g_p_norm
            dc_p = -sla.spsolve(jac, g.reshape((-1,1)))
            self.cnt_c_p += 1
            c_p_prev = c_p_vec.copy()
            g_norm_prev = g_norm
            g_p_norm_prev = g_p_norm
            alpha = 1.0
            while True:
                c_p_vec[...] = c_p_prev + alpha*dc_p
                self.construct_darcy_matrices(c=c_p[...,:-1], p=c_p[...,-1])          
                self.update_velocity_fields(p=c_p[...,-1])
                compute_jac = (k < self.num_concentration_iterations - 1)
                g, _ = self.construct_g_c_p(c_p, c_old=c_old, compute_jac=False)
                g_norm = np.linalg.norm(g[...,:-1].ravel(), ord=ord) * factor_norm_c
                g_p_norm = np.linalg.norm(g[...,-1].ravel(), ord=ord)* factor_norm_p
                if (g_norm < (1.0-1e-4*alpha)*g_norm_prev and g_p_norm < (1.0-1e-4*alpha)*g_p_norm_prev) or (not success):
                    break
                alpha *= 0.5
                if (alpha < 1e-3):
                    alpha = 0.0
                    success = False
                    if verbose > 0:
                        warnings.warn(f"Line search failed to improve residual for concentration solver: {g_norm} > {g_norm_prev}", RuntimeWarning)
                    #raise RuntimeWarning(f"Line search failed to improve residual: {g_norm} > {g_norm_prev}")
            if (g_norm < np.maximum(self.rtol_c * g_norm_init, self.atol_c) and g_p_norm < np.maximum(self.rtol_p * g_p_norm_init, self.atol_p)):
                break
        return g_norm, g_norm_init, g_p_norm, g_p_norm_init,success 

    def compute_dt_chem_min(self, rates, c_ret):
        """Return minimum explicit chemical time step avoiding negative c."""
        eps = 1e-30
        dt_chem_local = np.where(rates < 0,
                                    np.maximum(c_ret, eps) / (-rates + eps),
                                    np.inf)
        dt_chem_min = np.min(dt_chem_local)
        return dt_chem_min
        

    def compute_g_norm(self, c_old=None):
        """Compute norms of species residual and its sum (pressure consistency)."""
        g, _ = self.construct_g_c(self.c, c_old=c_old, compute_jac=False)
        g_norm = np.linalg.norm(g.ravel(), ord=self.ord_norm) * self.factor_norm_c
        g_p_norm = np.linalg.norm(np.sum(g, axis=-1).ravel(), ord=self.ord_norm) * self.factor_norm_p
        return g_norm, g_p_norm

    def _solve_step(self, c_old = None, T_old = None):
        """
        Helper method: Performs segregated Newton iterations for a fixed `self.factor_react`.
        This is the "corrector" part of the continuation scheme.
        
        Returns:
            num_iterations (int): The number of outer Newton iterations taken.
            success (bool): True if the solution converged within tolerances, False otherwise.
        """

        for j in range(self.num_newton_iterations):
            # --- Segregated Solves ---
            g_norm, g_norm_start, g_p_norm, g_p_norm_start, success_c_p = self.solve_c_p(c_old=c_old)
            print(f"j: {j}, g_norm: {g_norm}")
            if j == 0:
                g_norm_init = g_norm_start
                g_p_norm_init = g_p_norm_start

            success_T = True
            if not self.is_isothermal:
                success_T = self.solve_temperature(T_old=T_old)
                
            if (g_norm > 10*g_norm_init or g_p_norm > 10*g_p_norm_init):
                return j+1, g_norm, g_p_norm, False, g_norm/g_norm_init, g_p_norm/g_p_norm_init

            success = success_c_p and success_T
            if (g_norm < np.maximum(self.rtol * g_norm_init, self.atol)):
                return j+1, g_norm, g_p_norm, success, g_norm/g_norm_init, g_p_norm/g_p_norm_init # Converged!

        return self.num_newton_iterations-1, g_norm, g_p_norm, success, g_norm/g_norm_init, g_p_norm/g_p_norm_init # Failed to converge within max iterations

    def solve(self, num_timesteps=None, dt=None):
        """
        Solves the steady-state problem using an adaptive predictor-corrector
        continuation method on the reaction rate scaling factor. This is the
        recommended robust solver for difficult non-linear problems.
        """
        # --- Initialization ---
        
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        if dt is not None:
            self.dt = dt
        self.cnt_c_p, self.cnt_T = 0, 0
        for i in range(num_timesteps):
            T_old = self.T.copy()
            c_old = self.c_p[...,:-1].copy()
            num_iters, g_norm, g_p_norm, is_converged, conv_factor, conv_factor_p = self._solve_step(c_old=c_old, T_old=T_old)
        return is_converged

    def solve_adaptive(self, c_old=None, T_old=None, verbose = 0):
        """
        Solves the steady-state problem using an adaptive predictor-corrector
        continuation method on the reaction rate scaling factor. This is the
        recommended robust solver for difficult non-linear problems.
        """
        # --- Initialization ---
        
        conv_factor_min = math.exp(-self.newton_conv_rate_min)
        self.cnt_c_p, self.cnt_T = 0, 0, 0
        g_norm, g_p_norm = None, None

        # History for predictor step (current, previous)
        c_p_prev, T_prev = self.c_p.copy(), self.T.copy()
        c_p_prev_prev, T_prev_prev = None, None, None
        factor_react_prev, factor_react_prev_prev = 0.0, 0.0

        # --- Step 2: Main Continuation Loop ---
        is_first_step = True
        is_converged = False
        if (self.factor_react is None):
            self.factor_react = 1.0
            factor_react_max = 1.0
        else:
            factor_react_max = self.factor_react
        dfactor_react = min(self.dfactor_react_init, factor_react_max) 
        factor_react_prev = self.factor_react        
        
        while not is_converged:
            g_norm, g_p_norm = None, None
            
            # --- Predictor Step ---
            if c_p_prev_prev is not None:
                # Use a first-order (secant) predictor for a better initial guess
                d_factor_hist = factor_react_prev - factor_react_prev_prev
                if d_factor_hist > 1e-9: # Avoid division by zero on retry
                    step_ratio = (self.factor_react - factor_react_prev) / d_factor_hist
                    self.c_p = c_p_prev + (c_p_prev - c_p_prev_prev) * step_ratio
                    self.T = T_prev + (T_prev - T_prev_prev) * step_ratio
                    self.c_p = np.maximum(self.c, 0) # Ensure concentrations are non-negative
                    self.construct_darcy_matrices(compute_jac=False)
                    self.update_velocity_fields(p=self.c_p[...,-1])
            elif not is_first_step:
                # Use a zero-order predictor (the last solution) for the first step
                self.c_p, self.T = c_p_prev.copy(), T_prev.copy()
                self.construct_darcy_matrices(compute_jac=False)
                self.update_velocity_fields(p=self.c_p[...,-1])

            # --- Corrector Step ---
            if verbose > 1:
                print(f"Attempting factor_react = {self.factor_react:.4f} (step size = {dfactor_react:.4f})...")
            num_iters, g_norm, g_p_norm, is_converged, conv_factor, conv_factor_p = self._solve_step(c_old=c_old, T_old=T_old)
            is_converging = ((conv_factor < conv_factor_min) and (conv_factor_p < 1)) or is_converged
            if is_converged and self.factor_react == factor_react_max:
                break
            # --- Adapt Step Size ---
            if is_converging:
                if verbose > 1:
                    print(f"Converging in {num_iters} iterations with g_norm = {g_norm:.4e} and conv_factor = {conv_factor:.4e} and g_p_norm = {g_p_norm:.4e} and conv_factor_p = {conv_factor_p:.4e}. Increasing step size.")
                # Update history for the next predictor step
                c_p_prev_prev, p_prev_prev, T_prev_prev = c_p_prev, p_prev, T_prev
                factor_react_prev_prev = factor_react_prev
                c_p_prev, T_prev = self.c.copy(), self.T.copy()
                factor_react_prev = self.factor_react    
                # Increase step size
                if is_converged:
                    dfactor_react *= self.dfactor_react_increase_large
                self.factor_react = min(self.factor_react + dfactor_react, factor_react_max)
            else:
                if verbose > 1:
                    print(f"FAILED: Took {num_iters} iterations. Restoring state and reducing step size.")

                # Restore previous is_convergedful state
                self.c_p, self.T = c_p_prev, T_prev
                if is_first_step:
                    self.factor_react = 0.0    
                else:
                    self.factor_react = factor_react_prev
                    # Decrease step size and retry from the last good point
                    dfactor_react *= self.dfactor_react_decrease
                if dfactor_react < self.dfactor_react_min:
                    break
            is_first_step = False
        if is_converged:
            if verbose > 1:
                print("\nContinuation is_convergedfully completed. Final solution at factor_react = 1.0 reached.")
        else:
            if verbose > 0:
                warnings.warn(f"Continuation failed: final factor_react = {self.factor_react}", RuntimeWarning)
        return is_converged


        
    
    def compute_fluxes_diff(self, c = None, T = None, p = None):
        """Compute diffusive + permeation species fluxes (axis & radial)."""
        shape_c_ret = (self.num_z, self.num_r_ret, self.num_c)
        shape_c_perm = (self.num_z, self.num_r_perm, self.num_c)
        if c is None:
            c = self.c_p[...,:-1]
        if T is None:
            T = self.T
        if p is None:
            p = self.c_p[...,-1]
        c_perm, c_ret = self.split_perm_and_ret(c)
        T_perm, T_ret = self.split_perm_and_ret(T)
        p_perm, p_ret = self.split_perm_and_ret(p)
        c_vect = c.reshape((-1,1))
        
        y_ret = c_ret/np.sum(c_ret, axis=-1, keepdims=True)  # Mole fractions
        diff_field_ret = self.correlation.diffusion(y_ret, T_ret, p_ret)
        diff_field_ret_ax = interp_cntr_to_stagg(diff_field_ret, x_f=self.z_f, x_c=self.z_c, axis=0)
        diff_matrix_ret_ax = construct_coefficient_matrix(diff_field_ret_ax, shape_c_ret, axis=0)
        diff_field_ret_rad = interp_cntr_to_stagg(diff_field_ret, x_f=self.r_f_ret, x_c=self.r_c_ret, axis=1)
        diff_matrix_ret_rad = construct_coefficient_matrix(diff_field_ret_rad, shape_c_ret, axis=1)

        y_perm = c_perm/np.sum(c_perm, axis=-1, keepdims=True)  # Mole fractions
        diff_field_perm = self.correlation.diffusion(y_perm, T_perm, p_perm)
        diff_field_perm_ax = interp_cntr_to_stagg(diff_field_perm, x_f=self.z_f, x_c=self.z_c, axis=0)
        diff_matrix_perm_ax = construct_coefficient_matrix(diff_field_perm_ax, shape_c_perm, axis=0)
        diff_field_perm_rad = interp_cntr_to_stagg(diff_field_perm, x_f=self.r_f_perm, x_c=self.r_c_perm, axis=1)
        diff_matrix_perm_rad = construct_coefficient_matrix(diff_field_perm_rad, shape_c_perm, axis=1)
        
        fluxes_ret_ax  = (-diff_matrix_ret_ax @ (self.grad_c_ret_ax @ c_vect + self.grad_bc_c_ret_ax)).reshape((self.num_z+1, self.num_r_ret, self.num_c))
        fluxes_ret_rad = (-diff_matrix_ret_rad @ (self.grad_c_ret_rad @ c_vect)).reshape((self.num_z, self.num_r_ret+1, self.num_c))
        fluxes_perm_ax  = (-diff_matrix_perm_ax @ (self.grad_c_perm_ax @ c_vect + self.grad_bc_c_perm_ax)).reshape((self.num_z+1, self.num_r_perm, self.num_c))
        fluxes_perm_rad = (-diff_matrix_perm_rad @ (self.grad_c_perm_rad @ c_vect)).reshape((self.num_z, self.num_r_perm+1, self.num_c))
        
        if (T_perm.ndim > 1):
            T_perm_mem,_ = compute_boundary_values(T_perm, self.r_f_perm, self.r_c_perm, bc=None, axis=1, bound_id = 1)
            T_ret_mem,_ = compute_boundary_values(T_ret, self.r_f_ret, self.r_c_ret, bc=None, axis=1, bound_id = 0)
            P_perm_mem = self.Rg*T_perm_mem[:, 0, np.newaxis]*self.perm
            P_ret_mem = self.Rg*T_ret_mem[:, 0, np.newaxis]*self.perm
        else:
            P_perm_mem = np.broadcast_to((self.Rg*T_perm*self.perm).reshape((1,1,-1)), (self.num_z, self.num_c))
            P_ret_mem = np.broadcast_to((self.Rg*T_ret*self.perm).reshape((1,1,-1)), (self.num_z, self.num_c))

        c_perm_mem,_ = compute_boundary_values(c_perm, self.r_f_perm, self.r_c_perm, bc=None, axis=1, bound_id = 1)
        c_ret_mem,_ = compute_boundary_values(c_ret, self.r_f_ret, self.r_c_ret, bc=None, axis=1, bound_id = 0)

        fluxes_perm_rad[:,-1,:] = P_perm_mem * c_perm_mem[:,0,:] - P_ret_mem * c_ret_mem[:,0,:]
        factor_geom = (self.r_f_perm[-1]/self.r_f_ret[0])**self.nu
        fluxes_ret_rad[:,0,:] = factor_geom * fluxes_perm_rad[:,-1,:]

        return fluxes_ret_ax, fluxes_ret_rad, fluxes_perm_ax, fluxes_perm_rad
    
    def compute_fluxes_conv(self, c=None, compute_jac = False):
        """Compute convective species fluxes using upwind TVD reconstructions."""
        if c is None:
            c = self.c_p[...,:-1]
        bc_neumann_hom = {'a': 1, 'b': 0, 'd': 0}
        bc_none = {'a': 0, 'b': 0, 'd': 0}
        if (self.is_counter_current):
            bc_ret_ax = (bc_neumann_hom, bc_none)
        else:
            bc_ret_ax = (bc_none, bc_neumann_hom)
        
        c_perm = c[:, 0:self.num_r_perm, :]
        u_perm_ax = self.u_perm_ax[...,np.newaxis]
        u_perm_rad = self.u_perm_rad[...,np.newaxis]
        c_perm_ax,_ = interp_cntr_to_stagg_tvd(c_perm, self.z_f, self.z_c, bc = (bc_none, bc_neumann_hom), v = u_perm_ax, tvd_limiter = upwind, axis=0)
        fluxes_perm_ax = u_perm_ax * c_perm_ax
        c_perm_rad,_ = interp_cntr_to_stagg_tvd(c_perm, self.r_f_perm, self.r_c_perm, bc = (bc_neumann_hom, bc_neumann_hom), v = u_perm_rad, tvd_limiter = upwind, axis=1)
        fluxes_perm_rad = u_perm_rad * c_perm_rad
        
        c_ret = c[:, self.num_r_perm:, :]
        u_ret_ax = self.u_ret_ax[...,np.newaxis]
        u_ret_rad = self.u_ret_rad[...,np.newaxis]
        c_ret_ax,_ = interp_cntr_to_stagg_tvd(c_ret, self.z_f, self.z_c, bc = bc_ret_ax, v = u_ret_ax, tvd_limiter = upwind, axis=0)
        fluxes_ret_ax = u_ret_ax * c_ret_ax
        c_ret_rad,_ = interp_cntr_to_stagg_tvd(c_ret, self.r_f_ret, self.r_c_ret, bc = (bc_neumann_hom, bc_neumann_hom), v = u_ret_rad, tvd_limiter = upwind, axis=1)
        fluxes_ret_rad = u_ret_rad * c_ret_rad
        
        fluxes_perm_ax[0,:,:] = self.flux_perm_in[0,:,:]
        if (self.is_counter_current):
            fluxes_ret_ax[-1,:,:] = -self.flux_ret_in[0,:,:]
        else:
            fluxes_ret_ax[0,:,:] = self.flux_ret_in[0,:,:]       
        
        return fluxes_ret_ax, fluxes_ret_rad, fluxes_perm_ax, fluxes_perm_rad
 
    def compute_flows(self):
        """Integrate fluxes to obtain net axial inlet/outlet & membrane transfer."""
        fluxes_ret_ax, fluxes_ret_rad, fluxes_perm_ax, fluxes_perm_rad = self.compute_fluxes_diff()
        fluxes_conv_ret_ax,  fluxes_conv_ret_rad, fluxes_conv_perm_ax, fluxes_conv_perm_rad  = self.compute_fluxes_conv()
        fluxes_ret_ax += fluxes_conv_ret_ax
        fluxes_ret_rad += fluxes_conv_ret_rad
        fluxes_perm_ax += fluxes_conv_perm_ax
        fluxes_perm_rad += fluxes_conv_perm_rad
        
        # Compute cross-sectional areas for radial and axial directions
        dr_sq_ret = (self.r_f_ret[1:]**2 - self.r_f_ret[:-1]**2).reshape((-1,1))
        dr_sq_perm = (self.r_f_perm[1:]**2 - self.r_f_perm[:-1]**2).reshape((-1,1))
        dz = (self.z_f[1:] - self.z_f[:-1]).reshape((-1,1))

        flows_ret_ax = np.pi*np.sum(fluxes_ret_ax[[0,-1],:,:]*dr_sq_ret, axis=1)
        flows_ret_mem = 2.0*np.pi*self.r_f_ret[0]*np.sum(fluxes_ret_rad[:,0,:]*dz, axis=0)
        flows_perm_ax = np.pi*np.sum(fluxes_perm_ax[[0,-1],:,:]*dr_sq_perm, axis=1)
        flows_perm_mem = 2.0*np.pi*self.r_f_perm[-1]*np.sum(fluxes_perm_rad[:,-1,:]*dz, axis=0)

        return flows_ret_ax, flows_ret_mem, flows_perm_ax, flows_perm_mem
    
    def info(self):
        """
        Print information about the membrane reactor.
        """
        vol_ret = np.pi * (self.r_f_ret[-1]**2-self.r_f_ret[0]**2) * (self.z_f[-1] - self.z_f[0])
        vol_perm = np.pi * (self.r_f_perm[-1]**2-self.r_f_perm[0]**2) * (self.z_f[-1] - self.z_f[0])
        flow_vol_ret = self.F_ret_in* self.Rg * self.T_ret_in/self.p_ret_out
        flow_vol_perm = self.F_perm_in* self.Rg * self.T_perm_in/self.p_perm_out
        print(f"Residence time retentate side: {vol_ret/flow_vol_ret}")
        print(f"Residence time permeate side: {vol_perm/flow_vol_perm}")
