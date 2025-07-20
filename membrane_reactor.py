import math
import os
import importlib
import numpy as np
import json
import scipy as sp
import scipy.sparse.linalg as sla
from scipy.sparse import csc_array

from pymrm import non_uniform_grid, construct_coefficient_matrix, construct_grad, construct_div, construct_convflux_upwind, interp_cntr_to_stagg_tvd, upwind, update_csc_array_indices, interp_cntr_to_stagg, interp_stagg_to_cntr, compute_boundary_values, construct_interface_matrices, NumJac, clip_approach
from gas_mixture_correlations import GasMixtureCorrelations
from ammonia_synthesis_kinetics import AmmoniaSynthesisKinetics
import defaults  # Import the defaults module

class MembraneReactor:
    def __init__(self, config_file=None, c=None, p=None, T=None, **kwargs):
        """Initialize the reactor with defaults from defaults.py, optional JSON, and keyword arguments."""

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

        self.init_derived_parameters()
        self.init_fields(c, p, T)
        _, T_ret = self.split_perm_and_ret(self.T)
        _, p_ret = self.split_perm_and_ret(self.p)

        self.kinetics = AmmoniaSynthesisKinetics(self.species, T=T_ret, p=p_ret, rho_b=self.rho_b, rho_c=self.rho_c)
        self.init_jac()

    def init_derived_parameters(self):
        self.y_ret_in = np.asarray(self.y_ret_in, dtype=np.float64).reshape((1,1,-1))  # Convert to numpy array
        self.y_perm_in = np.asarray(self.y_perm_in, dtype=np.float64).reshape((1,1,-1))  # Convert to numpy array
        self.y_ret_init = np.asarray(self.y_ret_init, dtype=np.float64).reshape((1,1,-1))  # Convert to numpy array
        self.y_perm_init = np.asarray(self.y_perm_init, dtype=np.float64).reshape((1,1,-1))  # Convert to numpy array

        self.num_c = len(self.species)
        self.rho_b = self.rho_c*self.Dcat*(1-self.eps) #[Kgcat/m3R] bed density    
        
        self.correlation = GasMixtureCorrelations(self.species, self.database)    
        
        self.num_r_perm = np.round(self.r_max_perm / self.r_max *self.num_r).astype('int') + 3
        self.num_r_ret = self.num_r - self.num_r_perm
        self.create_spatial_discretization()

        # membrane permeabilities
        perm_dict= {'NH3': self.Perm_NH3, 'H2': self.Perm_NH3 / self.Sel_am_hy, 'N2': self.Perm_NH3 / self.Sel_am_ni}
        perm = np.zeros((1, 1, self.num_c))
        for i, species in enumerate(self.species):
            perm[0, 0, i] = perm_dict[species]
        perm = np.broadcast_to(perm, (self.num_z, 1, self.num_c))
        fltr = (self.z_c <= self.Lsealing) 
        if (any(fltr)):
            perm = perm.copy()
            perm[fltr,:,:] = 0
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
        
        jac_c_accum_perm = construct_coefficient_matrix(1.0 / self.dt, shape_c_perm)
        jac_c_accum_ret = construct_coefficient_matrix(self.eps / self.dt, shape_c_ret)
        self.div_c_perm_ax  = construct_div(shape_c_perm, self.z_f, nu=0, axis=0)
        self.div_c_perm_rad = construct_div(shape_c_perm, self.r_f_perm, nu=self.nu, axis=1)
        self.div_c_ret_ax  = construct_div(shape_c_ret, self.z_f, nu=0, axis=0)
        self.div_c_ret_rad = construct_div(shape_c_ret, self.r_f_ret, nu=self.nu, axis=1)
        self.grad_c_perm_ax, self.grad_bc_c_perm_ax   = construct_grad(shape_c_perm, self.z_f, self.z_c, bc=(bc_none, bc_neumann_hom), axis=0)
        self.grad_c_perm_rad, _, self.grad_bc_c_perm_rad = construct_grad(shape_c_perm, self.r_f_perm, self.r_c_perm, bc=(bc_neumann_hom, bc_dirichlet), axis=1, shapes_d = (None, (self.num_z, 1, self.num_c)))
        self.grad_c_ret_ax, self.grad_bc_c_ret_ax   = construct_grad(shape_c_ret, self.z_f, self.z_c, bc_ret_ax, axis=0)
        self.grad_c_ret_rad, self.grad_bc_c_ret_rad, _ = construct_grad(shape_c_ret, self.r_f_ret, self.r_c_ret, bc=(bc_dirichlet, bc_neumann_hom), axis=1, shapes_d = ((self.num_z, 1, self.num_c), None))
        
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
        
        self.jac_T_accum = construct_coefficient_matrix(1.0 / self.dt, shape_p)
        self.grad_T_perm_ax, self.grad_bc_T_perm_ax   = construct_grad(shape_p_perm, self.z_f, self.z_c, bc=(bc_dirichlet, bc_neumann_hom), axis=0)
        self.grad_bc_T_perm_ax *= self.T_perm_in
        self.grad_T_perm_rad, self.grad_bc_T_perm_rad = construct_grad(shape_p_perm, self.r_f_perm, self.r_c_perm, bc=(bc_neumann_hom, bc_dirichlet), axis=1)
        self.grad_T_ret_ax, self.grad_bc_T_ret_ax   = construct_grad(shape_p_ret, self.z_f, self.z_c, bc_T_ret_ax, axis=0)
        self.grad_bc_T_ret_ax *= self.T_ret_in
        self.grad_T_ret_rad, self.grad_bc_T_ret_rad = construct_grad(shape_p_ret, self.r_f_ret, self.r_c_ret, bc=(bc_dirichlet, bc_neumann_hom), axis=1)
        
        
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

        self.numjac = NumJac(shape_c_ret)
        self.numjac_p = NumJac(shape_p + (1,))
        values = np.ones(shape_c)
        num_rows = np.prod(shape_p)
        num_cols = np.prod(shape_c)
        row_indices = np.broadcast_to(np.arange(np.prod(shape_p),dtype=int).reshape(shape_p+(1,)), shape_c)
        col_ptrs = np.arange(np.prod(shape_c)+1,dtype=int)
        self.sum_c = csc_array((values.ravel(), row_indices.ravel(), col_ptrs.ravel()), shape=(num_rows, num_cols))
        self.g_c_in = self.div_c_perm_ax[:,0:self.flux_perm_in.size] @ self.flux_perm_in.ravel()
        if (self.is_counter_current):
            self.g_c_in -= self.div_c_ret_ax[:,-self.flux_ret_in.size:] @ self.flux_ret_in.ravel()
        else:
            self.g_c_in += self.div_c_ret_ax[:,0:self.flux_ret_in.size] @ self.flux_ret_in.ravel()
                                

    def init_fields(self, c=None, p=None, T=None):
        """
        Initialize the concentration field.
        """
        shape_c = (self.num_z, self.num_r, self.num_c)
        shape_p = (self.num_z, self.num_r)
        shape_T = (self.num_z, self.num_r)
        
        if c is None:
            self.c = np.empty(shape_c)
            c_ret = self.c[:, self.num_r_perm:, :]
            c_ret_0 = self.correlation.molar_density(self.y_ret_init, self.T_ret_init, self.p_ret_out)*self.y_ret_init
            c_ret[...] = np.broadcast_to(c_ret_0, c_ret.shape)
            c_perm = self.c[:, :self.num_r_perm, :]
            c_perm_0 = self.correlation.molar_density(self.y_perm_init, self.T_perm_init, self.p_perm_out)*self.y_perm_init
            c_perm[...] = np.broadcast_to(c_perm_0.reshape((1,1,-1)), c_perm.shape)
        else:
            self.c = np.broadcast_to(np.array(c), shape_c).copy()
            c_ret = self.c[:, self.num_r_perm:, :]
            c_perm = self.c[:, :self.num_r_perm, :]
        
        self.c_tot_ret_ax = np.sum(interp_cntr_to_stagg(c_ret, x_f=self.z_f, x_c=self.z_c, axis=0), axis=-1)
        self.c_tot_ret_rad = np.sum(interp_cntr_to_stagg(c_ret, x_f=self.r_f_ret, x_c=self.r_c_ret, axis=1), axis=-1)
        self.c_tot_perm_ax = np.sum(interp_cntr_to_stagg(c_perm, x_f=self.z_f, x_c=self.z_c, axis=0), axis=-1)
        self.c_tot_perm_rad = np.sum(interp_cntr_to_stagg(c_perm, x_f=self.r_f_perm, x_c=self.r_c_perm, axis=1), axis=-1)
        
        if p is None:
            self.p = np.empty(shape_p)
            p_ret = self.p[:, self.num_r_perm:]
            p_ret[:,:] = np.broadcast_to(np.array(self.p_ret_out).reshape((1,1)), p_ret.shape)
            p_perm = self.p[:, :self.num_r_perm]
            p_perm[:,:] = np.broadcast_to(np.array(self.p_perm_out).reshape((1,1)), p_perm.shape)            
        else:
            self.p = np.broadcast_to(np.array(p), shape_p).copy()
            
        if T is None:
            self.T = np.empty(shape_T)
            T_ret = self.T[:, self.num_r_perm:]
            T_ret[:,:] = np.broadcast_to(np.array(self.T_ret_init).reshape((1,1)), T_ret.shape)         
            T_perm = self.T[:, :self.num_r_perm]
            T_perm[:,:] = np.broadcast_to(np.array(self.T_perm_init).reshape((1,1)), T_perm.shape)
        else:
            self.T = np.broadcast_to(np.array(T), shape_T).copy()
            
        self.u_perm_ax = np.zeros((self.num_z+1, self.num_r_perm))
        self.u_perm_rad = np.zeros((self.num_z, self.num_r_perm+1))
        self.u_ret_ax = np.zeros((self.num_z+1, self.num_r_ret))
        self.u_ret_rad = np.zeros((self.num_z, self.num_r_ret+1))
            
        return self.c, self.p, self.T
            
    def split_perm_and_ret(self, c):
        c = np.asarray(c)
        if c.ndim > 1 and c.shape[1] == self.num_r:
            c_perm = c[:, 0:self.num_r_perm, ...]
            c_ret = c[:, self.num_r_perm:self.num_r, ...]
        else:
            c_perm = c
            c_ret = c
        return c_perm, c_ret

    def construct_darcy_jacobian(self, c=None, T=None, p=None):
        if c is None:
            c = self.c
        if T is None:
            T = self.T
        if p is None:
            p = self.p
        c_perm, c_ret = self.split_perm_and_ret(c)
        T_perm, T_ret = self.split_perm_and_ret(T)
        p_perm, p_ret = self.split_perm_and_ret(p)

        viscosity = self.correlation.viscosity(c_perm, T_perm)
        k_field =  0.25*(self.r_max_perm**2-self.r_c_perm**2).reshape((1, -1))/viscosity
        k_field_ax = interp_cntr_to_stagg(k_field, x_f=self.z_f, x_c=self.z_c, axis=0)
        k_matrix = construct_coefficient_matrix(k_field_ax, (self.num_z, self.num_r_perm), axis=0)
        ck_matrix = construct_coefficient_matrix(self.c_tot_perm_ax*k_field_ax, (self.num_z, self.num_r_perm), axis=0)
        self.vel_matrix_perm_ax = (-k_matrix) @ self.grad_p_perm_ax
        jac_darcy = self.div_p_perm_ax @ ((-ck_matrix) @ self.grad_p_perm_ax)
        #self.vel_matrix_perm_in = (-k_matrix) @ self.grad_bc_p_perm_ax_in
        #jac_darcy_bc_perm_in = self.div_p_perm_ax @ self.vel_matrix_perm_in
        self.vel_bc_perm_ax = (-k_matrix) @ self.grad_bc_p_perm_ax
        k_field   = 100*self.r_max_perm**2/viscosity
        k_field_rad = interp_cntr_to_stagg(k_field, x_f=self.r_f_perm, x_c=self.r_c_perm, axis=1)
        k_matrix = construct_coefficient_matrix(k_field_rad, (self.num_z, self.num_r_perm), axis=1) 
        ck_matrix = construct_coefficient_matrix(self.c_tot_perm_rad*k_field_rad, (self.num_z, self.num_r_perm), axis=1) 
        self.vel_matrix_perm_rad = (-k_matrix) @ self.grad_p_perm_rad
        jac_darcy += self.div_p_perm_rad @ ((-ck_matrix) @ self.grad_p_perm_rad)
        
        viscosity = self.correlation.viscosity(c_ret, T_ret)
        rho = self.correlation.density(c_ret, T_ret, p_ret)
        u_ax_abs = np.abs(interp_stagg_to_cntr(self.u_ret_ax, self.z_f, self.z_c, axis=0))
        beta_0 = 150.0 * (1-self.eps)**2*viscosity / (self.eps**3 * self.dp**2) 
        beta_1 = 1.75 * rho *(1-self.eps) * np.abs(u_ax_abs) / (self.eps**3 * self.dp)
        # test
        beta_1 *=0
        beta_inv = 1.0/(beta_0 + beta_1)
        shape_p_ret = (self.num_z, self.num_r_ret)
        k_field_ax = interp_cntr_to_stagg(beta_inv, x_f=self.z_f, x_c=self.z_c, axis=0)
        k_matrix = construct_coefficient_matrix(k_field_ax, shape_p_ret, axis=0)
        ck_matrix = construct_coefficient_matrix(self.c_tot_ret_ax*k_field_ax, shape_p_ret, axis=0)
        self.vel_matrix_ret_ax = (-k_matrix) @ self.grad_p_ret_ax
        jac_darcy += self.div_p_ret_ax @ ((-ck_matrix) @ self.grad_p_ret_ax)
        self.vel_bc_ret_out = -(k_matrix @ self.grad_bc_p_ret_ax)
        k_field_rad = interp_cntr_to_stagg(beta_inv, x_f=self.r_f_ret, x_c=self.r_c_ret, axis=1)
        k_matrix = construct_coefficient_matrix(k_field_rad, shape_p_ret, axis=1)
        ck_matrix = construct_coefficient_matrix(self.c_tot_ret_rad*k_field_rad, shape_p_ret, axis=1)
        self.vel_matrix_ret_rad = (-k_matrix) @ self.grad_p_ret_rad
        jac_darcy += self.div_p_ret_rad @ ((-ck_matrix) @ self.grad_p_ret_rad)
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
            c = self.c
        if T is None:
            T = self.T
        if p is None:
            p = self.p    
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
            jac_ic_c_diff_perm = self.div_c_perm_rad @ ((-diff_matrix_perm_rad) @ self.grad_bc_c_perm_rad)
            jac_ic_c_diff_ret = self.div_c_ret_rad @ ((-diff_matrix_ret_rad) @ self.grad_bc_c_ret_rad)
            #jac_c_diff = self.div_c_ret_ax @ self.grad_c_ret_ax + self.div_c_ret_rad @ self.grad_c_ret_rad + self.div_c_perm_ax @ self.grad_c_perm_ax + self.div_c_perm_rad @ self.grad_c_perm_rad 
            #jac_bc_c_diff = self.div_c_ret_ax @ self.grad_bc_c_ret_ax + self.div_c_ret_rad @ self.grad_bc_c_ret_rad +self.div_c_perm_ax @ self.grad_bc_c_perm_ax + self.div_c_perm_rad @ self.grad_bc_c_perm_rad
            if (T_perm.ndim > 1):
                bc_neumann_hom = {'a': 1, 'b': 0, 'd': 0}
                _, _ ,T_perm_i, _ = compute_boundary_values(T_perm, self.r_f_perm, self.r_c_ret, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
                T_ret_i,_,_,_ = compute_boundary_values(T_ret, self.r_f_ret, self.r_c_ret, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
                P_perm_i = self.Rg*T_perm_i[...,np.newaxis]*self.perm
                P_ret_i = self.Rg*T_ret_i[...,np.newaxis]*self.perm
            else:
                P_perm_i = self.Rg*T_perm*self.perm
                P_ret_i = self.Rg*T_ret*self.perm
            ic_1 = {'a':(diff_field_perm_rad[:,-1,:],0), 'b':(P_perm_i,-P_ret_i)}
            factor_geom = (self.r_f_ret[0]/self.r_f_perm[-1])**self.nu
            ic_2 = {'a':(0,factor_geom*diff_field_ret_rad[:,0,:]), 'b':(-P_perm_i,P_ret_i)}
            interf_mat_perm, _, interf_mat_ret, _ = construct_interface_matrices((shape_c_perm, shape_c_ret), (self.r_f_perm, self.r_f_ret), ic=(ic_1, ic_2), axis=1)
            self.jac_c_diff += jac_ic_c_diff_perm @ interf_mat_perm + jac_ic_c_diff_ret @ interf_mat_ret

        g_vect[...] = self.g_bc_c_diff + self.jac_c_diff @ c.reshape((-1,1))
        return g, self.jac_c_diff
    
    def update_velocity_fields(self):
        p_vec = self.p.reshape((-1,1))
        self.u_perm_ax.reshape((-1,1))[...] = self.vel_matrix_perm_ax @ p_vec + self.vel_bc_perm_ax
        self.u_perm_rad.reshape((-1,1))[...] = self.vel_matrix_perm_rad @ p_vec
        self.u_ret_ax.reshape((-1,1))[...] = self.vel_matrix_ret_ax @ p_vec + self.vel_bc_ret_out
        self.u_ret_rad.reshape((-1,1))[...] = self.vel_matrix_ret_rad @ p_vec

        self.u_perm_ax[0,:] = self.u_perm_ax[1,:] - (self.z_f[1]-self.z_f[0])/(self.z_f[2]-self.z_f[1])*(self.u_perm_ax[2,:]-self.u_perm_ax[1,:])
        if (self.is_counter_current):
            self.u_ret_ax[-1,:] = self.u_ret_ax[-2,:] - (self.z_f[-2]-self.z_f[-1])/(self.z_f[-3]-self.z_f[-2])*(self.u_ret_ax[-3,:]-self.u_ret_ax[-2,:])
        else:
            self.u_ret_ax[0,:] = self.u_ret_ax[1,:] - (self.z_f[1]-self.z_f[0])/(self.z_f[2]-self.z_f[1])*(self.u_ret_ax[2,:]-self.u_ret_ax[1,:])        

        self.div_u = (self.div_p_perm_ax @ self.u_perm_ax.ravel() + self.div_p_perm_rad @ self.u_perm_rad.ravel() 
                      + self.div_p_ret_ax @ self.u_ret_ax.ravel() + self.div_p_ret_rad @ self.u_ret_rad.ravel()).reshape(self.T.shape)

        return self.u_perm_ax, self.u_perm_rad, self.u_ret_ax, self.u_ret_rad
    
    def construct_g_conv(self, c=None, compute_jac = False):
        if c is None:
            c = self.c
        bc_neumann_hom = {'a': 1, 'b': 0, 'd': 0}
        bc_none = {'a': 0, 'b': 0, 'd': 0}
        if (self.is_counter_current):
            bc_ret_ax = (bc_neumann_hom, bc_none)
            is_inflow = self.u_ret_ax[0,:] > 0 
        else:
            bc_ret_ax = (bc_none, bc_neumann_hom)
            is_inflow = self.u_ret_ax[-1,:] < 0
            if (np.any(is_inflow)):
                b_out = (is_inflow*1.0).reshape((1,-1,1))
                a_out = (1.0-b_out)
                d_out = b_out*self.p_ret_out/(self.Rg*self.T[[-1],self.num_r_perm:,np.newaxis])*np.array([[[0.0,1.0,0.0]]])
                bc_ret_ax = (bc_none, {'a':a_out, 'b':b_out, 'd':d_out})
        
        g = np.empty(c.shape)
        g_vect = g.ravel()
        
        c_perm = c[:, 0:self.num_r_perm, :]
        u_perm_ax = self.u_perm_ax[...,np.newaxis]
        u_perm_rad = self.u_perm_rad[...,np.newaxis]
        self.c_perm_ax,_ = interp_cntr_to_stagg_tvd(c_perm, self.z_f, self.z_c, bc = (bc_none, bc_neumann_hom), v = u_perm_ax, tvd_limiter = upwind, axis=0)
        flux_perm_ax = u_perm_ax * self.c_perm_ax
        g_vect[:] = self.div_c_perm_ax @ flux_perm_ax.ravel()
        self.c_tot_perm_ax = np.sum(self.c_perm_ax, axis=-1)
        self.c_perm_rad,_ = interp_cntr_to_stagg_tvd(c_perm, self.r_f_perm, self.r_c_perm, bc = (bc_neumann_hom, bc_neumann_hom), v = u_perm_rad, tvd_limiter = upwind, axis=1)
        flux_perm_rad = u_perm_rad * self.c_perm_rad
        g_vect[:] += self.div_c_perm_rad @ flux_perm_rad.ravel()
        self.c_tot_perm_rad = np.sum(self.c_perm_rad, axis=-1)
        
        c_ret = c[:, self.num_r_perm:, :]
        u_ret_ax = self.u_ret_ax[...,np.newaxis]
        u_ret_rad = self.u_ret_rad[...,np.newaxis]
        self.c_ret_ax,_ = interp_cntr_to_stagg_tvd(c_ret, self.z_f, self.z_c, bc = bc_ret_ax, v = u_ret_ax, tvd_limiter = upwind, axis=0)
        flux_ret_ax = u_ret_ax * self.c_ret_ax
        g_vect[:] += self.div_c_ret_ax @ flux_ret_ax.ravel()
        self.c_tot_ret_ax = np.sum(self.c_ret_ax, axis=-1)
        self.c_ret_rad,_ = interp_cntr_to_stagg_tvd(c_ret, self.r_f_ret, self.r_c_ret, bc = (bc_neumann_hom, bc_neumann_hom), v = u_ret_rad, tvd_limiter = upwind, axis=1)
        flux_ret_rad = u_ret_rad * self.c_ret_rad
        g_vect[:] += self.div_c_ret_rad @ flux_ret_rad.ravel()        
        self.c_tot_ret_rad = np.sum(self.c_ret_rad, axis=-1)
        
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
           
    def construct_g(self, c=None, c_old=None, c_tot = None, compute_jac=False):
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
        if (c_old is None):
            c_old = c.copy()
        c_ret = c[:, self.num_r_perm:, :]
        _, c_tot_ret =  self.split_perm_and_ret(c_tot)
        _, p_ret =  self.split_perm_and_ret(self.p)
        
        g_accum = (self.jac_c_accum @ (c-c_old).reshape((-1, 1))).reshape(c.shape)
        g_conv, jac_conv = self.construct_g_conv(c, compute_jac=compute_jac)
        g_diff, jac_diff = self.construct_g_diff(c, compute_jac=compute_jac)       
        p_over_c_tot = p_ret[..., np.newaxis]/c_tot_ret[..., np.newaxis]
        if (compute_jac):
            g_react, jac_react = self.numjac(lambda c: self.kinetics(c*p_over_c_tot), c_ret)
            shape_c_ret = (self.num_z, self.num_r_ret, self.num_c)
            offset = (0, self.num_r_perm, 0)
            jac_react = update_csc_array_indices(jac_react, shape_c_ret, c.shape, offset=offset)
            self._jac = self.jac_c_accum + jac_conv + jac_diff - jac_react
        else:
            g_react = self.kinetics(c_ret*p_over_c_tot)
        g = self.g_c_in.reshape(c.shape) + g_accum + g_conv + g_diff
        g = self.g_c_in.reshape(c.shape) + g_conv + g_diff
        g_ret = g[:, self.num_r_perm:, :]
        g_ret[...] -= g_react
        return g, self._jac
    
    def construct_g_test(self, c=None, c_old=None, c_tot = None, compute_jac=False):
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
        if (c_old is None):
            c_old = c.copy()
        c_ret = c[:, self.num_r_perm:, :]
        _, c_tot_ret =  self.split_perm_and_ret(c_tot)
        _, p_ret =  self.split_perm_and_ret(self.p)
        
        g_accum = (self.jac_c_accum @ (c-c_old).reshape((-1, 1))).reshape(c.shape)
        g_conv, jac_conv = self.construct_g_conv(c, compute_jac=compute_jac)
        g_diff, jac_diff = self.construct_g_diff(c, compute_jac=compute_jac)       
        p_over_c_tot = p_ret[..., np.newaxis]/c_tot_ret[..., np.newaxis]
        if (compute_jac):
            g_react, jac_react = self.numjac(lambda c: self.kinetics(c*p_over_c_tot), c_ret)
            shape_c_ret = (self.num_z, self.num_r_ret, self.num_c)
            offset = (0, self.num_r_perm, 0)
            jac_react = update_csc_array_indices(jac_react, shape_c_ret, c.shape, offset=offset)
            self._jac = self.jac_c_accum + jac_conv + jac_diff - jac_react
        else:
            g_react = self.kinetics(c_ret*p_over_c_tot)
        g = self.g_c_in.reshape(c.shape) + g_accum + g_conv + g_diff
        g = self.g_c_in.reshape(c.shape) + g_conv + g_diff
        g_ret = g[:, self.num_r_perm:, :]
        g_ret[...] -= g_react
        return g, self._jac
    
    def construct_g_T_conv(self, T=None, compute_jac = False):
        if T is None:
            T = self.T
        bc_ret_dirichlet = {'a': 0, 'b': 1, 'd': self.T_ret_in}
        bc_neumann_hom = {'a': 1, 'b': 0, 'd': 0}
        if (self.is_counter_current):
            bc_ret_ax = (bc_neumann_hom, bc_ret_dirichlet)
        else:
            bc_ret_ax = (bc_ret_dirichlet, bc_neumann_hom)
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
        
    def construct_g_T(self, T=None, T_old=None, compute_jac=False):
        if (T is None):
            T = self.T
        if (T_old is None):
            T_old = T.copy()

        g_accum = (self.jac_T_accum @ (T-T_old).reshape((-1, 1))).reshape(T.shape)
        g_conv, jac_conv = self.construct_g_T_conv(T, compute_jac=compute_jac)
        #g_diff, jac_diff = self.construct_g_T_diff(T, compute_jac=compute_jac)       
        if (compute_jac):
            self._jac_T = self.jac_T_accum + jac_conv
        g = g_accum + g_conv

        return g, self._jac_T

    def solve(self, num_timesteps=None):
        """
        Solve the system for a specified number of time steps.

        Parameters:
        - num_timesteps (int): Number of time steps to solve for.
        """

        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        c = self.c
        c_vec = c.ravel()
        T_vec = self.T.ravel()

        i=0
        cnt_p = 0
        cnt_c = 0
        while (i < num_timesteps):
            c_old = c.copy()
            T_old = self.T.copy()
            g_norm = np.inf
            for j in range(self.num_newton_iterations):
                T_ret = self.T[:, self.num_r_perm:]
                p_ret = self.p[:, self.num_r_perm:]
                self.kinetics.set_T_and_p(T_ret, p_ret)
                y = c/np.sum(c, axis=-1, keepdims=True)  # Mole fractions
                k =0
                c_tot = self.correlation.molar_density(y, self.T, self.p)
                c[...] = c_tot[...,np.newaxis]*y
                g_p_norm = np.inf
                while True:
                    g, jac = self.construct_g(c, c_old=c_old, c_tot = c_tot, compute_jac=True)
                    g_p = np.sum(g, axis=-1).reshape((-1,1))
                    g_p_norm_prev = g_p_norm
                    g_p_norm = np.linalg.norm(g_p.ravel())
                    is_stalled = np.abs(g_p_norm - g_p_norm_prev) < 1e-3 * g_p_norm
                    if k==0:
                        g_p_norm_init = g_p_norm
                    if (k == self.num_pressure_iterations) or (g_p_norm < np.maximum(self.rtol_p * g_p_norm_init, self.atol_p)) or is_stalled:
                        break
                    _, dc_tot_dp_mat = self.numjac_p(lambda p: self.correlation.molar_density(y, self.T, p), self.p, f_value=c_tot)
                    dcdp = y.reshape((-1, self.num_c)) * dc_tot_dp_mat.data.reshape((-1, 1))
                    num_rows = self.c.size
                    num_cols = self.p.size
                    row_indices = np.arange(num_rows,dtype=int)
                    col_ptrs = np.arange(0, num_rows+1, self.c.shape[-1], dtype=int)
                    dcdp_mat = csc_array((dcdp.ravel(), row_indices.ravel(), col_ptrs.ravel()), shape=(num_rows, num_cols))
                    jac_darcy = self.construct_darcy_jacobian()
                    jac_p = self.sum_c @ jac @ dcdp_mat + jac_darcy
                    p_vec = self.p.ravel()
                    #p_prev = p_vec.copy()
                    dp = -sla.spsolve(jac_p, g_p)
                    cnt_p += 1
                    p_vec[...] +=  dp
                    #alpha = 1.0
                    #while True:
                        #clip_approach(self.p, g_p)
                    p_ret = self.p[:, self.num_r_perm:]
                    self.kinetics.set_T_and_p(p = p_ret)
                    c_tot = self.correlation.molar_density(y, self.T, self.p)
                    c[...] = c_tot[...,np.newaxis]*y
                    self.update_velocity_fields()
                        #g,_ = self.construct_g(c, c_old=c_old, c_tot = c_tot, compute_jac=False)
                        #g_p = np.sum(g, axis=-1).reshape((-1,1))
                        #g_p_norm = np.linalg.norm(g_p.ravel())
                        #if g_p_norm <= (1-1e-4*alpha)*g_p_norm_prev:
                        #    break
                        #break # test
                        #alpha *= 0.5
                        #p_vec[...] = p_prev + alpha * dp
                        #if (alpha < 1e-4):
                        #    raise RuntimeError(f"Line search failed to improve residual: {g_p_norm} > {g_p_norm_prev}")
                    k += 1
                g, jac= self.construct_g(c, c_old=c_old, c_tot = c_tot, compute_jac=True)
                #g, jac= self.construct_g_test(c, c_old=c_old, c_tot = c_tot, compute_jac=True)
                g_norm_prev = g_norm
                g_norm = np.linalg.norm(g.ravel())
                is_stalled = np.abs(g_norm - g_norm_prev) < 1e-3 * g_norm
                if j==0:
                    g_norm_init = g_norm
                if (g_norm < (self.rtol*g_norm_init + self.atol)) or is_stalled:
                    break
                #g_p = np.sum(g, axis=-1).reshape((-1,1))
                #g_p_norm_prev = g_p_norm
                #g_p_norm = np.linalg.norm(g_p.ravel())
                dc = -sla.spsolve(jac, g.reshape((-1,1)))
                cnt_c += 1
                #c_prev = c_vec.copy()
                g_norm_prev = g_norm
                c_vec[...] += dc
                
                #alpha = 1.0
                #while True:
                #    g,_ = self.construct_g(c, c_old=c_old, c_tot = c_tot, compute_jac=False)
                #    g_norm = np.linalg.norm(g)
                #    if g_norm <= (1.0-1e-4*alpha)*g_norm_prev:
                #        break
                #    break # test
                #    alpha *= 0.5
                #    c_vec[...] = c_prev + dc
                #    if (alpha < 1e-4):
                #        raise RuntimeError(f"Line search failed to improve residual: {g_norm} > {g_norm_prev}")
                #clip_approach(c, g)
                #g_T, jac_T = self.construct_g_T(self.T, T_old=T_old, compute_jac=True)
                #T_vec[...] -= sla.spsolve(jac_T, g_T.reshape((-1,1)))
            y = c/np.sum(c, axis=-1, keepdims=True)  # Mole fractions
            c_tot = self.correlation.molar_density(y, self.T, self.p)
            c[...] = c_tot[...,np.newaxis]*y
            dc_norm = np.linalg.norm((c-c_old).ravel())
            if (dc_norm < (self.rtol_dc*np.linalg.norm(c) + self.atol_dc)):
                break
            i += 1
        return i, j, k, cnt_p, cnt_c
    
    def compute_fluxes_diff(self, c = None, T = None, p = None):
        shape_c_ret = (self.num_z, self.num_r_ret, self.num_c)
        shape_c_perm = (self.num_z, self.num_r_perm, self.num_c)
        if c is None:
            c = self.c
        if T is None:
            T = self.T
        if p is None:
            p = self.p    
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
        
        # test: axial dispersion zero
        diff_matrix_perm_ax *= 0
        diff_matrix_ret_ax *= 0
        
        if (T_perm.ndim > 1):
            bc_neumann_hom = {'a': 1, 'b': 0, 'd': 0}
            _, _ ,T_perm_i, _ = compute_boundary_values(T_perm, self.r_f_perm, self.r_c_ret, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
            T_ret_i,_,_,_ = compute_boundary_values(T_ret, self.r_f_ret, self.r_c_ret, bc=(bc_neumann_hom, bc_neumann_hom), axis=1)
            P_perm_i = self.Rg*T_perm_i[...,np.newaxis]*self.perm
            P_ret_i = self.Rg*T_ret_i[...,np.newaxis]*self.perm
        else:
            P_perm_i = self.Rg*T_perm*self.perm
            P_ret_i = self.Rg*T_ret*self.perm
        ic_1 = {'a':(diff_field_perm_rad[:,-1,:],0), 'b':(P_perm_i,-P_ret_i)}
        factor_geom = (self.r_f_ret[0]/self.r_f_perm[-1])**self.nu
        ic_2 = {'a':(0,factor_geom*diff_field_ret_rad[:,0,:]), 'b':(-P_perm_i,P_ret_i)}
        interf_mat_perm, _, interf_mat_ret, _ = construct_interface_matrices((shape_c_perm, shape_c_ret), (self.r_f_perm, self.r_f_ret), ic=(ic_1, ic_2), axis=1)
        c_b_perm = interf_mat_perm @ c_vect
        c_b_ret = interf_mat_ret @ c_vect
            
        fluxes_ret_ax  = (-diff_matrix_ret_ax @ (self.grad_c_ret_ax @ c_vect + self.grad_bc_c_ret_ax)).reshape((self.num_z+1, self.num_r_ret, self.num_c))
        fluxes_ret_rad = (-diff_matrix_ret_rad @ (self.grad_c_ret_rad @ c_vect + self.grad_bc_c_ret_rad @ c_b_ret)).reshape((self.num_z, self.num_r_ret+1, self.num_c))
        fluxes_perm_ax  = (-diff_matrix_perm_ax @ (self.grad_c_perm_ax @ c_vect + self.grad_bc_c_perm_ax)).reshape((self.num_z+1, self.num_r_perm, self.num_c))
        fluxes_perm_rad = (-diff_matrix_perm_rad @ (self.grad_c_perm_rad @ c_vect + self.grad_bc_c_perm_rad @ c_b_perm)).reshape((self.num_z, self.num_r_perm+1, self.num_c))

        return fluxes_ret_ax, fluxes_ret_rad, fluxes_perm_ax, fluxes_perm_rad
    
    def compute_fluxes_conv(self, c=None, compute_jac = False):
        if c is None:
            c = self.c
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
        return fluxes_ret_ax, fluxes_ret_rad, fluxes_perm_ax, fluxes_perm_rad
 
    
    def compute_flows(self):
        fluxes_diff_ret_ax, fluxes_diff_ret_rad, fluxes_diff_perm_ax, fluxes_diff_perm_rad = self.compute_fluxes_diff()
        fluxes_ret_ax, fluxes_ret_rad, fluxes_perm_ax, fluxes_perm_rad = self.compute_fluxes_conv()
        fluxes_ret_ax += fluxes_diff_ret_ax
        fluxes_ret_rad += fluxes_diff_ret_rad
        fluxes_perm_ax += fluxes_diff_perm_ax
        fluxes_perm_rad += fluxes_diff_perm_rad        
        fluxes_perm_ax[0,:,:] = self.flux_perm_in[0,:,:]
        if (self.is_counter_current):
            fluxes_ret_ax[-1,:,:] = -self.flux_ret_in[0,:,:]
        else:
            fluxes_ret_ax[0,:,:] = self.flux_ret_in[0,:,:]
        
        # Compute cross-sectional areas for radial and axial directions
        dr_sq_ret = (self.r_f_ret[1:]**2 - self.r_f_ret[:-1]**2).reshape((1,-1,1))
        dr_sq_perm = (self.r_f_perm[1:]**2 - self.r_f_perm[:-1]**2).reshape((1,-1,1))
        dz = (self.z_f[1:] - self.z_f[:-1]).reshape((-1,1,1))

        flows_ret_ax = np.pi*np.sum(fluxes_ret_ax*dr_sq_ret, axis=1)
        flows_ret_rad = 2.0*np.pi*self.r_f_ret.reshape((-1,1))*np.sum(fluxes_ret_rad*dz, axis=0)
        flows_perm_ax = np.pi*np.sum(fluxes_perm_ax*dr_sq_perm, axis=1)
        flows_perm_rad = 2.0*np.pi*self.r_f_perm.reshape((-1,1))*np.sum(fluxes_perm_rad*dz, axis=0)

        return flows_ret_ax, flows_ret_rad, flows_perm_ax, flows_perm_rad
    
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
