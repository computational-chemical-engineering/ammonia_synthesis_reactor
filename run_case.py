import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import json
import types

# Import your model modules
from membrane_reactor import MembraneReactor
from defaults import DEFAULTS

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        # 1. Handle NumPy generic integers/floats
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        
        # 2. Handle NumPy Arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # 3. Handle Functions and Lambdas (The fix for your issue)
        #elif isinstance(obj, (types.FunctionType, types.LambdaType)) or callable(obj):
            # Return the string representation (e.g. "<function <lambda> at ...>")
            # or you can return "Skipped Function" if you prefer
        #    return str(obj)
        
        # 4. Fallback for other non-serializable objects
        return super().default(obj)

# --- Helper Function: Convert GHSV to Molar Flows ---
def calculate_flows(GHSV, eps, Dcat, rho_c, L_membrane, Lsealing, r_max, r_min, Nm, sweep_ratio, H2_N2_ratio, T_STP = 273.15, P_STP = 101325.0):

    # 1. Parameter calculation for reactor design
    L_bed = L_membrane + Lsealing      #catalyst bed
    A_reactor = np.pi * (r_max**2 - r_min**2 * Nm)
    A_membrane = 2 * np.pi * r_min * L_membrane * Nm   # to be used when the permeating flux wants to be calculate from the permeance
    vol_reactor = A_reactor * L_bed 
    vol_cat = Dcat * (1 - eps) * vol_reactor
    W_cat = vol_cat * rho_c
    
    #Membrane Area/reactor volume parameter
    S/V = 2 * (r_min *Nm) /(r_max**2 - Nm * r_min**2)
    
    """
    Converts GHSV (1/h) and Geometry to F_ret_in and F_perm_in (mol/s).
    Includes H2:N2 ratio logic.
    """   

    # 2. Volumetric Flow at STP (Standard T=273.15K, P=101325 Pa)
    vol_flow_std = (GHSV * vol_cat) / 3600.0

    WHSV = vol_flow_std / W_cat    #another way to express the gas space velocity

    # 3. Total Molar Flow (Ideal Gas Law at STP)
    F_ret_in = (P_STP * vol_flow_std) / (const.R * T_STP)
    
    
    # 4. Species Mole Fractions based on H2:N2 Ratio
    # Ratio R = H2/N2 -> H2 = R*N2 -> x_H2 + x_N2 = 1 (assuming pure feed for simplicity)
    # x_N2 * (R + 1) = 1  => x_N2 = 1 / (R + 1)
    y_N2_in = 1.0 / (H2_N2_ratio + 1.0)
    y_H2_in = 1.0 - y_N2_in
    
    # Note: If your defaults.py requires specific species fractions, 
    # you might need to adjust DEFAULTS['x_ret_in'] separately.
    
    # 5. Calculate Permeate Flow
    F_perm_in = F_ret_in * sweep_ratio

    # 6. Dimensionless number
    Re_ret = rho_ret * self.dp * np.abs(u_ret_i) / visc_ret
    Pr_ret = visc_ret * cp_ret / lmbda_ret_rad[:,[0]]
    Re_perm = rho_perm * d_tube * np.abs(u_perm_i) / visc_perm
    Pr_perm = visc_perm * cp_perm / lmbda_perm_rad[:,[-1]]
    Sc_ret = viscosity / rho_g / np.mean(D)
    Pe_ret = Re_ret * Sc_ret
    Da_local = rate / flows_ret_ax
    Da_z = np.mean(Da_local, axis=0)  # axial avg

    # Shwerwood correlation to be fitted
    #calculate mass transfer coefficient (k) from flux at membrane interface and concentration

    
    return F_ret_in, F_perm_in, y_H2_in, y_N2_in

# --- Main Execution Function ---
def run_case_studies(csv_path="case_studies.csv"):
    # 1. Load the CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # 2. Loop over each case
    for index, row in df.iterrows():
        case_id = row['Case_ID']
        print(f"\n=== Running Case: {case_id} ({row['Description']}) ===")
        
        # Create Output Directory
        out_dir = os.path.join("results", case_id)
        os.makedirs(out_dir, exist_ok=True)
        
        # -- Geometry -- (CSV file)
        r_min = DEFAULTS['r_min']
        r_max = row['r_max_m']
        Nm = row ['Nm']   #added
        L_membrane = row ['L_membrane']   #added
        Lsealing = DEFAULTS['Lsealing']    #added
        Dcat = row ['Dcat']   #added
        

        # -- Flows & Composition --
        GHSV = row['GHSV_h']
        sweep_ratio = row['Sweep_Ratio']
        H2_N2_ratio = row['H2_N2_ratio']
        F_ret_in, F_perm_in, y_H2_in, y_N2_in = calculate_flows( GHSV=GHSV, sweep_ratio=sweep_ratio, H2_N2_ratio=H2_N2_ratio, r_max=r_max, Nm=Nm, L_membrane=L_membrane, Dcat=Dcat )
        y_ret_in = np.array([y_H2_in, y_N2_in, 0.0])

        p_ret_out = row['p_ret'] * 1e5
        p_perm = row ['p_perm']   #added
        T_ret_in = row['T_ret_K']
        T_perm_in = row['T_perm_K']
        is_counter_current = row['Is_Counter_Current']
   
        # 4. Initialize and Solve
        try:
            reactor = MembraneReactor(
                L=L,
                r_min=r_min,
                r_max=r_max,
                p_ret_out=p_ret_out,
                T_ret_in=T_ret_in,
                T_perm_in=T_perm_in,
                F_ret_in=F_ret_in,
                F_perm_in=F_perm_in,
                y_ret_in=y_ret_in,
                is_counter_current=is_counter_current
            )
            reactor.solve()
            # 5. Save Raw Data
            # Save configuration for reproducibility
            with open(os.path.join(out_dir, "config.json"), 'w') as f:
                # Filter out non-serializable items (like numpy arrays) if necessary
                json.dump(reactor.param_dict, f, indent=4, cls=SafeEncoder)
            
            flows_ret_ax, flows_ret_mem, flows_perm_ax, flows_perm_mem = reactor.compute_flows()

            print(f'axial flows retentate side: left {flows_ret_ax[0,:]} right {flows_ret_ax[-1,:]}')
            print(f'axial flows permeate side: left {flows_perm_ax[0,:]} right {flows_perm_ax[-1,:]}')
            flows_tot = flows_ret_ax[0,:] + flows_perm_ax[0,:] -  flows_ret_ax[-1,:] - flows_perm_ax[-1,:]
            print(f'total balance per component: {flows_tot}')
            print(f'elemental H balance: {2*flows_tot[0] + 3*flows_tot[2]}')
            print(f'elemental N balance: {2*flows_tot[1] + flows_tot[2]}')

        except Exception as e:
            print(f"Case {case_id} failed: {e}")

if __name__ == "__main__":
    run_case_studies()
