import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import json
import math
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
def calculate_flows(GHSV, eps, Dcat, rho_c, L_membrane, Lsealing, r_min, Nm, sweep_ratio, H2_N2_ratio, T_STP = 273.15, P_STP = 101325.0):

    # 1. Parameter calculation for reactor design
    s=0.05                      #minimum clear distance between membrane surfaces
    P_t=2*r_min + s             #Center‑to‑center pitch
    C_layout = 0.907            #Triangular pitch
    D_bundle = P_t * math.sqrt(Nm /C_layout )
    D_reactor =D_bundle + 2* s
    r_max = D_reactor/2
    L_bed = L_membrane + Lsealing      #catalyst bed
    A_reactor = np.pi * ((D_reactor/2)**2 - r_min**2 * Nm)
    A_membrane = 2 * np.pi * r_min * L_membrane * Nm   # to be used when the permeating flux wants to be calculate from the permeance
    vol_reactor = A_reactor * L_bed 
    vol_cat = Dcat * (1 - eps) * vol_reactor
    W_cat = vol_cat * rho_c
    
    S_over_V = 2 * (r_min *Nm) /(r_max**2 - Nm * r_min**2)   #Membrane Area/reactor volume parameter
    
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
    #Re_ret = rho_ret * self.dp * np.abs(u_ret_i) / visc_ret
    #Pr_ret = visc_ret * cp_ret / lmbda_ret_rad[:,[0]]
    #Re_perm = rho_perm * d_tube * np.abs(u_perm_i) / visc_perm
    #Pr_perm = visc_perm * cp_perm / lmbda_perm_rad[:,[-1]]
    #Sc_ret = viscosity / rho_g / np.mean(D)
    #Pe_ret = Re_ret * Sc_ret
    #Da_local = rate / flows_ret_ax
    #Da_z = np.mean(Da_local, axis=0)  # axial avg

    # Shwerwood correlation to be fitted
    #calculate mass transfer coefficient (k) from flux at membrane interface and concentration

    
    return F_ret_in, F_perm_in, y_H2_in, y_N2_in, W_cat, D_reactor, WHSV, S_over_V, vol_reactor, A_membrane, A_reactor

# --- Main Execution Function ---
def run_case_studies(csv_path="case_studies_multiplemem.csv"):
    # 1. Load the CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    results = []

    # 2. Loop over each case
    for index, row in df.iterrows():
        case_id = row['Case_ID']
        print(f"\n=== Running Case: {case_id} ({row['Description']}) ===")
        
        # Create Output Directory
        out_dir = os.path.join("results", case_id)
        os.makedirs(out_dir, exist_ok=True)
        
        # -- Geometry -- (CSV file)
        r_min = DEFAULTS['r_min']
        #r_max = row['r_max_m']
        Nm = row ['N_mem']   #added
        L_membrane = row ['L_m']   #added
        Lsealing = DEFAULTS['Lsealing']    #added
        Dcat = row ['Dcat']   #added

        # -- Bed / catalyst properties from DEFAULTS --
        eps = DEFAULTS['eps']    #added
        rho_c = DEFAULTS['rho_c']    #added

        # -- Flows & Composition --
        GHSV = row['GHSV_h']
        sweep_ratio = row['Sweep_Ratio']
        H2_N2_ratio = row['H2_N2_ratio']

        F_ret_in, F_perm_in, y_H2_in, y_N2_in, W_cat, D_reactor, WHSV, S_over_V, vol_reactor, A_membrane, A_reactor, r_max = calculate_flows( GHSV=GHSV, eps=eps, rho_c=rho_c, sweep_ratio=sweep_ratio, H2_N2_ratio=H2_N2_ratio, r_min=r_min, Nm=Nm, Lsealing=Lsealing, L_membrane=L_membrane, Dcat=Dcat )
        y_ret_in = np.array([y_H2_in, y_N2_in, 0.0])

        p_ret_out = row['p_ret_bar'] * 1e5
        p_perm = row ['p_perm'] * 1e5  #added
        T_ret_in = row['T_ret_K']
        T_perm_in = row['T_perm_K']
        is_counter_current = row['Is_Counter_Current']
   
        # 4. Initialize and Solve
        try:
            reactor = MembraneReactor(
                L=L_membrane,
                r_min=r_min,
                # r_max=r_max,
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
            

            #OUTPUTS
            #compute key performance indicators
            F_H2_ret_z = flows_ret_ax[:, 0]
            F_H2_perm_z = flows_perm_ax[:, 0]

            F_NH3_ret_z = flows_ret_ax[:, 2]
            F_NH3_perm_z = flows_perm_ax[:, 2]

            F_H2_in_ret = F_H2_ret_z[0]
            F_H2_in_perm = F_H2_perm_z[0]
            F_H2_in_tot  = F_H2_in_ret + F_H2_in_perm

            # net H2 that has gone into permeate up to each z
            F_H2_tm = F_H2_perm_z - F_H2_perm_z[0]       # array

            # only count segments where net flow is from retentate to permeate
            F_H2_tm_star = np.where(F_H2_tm <= 0.0, F_H2_tm, 0.0)
            
            F_H2_out = flows_ret_ax[-1, 0]         # H2 out at retentate outlet
            F_NH3_in = flows_ret_ax[0, 2]        # NH3 at outlet
            F_NH3_out = flows_ret_ax[-1, 2]        # NH3 at outlet

            # H2 conversion (array vs z)
            X_H2 = (F_H2_in_ret - F_H2_ret_z - F_H2_tm) / (F_H2_in_ret - F_H2_tm_star)

            # NH3 productivity (mol/Kgcat/s) at outlet
            NH3_prod = F_NH3_out/W_cat

            # NH3 recovery in permeate (array vs z)
            NH3_rec = F_NH3_perm_z / (F_NH3_perm_z + F_NH3_ret_z)

            # NH3 yield based on total fed H2
            NH3_yield = 3 *(F_NH3_perm_z + F_NH3_ret_z) / ( 2 * F_H2_in_tot)

            # Example: average H2 flux at membrane
            J_H2 = flows_ret_mem[:, 0] / reactor.A_membrane 
            J_H2_avg = float(np.mean(J_H2))

            # Example: maximum temperature rise
            DeltaT_max = float(np.max(reactor.T - reactor.T_ret_in))

            # ---- KPIs as scalars for the table ----

            # H2 conversion: take outlet value (last z)
            X_H2_out = float(X_H2[-1])

            # NH3 productivity at outlet (already scalar)
            NH3_prod_out = float(NH3_prod)

            # NH3 recovery in permeate: outlet value
            NH3_rec_out = float(NH3_rec[-1])

            # NH3 yield based on total fed H2: outlet value
            NH3_yield_out = float(NH3_yield[-1])

            # already scalar:
            J_H2_avg = float(J_H2_avg)
            DeltaT_max = float(DeltaT_max)

            results.append(dict(
            # INPUTS (to identify the case)
            Case_ID = case_id,
            Description = row["Description"],
            GHSV_h = GHSV,
            Nm = Nm,
            L_m = L_membrane,
            Dcat = Dcat,
            p_ret_bar = row["p_ret_bar"],
            p_perm_bar = row["p_perm_bar"],
            T_ret_K = T_ret_in,
            T_perm_K = T_perm_in,
            Sweep_Ratio = sweep_ratio,
            H2_N2_ratio = H2_N2_ratio,
            D_reactor_m = D_reactor,
            WHSV = WHSV,
            S_over_V = S_over_V,
            vol_reactor_m3 = vol_reactor,
            A_membrane_m2 = A_membrane,
            A_reactor_m2 = A_reactor,
            r_max = r_max,

            # OUTPUTS (KPIs)
            X_H2_out=X_H2_out,
            NH3_prod_out=NH3_prod_out,
            NH3_rec_out=NH3_rec_out,
            NH3_yield_out=NH3_yield_out,
            J_H2_avg=J_H2_avg,
            DeltaT_max=DeltaT_max
            ))
            

            print(f'axial flows retentate side: left {flows_ret_ax[0,:]} right {flows_ret_ax[-1,:]}')
            print(f'axial flows permeate side: left {flows_perm_ax[0,:]} right {flows_perm_ax[-1,:]}')
            flows_tot = flows_ret_ax[0,:] + flows_perm_ax[0,:] -  flows_ret_ax[-1,:] - flows_perm_ax[-1,:]
            print(f'total balance per component: {flows_tot}')
            print(f'elemental H balance: {2*flows_tot[0] + 3*flows_tot[2]}')
            print(f'elemental N balance: {2*flows_tot[1] + flows_tot[2]}')

        except Exception as e:
            print(f"Case {case_id} failed: {e}")
    
    if results:
       df_out = pd.DataFrame(results)
       os.makedirs("results", exist_ok=True)
       out_csv = os.path.join("results", "summary_results.csv")
       df_out.to_csv(out_csv, index=False)
       print(f"\nSaved summary results to: {out_csv}")
    else:
       print("No successful cases, no summary file written.")

if __name__ == "__main__":
    run_case_studies()