import os
import csv
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
        elif isinstance(obj, (types.FunctionType, types.LambdaType)) or callable(obj):
            # Return the string representation (e.g. "<function <lambda> at ...>")
            # or you can return "Skipped Function" if you prefer
            return str(obj)
        
        # 4. Fallback for other non-serializable objects
        return super().default(obj)

# --- Helper Function: Convert GHSV to Molar Flows ---
def calculate_flows(GHSV, vol_reactor, sweep_ratio, H2_N2_ratio, T_STP = 273.15, P_STP = 101325.0):
    """
    Converts GHSV (1/h) and Geometry to F_ret_in and F_perm_in (mol/s).
    Includes H2:N2 ratio logic.
    """    
    # 1. Volumetric Flow at STP (Standard T=273.15K, P=101325 Pa)
    vol_flow_std = (GHSV * vol_reactor) / 3600.0
    
    # 2. Total Molar Flow (Ideal Gas Law at STP)
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
    
    return F_ret_in, F_perm_in, y_H2_in, y_N2_in

# --- Main Execution Function ---
def run_case_studies(csv_path="debug.csv"):
    # 1. Load the CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Define output CSV path and header
    summary_csv_path = "case_studies_summary.csv"
    summary_header = [
        'Case_ID', 'Description',
        'Ret_Left_H2', 'Ret_Left_N2', 'Ret_Left_NH3',
        'Ret_Right_H2', 'Ret_Right_N2', 'Ret_Right_NH3',
        'Perm_Left_H2', 'Perm_Left_N2', 'Perm_Left_NH3',
        'Perm_Right_H2', 'Perm_Right_N2', 'Perm_Right_NH3',
        'Bal_H2', 'Bal_N2', 'Bal_NH3',
        'Elem_H_Bal', 'Elem_N_Bal'
    ]

    # Initialize the summary CSV file with header
    # We open in 'w' mode to overwrite or start fresh. 
    # If appending to existing logs is desired across multiple runs, 'a' could be used, 
    # but usually a clean start per run is safer unless specified otherwise.
    with open(summary_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(summary_header)

    # 2. Loop over each case
    for index, row in df.iterrows():
        case_id = row['Case_ID']
        print(f"\n=== Running Case: {case_id} ({row['Description']}) ===")
        
        # Create Output Directory
        out_dir = os.path.join("results", case_id)
        os.makedirs(out_dir, exist_ok=True)
        
        # -- Geometry --
        L = row['L_m']
        r_min = DEFAULTS['r_min']
        r_max = row['r_max_m']
        vol_reactor = np.pi * (r_max**2 - r_min**2) * L # volume of annular retentate side

        # -- Flows & Composition --
        GHSV = row['GHSV_h']
        sweep_ratio = row['Sweep_Ratio']
        H2_N2_ratio = row['H2_N2_ratio']
        F_ret_in, F_perm_in, y_H2_in, y_N2_in = calculate_flows(
            GHSV=GHSV,
            vol_reactor=vol_reactor,
            sweep_ratio=sweep_ratio,
            H2_N2_ratio=H2_N2_ratio
        )
        y_ret_in = [y_H2_in, y_N2_in, 0.0]
        print(GHSV, sweep_ratio, H2_N2_ratio, y_ret_in)

        p_ret_out = row['p_ret_bar'] * 1e5
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
            elem_H_bal = 2*flows_tot[0] + 3*flows_tot[2]
            elem_N_bal = 2*flows_tot[1] + flows_tot[2]
            print(f'elemental H balance: {elem_H_bal}')
            print(f'elemental N balance: {elem_N_bal}')

            # Append results to CSV
            with open(summary_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                row_data = [
                    case_id, row['Description'],
                    # Retentate Left (Inlet) - Index 0
                    flows_ret_ax[0, 0], flows_ret_ax[0, 1], flows_ret_ax[0, 2],
                    # Retentate Right (Outlet) - Index -1
                    flows_ret_ax[-1, 0], flows_ret_ax[-1, 1], flows_ret_ax[-1, 2],
                    # Permeate Left - Index 0
                    flows_perm_ax[0, 0], flows_perm_ax[0 , 1], flows_perm_ax[0, 2],
                    # Permeate Right - Index -1
                    flows_perm_ax[-1, 0], flows_perm_ax[-1, 1], flows_perm_ax[-1, 2],
                    # Balances
                    flows_tot[0], flows_tot[1], flows_tot[2],
                    elem_H_bal, elem_N_bal
                ]
                writer.writerow(row_data)

        except Exception as e:
            print(f"Case {case_id} failed: {e}")

if __name__ == "__main__":
    run_case_studies()