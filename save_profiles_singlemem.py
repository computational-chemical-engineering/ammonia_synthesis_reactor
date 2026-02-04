import os
import numpy as np
import pandas as pd
from membrane_reactor import MembraneReactor
from defaults import DEFAULTS
from run_case_singlemembrane import calculate_flows  # or copy the function

def save_profiles(csv_path="case_studies_singlemem.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    for _, row in df.iterrows():
        case_id = row["Case_ID"]
        print(f"\n=== Saving profiles for Case: {case_id} ({row['Description']}) ===")

        out_dir = os.path.join("results", case_id)
        os.makedirs(out_dir, exist_ok=True)

        # geometry and properties
        r_min      = DEFAULTS["r_min"]
        Nm         = row["N_mem"]
        L_membrane = row["L_m"]
        Lsealing   = DEFAULTS["Lsealing"]
        Dcat       = row["Dcat"]
        eps        = DEFAULTS["eps"]
        rho_c      = DEFAULTS["rho_c"]

        # flows & composition
        GHSV        = row["GHSV_h"]
        sweep_ratio = row["Sweep_Ratio"]
        H2_N2_ratio = row["H2_N2_ratio"]

        F_ret_in, F_perm_in, y_H2_in, y_N2_in, W_cat, D_reactor, WHSV, S_over_V, vol_reactor, A_membrane, A_reactor, r_max = calculate_flows(
            GHSV=GHSV,
            eps=eps,
            Dcat=Dcat,
            rho_c=rho_c,
            L_membrane=L_membrane,
            Lsealing=Lsealing,
            r_min=r_min,
            Nm=Nm,
            sweep_ratio=sweep_ratio,
            H2_N2_ratio=H2_N2_ratio
        )

        y_ret_in = np.array([y_H2_in, y_N2_in, 0.0])

        p_ret_out        = row["p_ret_bar"] * 1e5
        p_perm           = row["p_perm"] * 1e5
        T_ret_in         = row["T_ret_K"]
        T_perm_in        = row["T_perm_K"]
        is_counter_current = row["Is_Counter_Current"]

        try:
            reactor = MembraneReactor(
                L=L_membrane,
                r_min=r_min,
                p_ret_out=p_ret_out,
                T_ret_in=T_ret_in,
                T_perm_in=T_perm_in,
                F_ret_in=F_ret_in,
                F_perm_in=F_perm_in,
                y_ret_in=y_ret_in,
                is_counter_current=is_counter_current
            )
            reactor.solve()
            flows_ret_ax, flows_ret_mem, flows_perm_ax, flows_perm_mem = reactor.compute_flows()

            # fluxes_ret_ax, fluxes_ret_rad, fluxes_perm_ax, fluxes_perm_rad  ---- this can be taken directly from compute_flows and be added in the membrane reactor script

            # axial grid
            z = reactor.z
            # radial grid
            r = reactor.r

            # 1D profiles
            F_H2_ret_z   = flows_ret_ax[:, 0]
            F_N2_ret_z   = flows_ret_ax[:, 1]
            F_NH3_ret_z  = flows_ret_ax[:, 2]
            F_H2_perm_z  = flows_perm_ax[:, 0]
            F_N2_perm_z   = flows_perm_ax[:, 1]
            F_NH3_perm_z = flows_perm_ax[:, 2]
            #y_H2_ret_z = y_ret [:, 0]   #to recall from the main script
            #y_N2_ret_z = y_ret [:, 1]   #to recall from the main script
            #y_NH3_ret_z = y_ret [:, 2]   #to recall from the main script
            #y_H2_perm_z = y_perm [:, 0]   #to recall from the main script
            #y_N2_perm_z = y_perm [:, 1]    #to recall from the main script
            #y_NH3_perm_z = y_perm [:, 2]   #to recall from the main script


            #2D profiles
            J_H2_ret_ax_zr = fluxes_ret_ax[:, :, 0]
            J_N2_ret_ax_zr = fluxes_ret_ax[:, :, 1]
            J_NH3_ret_ax_zr = fluxes_ret_ax[:, :, 2]
            J_H2_ret_rad_zr = fluxes_ret_rad[:, :, 0]
            J_N2_ret_rad_zr = fluxes_ret_rad[:, :, 1]
            J_NH3_ret_rad_zr = fluxes_ret_rad[:, :, 2]

            y_H2_ret_zr = y_ret[:, :, 0]
            y_N2_ret_zr = y_ret[:, :, 1]
            y_NH3_ret_zr = y_ret[:, :, 2]

            u_ret_ax = reactor.u_ret_ax
            u_ret_rad = reactor.u_ret_rad 

            T_ret= reactor.T_ret   #check name within the membrane reactor script  ---  shape (nz, nr) → 2D temperature
            T_perm = reactor.T_perm   #check name within the membrane reactor script ---  shape (nz, nr) → 2D temperature

            #KPI
            F_H2_in_ret  = F_H2_ret_z[0]  # it come from integrating local fluxes over area, so all radial information is already “summed into” 
            F_H2_in_perm = F_H2_perm_z[0]
            F_H2_in_tot  = F_H2_in_ret + F_H2_in_perm
            F_H2_tm = F_H2_perm_z - F_H2_perm_z[0]
            F_H2_tm_star = np.where(F_H2_tm <= 0.0, F_H2_tm, 0.0)

            X_H2 = (F_H2_in_ret - F_H2_ret_z - F_H2_tm) / (F_H2_in_ret - F_H2_tm_star)
            NH3_rec = F_NH3_perm_z / (F_NH3_perm_z + F_NH3_ret_z)
            NH3_yield = 3.0 * (F_NH3_perm_z + F_NH3_ret_z) / (2.0 * F_H2_in_tot)

            #dimensionless number
            #Re
            #Pe
            #...
            
            # save profiles as CSVs
            np.savetxt(os.path.join(out_dir, "z_grid.csv"),
                       z, delimiter=",", header="z", comments="")

            np.savetxt(os.path.join(out_dir, "r_grid.csv"),
                     r, delimiter=",", header="r", comments="")

            np.savetxt(os.path.join(out_dir, "F_H2_ret_vs_z.csv"),
                       np.column_stack([z, F_H2_ret_z]),
                       delimiter=",", header="z,F_H2_ret", comments="")

            np.savetxt(os.path.join(out_dir, "F_NH3_ret_vs_z.csv"),
                       np.column_stack([z, F_NH3_ret_z]),
                       delimiter=",", header="z,F_NH3_ret", comments="")

            np.savetxt(os.path.join(out_dir, "F_H2_perm_vs_z.csv"),
                       np.column_stack([z, F_H2_perm_z]),
                       delimiter=",", header="z,F_H2_perm", comments="")

            np.savetxt(os.path.join(out_dir, "F_NH3_perm_vs_z.csv"),
                       np.column_stack([z, F_NH3_perm_z]),
                       delimiter=",", header="z,F_NH3_perm", comments="")

            np.savetxt(os.path.join(out_dir, "X_H2_vs_z.csv"),
                       np.column_stack([z, X_H2]),
                       delimiter=",", header="z,X_H2", comments="")

            np.savetxt(os.path.join(out_dir, "NH3_rec_vs_z.csv"),
                       np.column_stack([z, NH3_rec]),
                       delimiter=",", header="z,NH3_rec", comments="")

            np.savetxt(os.path.join(out_dir, "NH3_yield_vs_z.csv"),
                       np.column_stack([z, NH3_yield]),
                       delimiter=",", header="z,NH3_yield", comments="")
            
            np.savetxt(os.path.join(out_dir, "T_z_r.csv"),
                       T_ret, delimiter=",")

            np.savetxt(os.path.join(out_dir, "T_z_p.csv"),
                       T_perm, delimiter=",")
            
            np.savetxt(os.path.join(out_dir, "J_H2_ret_ax_zr.csv"),
                        J_H2_ret_ax_zr, delimiter=",")
            
            np.savetxt(os.path.join(out_dir, "J_N2_ret_ax_zr.csv"),
                       J_N2_ret_ax_zr, delimiter=",")

            np.savetxt(os.path.join(out_dir, "J_NH3_ret_ax_zr.csv"),
                      J_NH3_ret_ax_zr, delimiter=",")
            
            np.savetxt(os.path.join(out_dir, "J_H2_ret_rad_zr.csv"),
                       J_H2_ret_rad_zr, delimiter=",")
            
            np.savetxt(os.path.join(out_dir, "J_N2_ret_rad_zr.csv"),
                       J_N2_ret_rad_zr, delimiter=",")

            np.savetxt(os.path.join(out_dir, "J_NH3_ret_rad_zr.csv"),
                       J_NH3_ret_rad_zr, delimiter=",")

            np.savetxt(os.path.join(out_dir, "u_ret_ax_zr.csv"), 
                        u_ret_ax, delimiter=",")

            np.savetxt(os.path.join(out_dir, "u_ret_rad_zr.csv"), 
                         u_ret_rad, delimiter=",")


            print(f"Saved profiles for case {case_id} in {out_dir}")

        except Exception as e:
            print(f"Case {case_id} failed while saving profiles: {e}")

if __name__ == "__main__":
    save_profiles()