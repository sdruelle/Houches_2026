import os
import numpy as np
import matplotlib.pyplot as plt
import re
import h5py

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

base_path = "/Xnfs/Houches2026/sdruelle/post_process"

sim_dirs = [f.path for f in os.scandir(base_path) if f.is_dir() and not f.name.startswith('.')]
sim_dirs.sort()

all_spectra = {}

for sim_path in sim_dirs:  
    print(f"\n==============================================")
    print(f" Processing : {sim_path}")
    print(f"================================================")
        
    print("... Stacking spectra")
    stacks = {
        "v_tot": 0.0, "vs": 0.0, "vc": 0.0,
        "kr_tot": 0.0, "krs": 0.0, "krc": 0.0,
        "ekin_tot": 0.0, "ekins": 0.0, "ekinc": 0.0,
        "rho_tot": 0.0, "rho_par": 0.0, "rho_perp": 0.0
    }

    h5_files = [f.path for f in os.scandir(sim_path) if f.is_file() and f.name.endswith('.h5')]

    if not h5_files:
        print(f" ---> No .h5 files found. Skipping.")
        continue
    
    kbins_extracted = False 

    for filename in h5_files:
        with h5py.File(filename, 'r') as f:
            
            match = re.search(r"(\d+)\.h5", filename)
            if match:
                output_number = match.group(1)
                iout = f"out_{output_number.zfill(5)}"
            else:
                print(f"Warning: Could not parse output number from {filename}")
                continue

            # Velocity
            stacks["v_tot"] += f[iout]["d3"]["velocity"]["pspec"][:]
            stacks["vs"]    += f[iout]["d3"]["velocity_s"]["pspec"][:]
            stacks["vc"]    += f[iout]["d3"]["velocity_c"]["pspec"][:]
            
            # Kritsuk
            stacks["kr_tot"] += f[iout]["d3"]["kr"]["pspec"][:]
            stacks["krs"]    += f[iout]["d3"]["krs"]["pspec"][:]
            stacks["krc"]    += f[iout]["d3"]["krc"]["pspec"][:]
            
            # Kinetic energy
            stacks["ekin_tot"] += f[iout]["d3"]["ekin"]["pspec"][:]
            stacks["ekins"]    += f[iout]["d3"]["ekins"]["pspec"][:]
            stacks["ekinc"]    += f[iout]["d3"]["ekinc"]["pspec"][:]
            
            # Density
            stacks["rho_tot"]  += f[iout]["d3"]["density"]["pspec"][:]
            stacks["rho_par"]  += f[iout]["d3"]["density"]["pspec_Bpar"][:]
            stacks["rho_perp"] += f[iout]["d3"]["density"]["pspec_Bperp"][:]

            if not kbins_extracted:
                raw_kbins = f[iout]["d3"]["velocity_c"]["kbins"][:]
                kbins = 0.5 * (raw_kbins[:-1] + raw_kbins[1:])
                kbins_extracted = True

    # Temporal mean
    for key in stacks:
        stacks[key] /= len(h5_files)
        
    stacks["kbins"] = kbins
    all_spectra[sim_path] = stacks

print("\n... Plotting")

def plot_quantity(ax, quantity_key, title, ylabel, k_power):
    for sim_path, data in all_spectra.items():
        k = data["kbins"]
        pspec_val = data[quantity_key]
        
        # Extract just the folder name for the legend
        folder_name = os.path.basename(sim_path)
        
        ax.loglog(k, (k**k_power) * pspec_val, label=folder_name, linewidth=1.5)
    
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize='small')

# VELOCITY SPECTRA
fig_v_tot, ax_v_tot = plt.subplots(1, 1, figsize=(6, 5))
plot_quantity(ax_v_tot, "v_tot", "Velocity Total", r"$k^4 P(k)$", 4)
plt.tight_layout()
fig_v_tot.savefig("v_tot.png")

fig_v_comp, axs_v_comp = plt.subplots(1, 2, figsize=(12, 5))
plot_quantity(axs_v_comp[0], "vs", "Solenoidal", r"$k^4 P(k)$", 4)
plot_quantity(axs_v_comp[1], "vc", "Compressive", r"$k^4 P(k)$", 4)
plt.tight_layout()
fig_v_comp.savefig("v_components.png")


# KRITSUK SPECTRA
fig_kr_tot, ax_kr_tot = plt.subplots(1, 1, figsize=(6, 5))
plot_quantity(ax_kr_tot, "kr_tot", "Kritsuk Total", r"$k^{3.5} P(k)$", 3.5)
plt.tight_layout()
fig_kr_tot.savefig("kr_tot.png")

fig_kr_comp, axs_kr_comp = plt.subplots(1, 2, figsize=(12, 5))
plot_quantity(axs_kr_comp[0], "krs", "Solenoidal", r"$k^{3.5} P(k)$", 3.5)
plot_quantity(axs_kr_comp[1], "krc", "Compressive", r"$k^{3.5} P(k)$", 3.5)
plt.tight_layout()
fig_kr_comp.savefig("kr_components.png")


# KINETIC ENERGY SPECTRA
fig_ek_tot, ax_ek_tot = plt.subplots(1, 1, figsize=(6, 5))
plot_quantity(ax_ek_tot, "ekin_tot", "Kinetic Energy Total", r"$k^{3.5} P(k)$", 3.5)
plt.tight_layout()
fig_ek_tot.savefig("ekin_tot.png")

fig_ek_comp, axs_ek_comp = plt.subplots(1, 2, figsize=(12, 5))
plot_quantity(axs_ek_comp[0], "ekins", "Solenoidal", r"$k^{3.5} P(k)$", 3.5)
plot_quantity(axs_ek_comp[1], "ekinc", "Compressive", r"$k^{3.5} P(k)$", 3.5)
plt.tight_layout()
fig_ek_comp.savefig("ekin_components.png")


# DENSITY SPECTRA
fig_rho_tot, ax_rho_tot = plt.subplots(1, 1, figsize=(6, 5))
plot_quantity(ax_rho_tot, "rho_tot", "Density Total", r"$k^2 P(k)$", 2)
plt.tight_layout()
fig_rho_tot.savefig("rho_tot.png")

fig_rho_comp, axs_rho_comp = plt.subplots(1, 2, figsize=(12, 5))
plot_quantity(axs_rho_comp[0], "rho_par", r"Density $\parallel$ to B", r"$k^{0.5} P(k)$", 0.5)
plot_quantity(axs_rho_comp[1], "rho_perp", r"Density $\perp$ to B", r"$k^{0.5} P(k)$", 0.5)
plt.tight_layout()
fig_rho_comp.savefig("rho_components.png")

plt.show()