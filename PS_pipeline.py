import os
import numpy as np
import matplotlib.pyplot as plt
import pspec

iouts = np.arange(40, 50)
base_path = "/Xnfs/Houches2026/group7/"

sim_dirs = [f.path for f in os.scandir(base_path) if f.is_dir()]
sim_dirs.sort()

all_spectra = {}

for sim_path in sim_dirs:
    sim_name = os.path.basename(sim_path)
    print(f"\n==============================================")
    print(f" Processing : {sim_name}")
    print(f"================================================")
    
    try:
        print("... Computing power spectra")
        ps = pspec.pspec(path=sim_path, iouts=iouts, magnetic=True, bidimensional=False, outfile="pspec_" + sim_name + "_%(iout)d.h5")
    except Exception as e:
        print(f"File ignored : {e}")
        continue
        
    print("... Stacking spectra")
    stacks = {
        "v_tot": 0.0, "vs": 0.0, "vc": 0.0,
        "kr_tot": 0.0, "krs": 0.0, "krc": 0.0,
        "ekin_tot": 0.0, "ekins": 0.0, "ekinc": 0.0,
        "rho_tot": 0.0, "rho_par": 0.0, "rho_perp": 0.0
    }
    
    for iout in iouts:
        # Velocity
        stacks["v_tot"] += ps[iout]["3d"]["velocity"]["pspec"]
        stacks["vs"]    += ps[iout]["3d"]["velocity_s"]["pspec"]
        stacks["vc"]    += ps[iout]["3d"]["velocity_c"]["pspec"]
        
        # Kritsuk
        stacks["kr_tot"] += ps[iout]["3d"]["kr"]["pspec"]
        stacks["krs"]    += ps[iout]["3d"]["krs"]["pspec"]
        stacks["krc"]    += ps[iout]["3d"]["krc"]["pspec"]
        
        # Kinetic energy
        stacks["ekin_tot"] += ps[iout]["3d"]["ekin"]["pspec"]
        stacks["ekins"]    += ps[iout]["3d"]["ekins"]["pspec"]
        stacks["ekinc"]    += ps[iout]["3d"]["ekinc"]["pspec"]
        
        # Density
        stacks["rho_tot"]  += ps[iout]["3d"]["density"]["pspec"]
        stacks["rho_par"]  += ps[iout]["3d"]["density"]["pspec_Bpar"]
        stacks["rho_perp"] += ps[iout]["3d"]["density"]["pspec_Bperp"]
        
    kbins = ps[iouts[0]]["3d"]["velocity_c"]["kbins"]
    kbins = 0.5 * (kbins[:-1] + kbins[1:])
    
    # Temporal mean
    for key in stacks:
        stacks[key] /= len(iouts)
        
    stacks["kbins"] = kbins
    all_spectra[sim_name] = stacks


print("\n... Plotting")

def plot_quantity(ax, quantity_key, title, ylabel, k_power):
    for sim_name, data in all_spectra.items():
        k = data["kbins"]
        pspec_val = data[quantity_key]
        ax.loglog(k, (k**k_power) * pspec_val, label=sim_name, linewidth=1.5)
    
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize='small')
    ax.savefig(quantity_key + ".png")

# --- Figure 1 : Velocity ---
fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle("Velocity power spectra", fontsize=14, fontweight='bold')
plot_quantity(axs1[0], "v_tot", "Total", r"$k^4 \: $P(k)", 4)
plot_quantity(axs1[1], "vs", "Solenoidal", r"$k^4 \: $P(k)", 4)
plot_quantity(axs1[2], "vc", "Compressive", r"$k^4 \: $P(k)", 4)
plt.tight_layout()

# --- Figure 2 : Kritsuk ---
fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("Kritsuk", fontsize=14, fontweight='bold')
plot_quantity(axs2[0], "kr_tot", "Total", r"$k^{3.5} \: $P(k)", 3.5)
plot_quantity(axs2[1], "krs", "Solenoidal", r"$k^{3.5} \: $P(k)", 3.5)
plot_quantity(axs2[2], "krc", "Compressive", r"$k^{3.5} \: $P(k)", 3.5)
plt.tight_layout()

# --- Figure 3 : Kinetic energy ---
fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle("Kinetic energy power spectra", fontsize=14, fontweight='bold')
plot_quantity(axs3[0], "ekin_tot", "Total", r"$k^{3.5} \: $P(k)", 3.5)
plot_quantity(axs3[1], "ekins", "Solenoidal", r"$k^{3.5} \: $P(k)", 3.5)
plot_quantity(axs3[2], "ekinc", "Compressive", r"$k^{3.5} \: $P(k)", 3.5)
plt.tight_layout()

# --- Figure 4 : Density and Anisotropy B ---
fig4, axs4 = plt.subplots(1, 3, figsize=(18, 5))
fig4.suptitle("Density power spectra", fontsize=14, fontweight='bold')
plot_quantity(axs4[0], "rho_tot", "Total density", r"$k^2 \: $P(k)", 2)
plot_quantity(axs4[1], "rho_par", r"density $\parallel$ to B", r"$k^{0.5} \: $P(k)", 0.5)
plot_quantity(axs4[2], "rho_perp", r"Density $\perp$ to B", r"$k^{0.5} \: $P(k)", 0.5)
plt.tight_layout()

plt.show()