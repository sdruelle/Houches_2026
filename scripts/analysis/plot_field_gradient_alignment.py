import matplotlib.pyplot as plt
from pywavan import fan_trans
import numpy as np
import yt

def get_cubes(ds):
    levelmin = ds.parameters["levelmin"]
    res = 1 << levelmin
    grid = ds.r[::res*1j, ::res*1j, ::res*1j,]

    fits = grid.to_fits_data(fields=[("gas", "density"),("gas", "magnetic_field_x"),("gas", "magnetic_field_y"),("gas", "magnetic_field_z")])
    density_cube = fits["density"].data
    Bx_cube = fits["magnetic_field_x"].data
    By_cube = fits["magnetic_field_y"].data
    Bz_cube = fits["magnetic_field_z"].data

    B_cube = np.stack([Bx_cube, By_cube, Bz_cube], axis=-1)
    return density_cube, B_cube

def get_gradient(density_cube):
    dx,dy,dz = np.gradient(density_cube)
    grad_density_cube = np.stack([dx, dy, dz], axis=-1)
    return grad_density_cube

def get_angle(B_cube,grad_cube):
    dot = np.sum(grad_cube * B_cube, axis=-1)

    norm_dens = np.linalg.norm(grad_cube, axis=-1)
    norm_B = np.linalg.norm(B_cube, axis=-1)

    cos_theta = dot / (norm_dens * norm_B)

    return cos_theta

def yt2angle(ds):
    density_cube,B_cube = get_cubes(ds)
    grad_density_cube = get_gradient(density_cube)
    cos_theta = get_angle(B_cube, grad_density_cube)

    return cos_theta

sim_dir = "/Xnfs/Houches2026/group7"
sim_names = np.array(["default_setup","low_turbrms","high_turbrms"])

cos_theta = np.zeros(3,dtype=object)

output_num = 49

for i,name in enumerate(sim_names):
    data = yt.load(f"{sim_dir}/{name}/output_{output_num:05d}/info_{output_num:05d}.txt")
    cos_theta[i] = yt2angle(data)

fig, ax = plt.subplots(figsize=(8, 6))

colors = ["#1f77b4", "#d62728", "#2ca02c"]
labels = [r"$B_0$ = 2.5 $\mu$G (half_B)", 
          r"$B_0$ = 5.0 $\mu$G (default)", 
          r"$B_0$ = 10.0 $\mu$G (double_B)"]

bins = np.linspace(-1, 1, 50)

for i in range(3):
    flat_data = cos_theta[i].flatten()
    print(flat_data)
    clean_data = flat_data[~np.isnan(flat_data)]
    
    ax.hist(clean_data, bins=bins, density=True, color=colors[i], 
            histtype='step', linewidth=2.5, label=labels[i])
    
    ax.hist(clean_data, bins=bins, density=True, color=colors[i], 
            histtype='stepfilled', alpha=0.1)

ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

ax.set_xlabel(r"cos $\theta$ (Alignement magnetic field - density gradient)", fontsize=14)
ax.set_ylabel("PDF", fontsize=14)
ax.set_xlim(-1, 1)

ax.tick_params(axis='both', which='major', labelsize=12, direction='in', top=True, right=True)
ax.minorticks_on()
ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True)

ax.legend(fontsize=12, loc='upper left', frameon=True, edgecolor='black')

plt.tight_layout()
plt.savefig("pdf_alignement_turbulence.png", dpi=300, bbox_inches="tight")
plt.close()