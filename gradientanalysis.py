import matplotlib.pyplot as plt
from pywavan import fan_trans
import numpy as np

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
sim_names = np.array(["half_B","default_setup","double_B"])

cos_theta = np.zeros(3,dtype=object)

for i,name in enumerate(sim_names):
    data = yt.load(f"{sim_dir}/{name}/output_{iout:05d}/info_{iout:05d}.txt")
    cos_theta[i] = yt2angle(data)


fig = plt.figure(figsize=(10,6))
_ = plt.hist(cos_theta[0].flatten(), bins=25, density=True, color="blue",  rwidth=0.9, alpha=0.7, lable=r"2.5 $\mu$G")
_ = plt.hist(cos_theta[1].flatten(), bins=25, density=True, color="red",   rwidth=0.9, alpha=0.7, lable=r"5 $\mu$G")
_ = plt.hist(cos_theta[2].flatten(), bins=25, density=True, color="green", rwidth=0.9, alpha=0.7, lable=r"10 $\mu$G")
plt.xlabel(r"cos $\theta$")
plt.ylabel("PDF")
plt.legend()
plt.show()
plt.close()