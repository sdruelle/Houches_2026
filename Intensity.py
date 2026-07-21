import numpy as np
from astropy.io import fits
from astropy import constants as const
import matplotlib.pyplot as plt

# Defining constants
k_B = const.k_B.cgs.value
m_H = const.m_p.cgs.value
c = const.c.cgs.value
h = const.h.cgs.value
mu = 1.4

# --- Generic transition parameters (Updated for HI 21cm line) ---
nu_ul = 1.420405751e9   
A_ul = 2.8843e-15       
g_u = 3.0               
g_l = 1.0               

# Particle mass for thermal broadening and fraction in lower state
m_part = 1.0 * m_H 
f_l = 0.25   

# Turbulent velocity dispersion, in cm/s.
sigma_turb = 10.0e5  # 10 km/s

print("...Opening fits file.")
fits_image_filename = "cube_info_00051.fits"
try:
    hdul = fits.open(fits_image_filename)
except FileNotFoundError:
    print(f"Error : {fits_image_filename} can't be found.")
    exit()

# Recovering the cube data
# Density
try:
    density = hdul['DENSITY'].data
    print("Density recovered successfully.")
except KeyError:
    print("Error: 'DENSITY' doesn't exist.")
# Velocity
try:
    vx = hdul['VELOCITY_X'].data
    vy = hdul['VELOCITY_Y'].data
    vz = hdul['VELOCITY_Z'].data
    print("Velocity recovered successfully.")
except KeyError:
    print("Warning : Velocity wasn't recovered.")
# Pressure
try:
    pressure = hdul['PRESSURE'].data
    print("Pressure recovered successfully.")
except KeyError:
    print("Warning : Pressure wasn't recovered.")
# Length
try:
    ds_pc = hdul[0].header['CDELT3']
except KeyError:
    ds_pc = 1.0
ds_cm = ds_pc * const.pc.cgs.value
# Temperature
try:
    temperature = (pressure * mu * m_H) / (density * k_B)
    print("Temperature computed successfully.")
    print(f"Mean temperature : {np.mean(temperature):.2e} K")
    print(f"Min temperature : {np.min(temperature):.2e} K")
    print(f"Max temperature : {np.max(temperature):.2e} K")
except NameError:
    print("Computing the temperature failed.")

print("\n...Computing the intensity.")

n_tot = density / (mu * m_H)
n_l_cube = n_tot * f_l  

# Choosing the LOS
x_c, y_c = 64, 64
T_los = temperature[:, y_c, x_c]
vz_los = vz[:, y_c, x_c]
nl_los = n_l_cube[:, y_c, x_c]

# Subgrid frequency/velocity grid
sigma_v_thermal_max = np.sqrt(k_B * np.max(T_los) / m_part)  # cm/s
v_half_width = np.max(np.abs(vz_los)) + 5.0 * (sigma_v_thermal_max + sigma_turb)
v_grid = np.linspace(-v_half_width, v_half_width, 200)
nu_grid = nu_ul * (1.0 - v_grid / c)

I_nu = 0

# Integration
for i in range(len(T_los)):
    T_i = T_los[i]
    vz_i = vz_los[i]
    nl_i = nl_los[i]

    # Computing Kappa_nu and Tau_nu
    sigma_v = np.sqrt(k_B * T_i / m_part + sigma_turb**2)
    sigma_nu = (nu_ul / c) * sigma_v
    nu_c = nu_ul * (1.0 - vz_i / c)
    phi_nu = (1.0 / (sigma_nu * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((nu_grid - nu_c) / sigma_nu)**2)

    term1 = c**2 / (8.0 * np.pi * nu_ul**2)
    term2 = A_ul * nl_i * (g_u / g_l)
    term3 = 1.0 - np.exp(-(h * nu_ul) / (k_B * T_i))
    kappa_nu = term1 * term2 * term3 * phi_nu
    dtau_nu = kappa_nu * ds_cm

    # Computing S_nu
    S_nu = (2.0 * h * nu_ul**3 / c**2) / (np.exp((h * nu_ul) / (k_B * T_i)) - 1.0)

    # Computing I_nu
    I_nu = I_nu * np.exp(-dtau_nu) + S_nu * (1.0 - np.exp(-dtau_nu))

hdul.close()

print("\n...Plotting.")
v_grid_km_s = v_grid / 1e5
plt.figure(figsize=(10, 6))
plt.plot(v_grid_km_s, I_nu, color='blue', linewidth=2)
plt.xlabel("Velocity (km/s)", fontsize=12)
plt.ylabel("Intensity $I_\\nu$ (erg / s / cm$^2$ / Hz / sr)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Intensity.png")
plt.show()

