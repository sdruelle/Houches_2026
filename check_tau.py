import numpy as np
from astropy.io import fits
from astropy import constants as const
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

# --- Choice of the LOS ---
los_axis = 'Z' 

# --- Constants and parameters ---
k_B = const.k_B.cgs.value
m_H = const.m_p.cgs.value
c = const.c.cgs.value
h = const.h.cgs.value
mu = 1.4

# --- Generic transition parameters ---
nu_ul = 1.420405751e9   
A_ul = 2.8843e-15       
g_u = 3.0               
g_l = 1.0               

# Particle mass for thermal broadening and fraction in lower state
m_part = 1.0 * m_H 
f_l = 0.25             

# Turbulent velocity dispersion, in cm/s.
sigma_turb = 0  # 10 km/s

print("...Opening FITS file.")
fits_image_filename = "cube_info_00051.fits"
try:
    hdul = fits.open(fits_image_filename)
except FileNotFoundError:
    print(f"Error : {fits_image_filename} can't be found.")
    exit()

# Recovering cube data
density = hdul['DENSITY'].data
pressure = hdul['PRESSURE'].data
nz, ny, nx = density.shape 

if los_axis == 'Z':
    v_los = hdul['VELOCITY_Z'].data
    ds_pc = hdul[0].header.get('CDELT3', 1.0)
    n_los = nz
    shape_2d = (ny, nx)
    idx_func = lambda i: (i, slice(None), slice(None)) 
    axis_idx = 0
elif los_axis == 'Y':
    v_los = hdul['VELOCITY_Y'].data
    ds_pc = hdul[0].header.get('CDELT2', 1.0)
    n_los = ny
    shape_2d = (nz, nx)
    idx_func = lambda i: (slice(None), i, slice(None)) 
    axis_idx = 1
elif los_axis == 'X':
    v_los = hdul['VELOCITY_X'].data
    ds_pc = hdul[0].header.get('CDELT1', 1.0)
    n_los = nx
    shape_2d = (nz, ny)
    idx_func = lambda i: (slice(None), slice(None), i) 
    axis_idx = 2
else:
    raise ValueError("los_axis must be 'X', 'Y', or 'Z'")

ds_cm = ds_pc * const.pc.cgs.value

temperature = (pressure * mu * m_H) / (density * k_B)
n_tot = density / (mu * m_H)
n_l_cube = n_tot * f_l  

print(f"...Initializing PPV grid for LOS = {los_axis}-axis.")

# Velocity/Frequency grid creation
sigma_v_thermal_max = np.sqrt(k_B * np.max(temperature) / m_part)
v_half_width = np.max(np.abs(v_los)) + 5.0 * (sigma_v_thermal_max + sigma_turb)

Nv = 200
v_grid = np.linspace(-v_half_width, v_half_width, Nv) # in cm/s
nu_grid = nu_ul * (1.0 - v_grid / c)
nu_grid_3d = nu_grid[:, np.newaxis, np.newaxis]

I_nu_cube = np.zeros((Nv, shape_2d[0], shape_2d[1]))
tau_cube = np.zeros((Nv, shape_2d[0], shape_2d[1])) # Initialize optical depth cube

print("...Computing Radiative Transfer (PPV Cube).")

# Integration along the LOS
for i in range(n_los):
    T_i = temperature[idx_func(i)][np.newaxis, :, :]
    v_i = v_los[idx_func(i)][np.newaxis, :, :]
    nl_i = n_l_cube[idx_func(i)][np.newaxis, :, :]

    T_i = np.maximum(T_i, 1e-10)

    sigma_v = np.sqrt(k_B * T_i / m_part + sigma_turb**2)
    sigma_nu = (nu_ul / c) * sigma_v
    nu_c = nu_ul * (1.0 - v_i / c)
    
    phi_nu = (1.0 / (sigma_nu * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((nu_grid_3d - nu_c) / sigma_nu)**2)

    term1 = c**2 / (8.0 * np.pi * nu_ul**2)
    term2 = A_ul * nl_i * (g_u / g_l)
    term3 = 1.0 - np.exp(-(h * nu_ul) / (k_B * T_i))
    
    kappa_nu = term1 * term2 * term3 * phi_nu
    dtau_nu = kappa_nu * ds_cm

    S_nu = (2.0 * h * nu_ul**3 / c**2) / (np.exp((h * nu_ul) / (k_B * T_i)) - 1.0)

    # Accumulate optical depth and intensity
    tau_cube += dtau_nu
    I_nu_cube = I_nu_cube * np.exp(-dtau_nu) + S_nu * (1.0 - np.exp(-dtau_nu))

hdul.close()

print("...Converting Specific Intensity to Brightness Temperature.")
conversion_factor = c**2 / (2.0 * k_B * nu_ul**2)
T_B_cube = I_nu_cube * conversion_factor

print("...Computing Moments and Column Densities.")
v_grid_kms = v_grid / 1.0e5 

# Moment 0: Integrated Brightness Temperature
moment_0 = np.trapezoid(T_B_cube, x=v_grid_kms, axis=0)

# True Column Density (integrated along LOS directly from physical cube)
N_H_true = np.sum(n_tot, axis=axis_idx) * ds_cm

# Inferred Column Density (using the optically thin relation)
N_H_inferred = 1.823e18 * moment_0

# Maximum optical depth along velocity axis for each spatial pixel
tau_max = np.max(tau_cube, axis=0)

# Total integrated optical depth along the velocity axis
tau_total = np.trapezoid(tau_cube, x=v_grid_kms, axis=0)

# Optically thin mask (tau < 1)
thin_mask = tau_max < 1.0

# Calculate the relation ratio (Inferred / True). Should be ~1 where thin_mask is True.
relation_ratio = np.full_like(N_H_true, np.nan)
# We only calculate the ratio where True column density is greater than zero to avoid division by zero
valid_pixels = thin_mask & (N_H_true > 0)
relation_ratio[valid_pixels] = N_H_inferred[valid_pixels] / N_H_true[valid_pixels]

print("...Plotting Validation Maps.")

# Axis label logic
if los_axis == 'Z':
    xlabel, ylabel = "X (pixels)", "Y (pixels)"
elif los_axis == 'Y':
    xlabel, ylabel = "X (pixels)", "Z (pixels)"
elif los_axis == 'X':
    xlabel, ylabel = "Y (pixels)", "Z (pixels)"

# Create a 2x3 grid for validation checks (figsize widened to fit 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. True Column Density
im0 = axes[0, 0].imshow(np.log10(N_H_true + 1e-10), origin='lower', cmap='viridis')
fig.colorbar(im0, ax=axes[0, 0], label=r"$\log_{10}(N_{H, true})$ (cm$^{-2}$)")
axes[0, 0].set_title(f"True Column Density - LOS: {los_axis}")
axes[0, 0].set_xlabel(xlabel)
axes[0, 0].set_ylabel(ylabel)

# 2. Maximum Optical Depth
im1 = axes[0, 1].imshow(tau_max, origin='lower', cmap='magma', vmax=2.0) 
fig.colorbar(im1, ax=axes[0, 1], label=r"Max $\tau_\nu$")
axes[0, 1].set_title(rf"Max Optical Depth ($\tau$) - LOS: {los_axis}")
axes[0, 1].set_xlabel(xlabel)
axes[0, 1].set_ylabel(ylabel)
axes[0, 1].contour(tau_max, levels=[1.0], colors='white', linestyles='dashed')

# 3. Total Integrated Optical Depth
im2 = axes[0, 2].imshow(tau_total, origin='lower', cmap='magma') 
fig.colorbar(im2, ax=axes[0, 2], label=r"$\int \tau_\nu dv$ (km s$^{-1}$)")
axes[0, 2].set_title(rf"Total Integrated Optical Depth - LOS: {los_axis}")
axes[0, 2].set_xlabel(xlabel)
axes[0, 2].set_ylabel(ylabel)

# 4. Inferred Column Density
im3 = axes[1, 0].imshow(np.log10(N_H_inferred + 1e-10), origin='lower', cmap='viridis')
fig.colorbar(im3, ax=axes[1, 0], label=r"$\log_{10}(N_{H, inferred})$ (cm$^{-2}$)")
axes[1, 0].set_title(f"Inferred Column Density (Optically Thin Approx)")
axes[1, 0].set_xlabel(xlabel)
axes[1, 0].set_ylabel(ylabel)

# 5. Relation Check Ratio (Inferred / True)
im4 = axes[1, 1].imshow(relation_ratio, origin='lower', cmap='RdBu_r', vmin=0.5, vmax=1.5)
fig.colorbar(im4, ax=axes[1, 1], label=r"$N_{H, inferred} / N_{H, true}$")
axes[1, 1].set_title(r"Relation Validity (Inferred / True) for $\tau < 1$")
axes[1, 1].set_xlabel(xlabel)
axes[1, 1].set_ylabel(ylabel)

# 6. Hide the empty 6th subplot 
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(f"Optical_Thickness_Validation_LOS_{los_axis}.png")