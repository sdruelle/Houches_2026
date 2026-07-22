import numpy as np
from astropy.io import fits
from astropy import constants as const
import matplotlib
matplotlib.use('Agg') # Mode non-interactif pour éviter les crashs sur serveur
import matplotlib.pyplot as plt

# --- Choice of the LOS and parameters ---
los_axis = 'Z' 
fits_image_filename = "cube_info_00051.fits"
lags = [1, 4, 8, 16, 32]

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

m_part = 1.0 * m_H 
f_l = 0.25             
sigma_turb = 0.0  

# Recovering cube data
print(f"...Opening FITS file: {fits_image_filename}")
try:
    hdul = fits.open(fits_image_filename)
except FileNotFoundError as exc:
    raise FileNotFoundError(f"Error : {fits_image_filename} can't be found.") from exc

density = hdul['DENSITY'].data
pressure = hdul['PRESSURE'].data
nz, ny, nx = density.shape 

if los_axis == 'Z':
    v_los = hdul['VELOCITY_Z'].data
    ds_pc = hdul[0].header.get('CDELT3', 1.0)
    n_los = nz
    shape_2d = (ny, nx)
    idx_func = lambda i: (i, slice(None), slice(None)) 
elif los_axis == 'Y':
    v_los = hdul['VELOCITY_Y'].data
    ds_pc = hdul[0].header.get('CDELT2', 1.0)
    n_los = ny
    shape_2d = (nz, nx)
    idx_func = lambda i: (slice(None), i, slice(None)) 
elif los_axis == 'X':
    v_los = hdul['VELOCITY_X'].data
    ds_pc = hdul[0].header.get('CDELT1', 1.0)
    n_los = nx
    shape_2d = (nz, ny)
    idx_func = lambda i: (slice(None), slice(None), i) 
else:
    raise ValueError("los_axis must be 'X', 'Y', or 'Z'")

ds_cm = ds_pc * const.pc.cgs.value

temperature = (pressure * mu * m_H) / (density * k_B)
n_tot = density / (mu * m_H)
n_l_cube = n_tot * f_l  

# Velocity/Frequency grid creation
print(f"...Initializing PPV grid for LOS = {los_axis}-axis.")
sigma_v_thermal_max = np.sqrt(k_B * np.max(temperature) / m_part)
v_half_width = np.max(np.abs(v_los)) + 5.0 * (sigma_v_thermal_max + sigma_turb)

Nv = 200
v_grid = np.linspace(-v_half_width, v_half_width, Nv) # in cm/s
nu_grid = nu_ul * (1.0 - v_grid / c)
nu_grid_3d = nu_grid[:, np.newaxis, np.newaxis]

I_nu_cube = np.zeros((Nv, shape_2d[0], shape_2d[1]))

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
    
    I_nu_cube = I_nu_cube * np.exp(-dtau_nu) + S_nu * (1.0 - np.exp(-dtau_nu))

hdul.close()


print("...Computing Moments.")
conversion_factor = c**2 / (2.0 * k_B * nu_ul**2)
T_B_cube = I_nu_cube * conversion_factor
v_grid_kms = v_grid / 1.0e5 

moment_0 = np.trapezoid(T_B_cube, x=v_grid_kms, axis=0)
moment_1_numerator = np.trapezoid(T_B_cube * v_grid_kms[:, np.newaxis, np.newaxis], x=v_grid_kms, axis=0)

threshold = 1e-3 * np.max(moment_0)
mask = (np.isfinite(moment_0) & (moment_0 > threshold))

moment_1 = np.full_like(moment_0, np.nan)
moment_1[mask] = moment_1_numerator[mask] / moment_0[mask]

print("...Computing and Plotting Velocity Increments PDFs with Gaussian Fits.")

lags = [1, 4, 8, 16, 32] 

lags_to_fit = [1, 32]

fig_inc, ax_inc = plt.subplots(figsize=(8, 6))

for l in lags:
    delta_v_x = moment_1[:, l:] - moment_1[:, :-l]
    delta_v_y = moment_1[l:, :] - moment_1[:-l, :]
    
    delta_v_combined = np.concatenate([delta_v_x.flatten(), delta_v_y.flatten()])
    valid_delta_v = delta_v_combined[np.isfinite(delta_v_combined)]
    
    if valid_delta_v.size > 0:
        hist, bin_edges = np.histogram(valid_delta_v, bins=150, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        p = ax_inc.plot(
            bin_centers, 
            hist, 
            drawstyle="steps-mid", 
            linewidth=2.0, 
            label=rf"$l = {l}$ px"
        )
        
        if l in lags_to_fit:
            mu = np.mean(valid_delta_v)
            sigma = np.std(valid_delta_v)
            x_gauss = np.linspace(bin_centers.min(), bin_centers.max(), 500)
            y_gauss = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x_gauss - mu) / sigma)**2)
            
            ax_inc.plot(
                x_gauss, 
                y_gauss, 
                linestyle=":", 
                color=p[0].get_color(), 
                linewidth=1.8, 
                label=rf"Gauss. fit ($l={l}$)"
            )

ax_inc.set_yscale('log')
ax_inc.set_ylim(bottom=1e-4)

ax_inc.set_xlabel(r"Velocity Increment $\Delta v(l)$ (km s$^{-1}$)")
ax_inc.set_ylabel("Probability Density")
ax_inc.set_title(f"PDF of Velocity Increments & Gaussian Fits — LOS: {los_axis}")
ax_inc.legend()
ax_inc.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig(f"PDF_Velocity_Increments_Fits_LOS_{los_axis}.png", dpi=200, bbox_inches="tight")
print(f"...Saved 'PDF_Velocity_Increments_Fits_LOS_{los_axis}.png'.")

print("\n...Saving 2D Moment maps to FITS files for Power Spectrum analysis.")

hdu_m0 = fits.PrimaryHDU(moment_0)
hdu_m0.header['BUNIT'] = 'K km/s'
file_m0 = f"moment_0_map_LOS_{los_axis}.fits"
hdu_m0.writeto(file_m0, overwrite=True)
print(f"Saved '{file_m0}'")

moment_1_clean = np.nan_to_num(moment_1, nan=0.0)

hdu_m1 = fits.PrimaryHDU(moment_1_clean)
hdu_m1.header['BUNIT'] = 'km/s'
file_m1 = f"moment_1_map_LOS_{los_axis}.fits"
hdu_m1.writeto(file_m1, overwrite=True)
print(f"Saved '{file_m1}'")