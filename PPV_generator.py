import numpy as np
from astropy.io import fits
from astropy import constants as const
import matplotlib.pyplot as plt

# --- Choice of the LOS ---
los_axis = 'Y' 

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
sigma_turb = 10.0e5  # 10 km/s

print("...Opening FITS file.")
fits_image_filename = "cube_info_00051.fits"
try:
    hdul = fits.open(fits_image_filename)
except FileNotFoundError:
    print(f"Error : {fits_image_filename} can't be found.")
    exit()

# Recovering basic cube data
density = hdul['DENSITY'].data
pressure = hdul['PRESSURE'].data
nz, ny, nx = density.shape 

if los_axis == 'Z':
    v_los = hdul['VELOCITY_Z'].data
    ds_pc = hdul[0].header.get('CDELT3', 1.0)
    n_los = nz
    shape_2d = (ny, nx)
    idx_func = lambda i: (i, slice(None), slice(None)) # [i, :, :]
elif los_axis == 'Y':
    v_los = hdul['VELOCITY_Y'].data
    ds_pc = hdul[0].header.get('CDELT2', 1.0)
    n_los = ny
    shape_2d = (nz, nx)
    idx_func = lambda i: (slice(None), i, slice(None)) # [:, i, :]
elif los_axis == 'X':
    v_los = hdul['VELOCITY_X'].data
    ds_pc = hdul[0].header.get('CDELT1', 1.0)
    n_los = nx
    shape_2d = (nz, ny)
    idx_func = lambda i: (slice(None), slice(None), i) # [:, :, i]
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
v_grid = np.linspace(-v_half_width, v_half_width, Nv)
nu_grid = nu_ul * (1.0 - v_grid / c)
nu_grid_3d = nu_grid[:, np.newaxis, np.newaxis]

# Le cube I_nu_cube s'adapte aux dimensions du plan d'observation
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

print("\n...Saving PPV cube to FITS.")
hdu_out = fits.PrimaryHDU(I_nu_cube)
hdr = hdu_out.header

# Adaptation des headers FITS selon l'axe
hdr['BUNIT'] = 'erg/s/cm2/Hz/sr'
hdr['CTYPE1'] = 'X' if los_axis in ['Y', 'Z'] else 'Y'
hdr['CTYPE2'] = 'Y' if los_axis == 'Z' else 'Z'
hdr['CTYPE3'] = 'VRAD'            
hdr['CRVAL3'] = 0.0               
hdr['CDELT3'] = v_grid[1] - v_grid[0] 
hdr['CRPIX3'] = Nv // 2          
hdr['CUNIT3'] = 'cm/s'           

filename_out = f'cube_ppv_emission_LOS_{los_axis}.fits'
hdu_out.writeto(filename_out, overwrite=True)
print(f"Saved as '{filename_out}'.")

print("\n...Plotting Moment 0 map.")
moment_0 = np.trapezoid(I_nu_cube, x=v_grid, axis=0) 

plt.figure(figsize=(8, 6))
plt.imshow(moment_0, origin='lower', cmap='inferno')
plt.colorbar(label="Integrated Intensity (erg / s / cm$^2$ / sr)")
plt.title(f"Integrated Emission Map (Moment 0) - LOS: {los_axis}")

# Adaptation des labels des axes du plot
if los_axis == 'Z':
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
elif los_axis == 'Y':
    plt.xlabel("X (pixels)")
    plt.ylabel("Z (pixels)")
elif los_axis == 'X':
    plt.xlabel("Y (pixels)")
    plt.ylabel("Z (pixels)")

plt.tight_layout()
plt.savefig(f"Moment0_map_LOS_{los_axis}.png")
plt.show()