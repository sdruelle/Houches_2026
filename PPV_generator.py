import numpy as np
from astropy.io import fits
from astropy import constants as const
import matplotlib.pyplot as plt

# --- Constants and parameters ---
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

print("...Opening FITS file.")
fits_image_filename = "cube_info_00051.fits"
try:
    hdul = fits.open(fits_image_filename)
except FileNotFoundError:
    print(f"Error : {fits_image_filename} can't be found.")
    exit()

# Recovering the cube data
density = hdul['DENSITY'].data
vz = hdul['VELOCITY_Z'].data  # Z-axis is our Line of Sight (LOS)
pressure = hdul['PRESSURE'].data
ds_pc = hdul[0].header.get('CDELT3', 1.0)
ds_cm = ds_pc * const.pc.cgs.value

temperature = (pressure * mu * m_H) / (density * k_B)
n_tot = density / (mu * m_H)
n_l_cube = n_tot * f_l  

nz, ny, nx = density.shape 

print("...Initializing PPV grid.")

# Velocity/Frequency grid creation
sigma_v_thermal_max = np.sqrt(k_B * np.max(temperature) / m_part)
v_half_width = np.max(np.abs(vz)) + 5.0 * (sigma_v_thermal_max + sigma_turb)

Nv = 200
v_grid = np.linspace(-v_half_width, v_half_width, Nv)
nu_grid = nu_ul * (1.0 - v_grid / c)

nu_grid_3d = nu_grid[:, np.newaxis, np.newaxis]

I_nu_cube = np.zeros((Nv, ny, nx))

print("...Computing Radiative Transfer (PPV Cube).")

# Integration along the Z axis
for i in range(nz):
    T_i = temperature[i, :, :][np.newaxis, :, :]
    vz_i = vz[i, :, :][np.newaxis, :, :]
    nl_i = n_l_cube[i, :, :][np.newaxis, :, :]

    T_i = np.maximum(T_i, 1e-10)

    sigma_v = np.sqrt(k_B * T_i / m_part + sigma_turb**2)
    sigma_nu = (nu_ul / c) * sigma_v
    nu_c = nu_ul * (1.0 - vz_i / c)
    
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

hdr['BUNIT'] = 'erg/s/cm2/Hz/sr'
hdr['CTYPE1'] = 'X'
hdr['CTYPE2'] = 'Y'
hdr['CTYPE3'] = 'VRAD'             
hdr['CRVAL3'] = 0.0               
hdr['CDELT3'] = v_grid[1] - v_grid[0] 
hdr['CRPIX3'] = Nv // 2          
hdr['CUNIT3'] = 'cm/s'          

hdu_out.writeto('cube_ppv_emission.fits', overwrite=True)
print("Saved as 'cube_ppv_emission.fits'.")

print("\n...Plotting Moment 0 map.")
moment_0 = np.trapezoid(I_nu_cube, x=v_grid, axis=0) 

plt.figure(figsize=(8, 6))
plt.imshow(moment_0, origin='lower', cmap='inferno')
plt.colorbar(label="Integrated Intensity (erg / s / cm$^2$ / sr)")
plt.title("Integrated Emission Map (Moment 0)")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.tight_layout()
plt.savefig("Moment0_map.png")
plt.show()