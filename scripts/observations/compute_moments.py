import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os

# PARAMETERS
input_cube = "default_setup/cubes_default_setup/cube_ppv_Tb_LOS_Y_EBHIS_200pc.fits" 

# Instruments parameters
rms_noise_K = 0.09  
sigma_threshold = 3.0

# RECOVERING CUBE
print(f"...Opening file: {input_cube}")
hdul = fits.open(input_cube)
cube = hdul[0].data
hdr = hdul[0].header

# Recovering the velocity axis
crval3 = hdr['CRVAL3']    
cdelt3 = hdr['CDELT3']    
crpix3 = hdr['CRPIX3']    
n_v = cube.shape[0]       

v_axis_cms = crval3 + (np.arange(n_v) - (crpix3 - 1)) * cdelt3
v_axis_kms = v_axis_cms / 1e5
dv_kms = np.abs(cdelt3) / 1e5

print(f"...Velocity axis: from {v_axis_kms[0]:.2f} to {v_axis_kms[-1]:.2f} km/s")

# Moments computation
threshold_K = sigma_threshold * rms_noise_K
print(f"...Applying a {sigma_threshold} sigma mask, ({threshold_K:.3f} K)")

cube_masked = np.where(cube > threshold_K, cube, 0.0)

# Moment 0
print("...Computing Moment 0")
moment_0 = np.sum(cube_masked, axis=0) * dv_kms

# Moment 1
print("...Computing Moment 1")
v_3d = v_axis_kms[:, np.newaxis, np.newaxis]

moment_1 = np.zeros_like(moment_0)
valid_pixels = moment_0 > 0
moment_1[valid_pixels] = np.sum(cube_masked * v_3d, axis=0)[valid_pixels] / np.sum(cube_masked, axis=0)[valid_pixels]
moment_1[~valid_pixels] = np.nan

# Moment 2
print("...Computing Moment 2")
moment_2 = np.zeros_like(moment_0)
v_diff_sq = (v_3d - moment_1)**2
moment_2[valid_pixels] = np.sqrt(np.sum(cube_masked * v_diff_sq, axis=0)[valid_pixels] / np.sum(cube_masked, axis=0)[valid_pixels])
moment_2[~valid_pixels] = np.nan

# Saving fits
def create_2d_header(header_3d):
    hdr_2d = header_3d.copy()
    keys_to_remove = ['NAXIS3', 'CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3']
    for key in keys_to_remove:
        if key in hdr_2d:
            del hdr_2d[key]
    hdr_2d['NAXIS'] = 2
    return hdr_2d

hdr_2d = create_2d_header(hdr)

out_m0 = input_cube.replace('.fits', '_Mom0.fits')
hdr_m0 = hdr_2d.copy()
hdr_m0['BUNIT'] = 'K km/s'
hdr_m0.add_history(f'Moment 0 calculated with > {sigma_threshold} sigma threshold')
fits.writeto(out_m0, moment_0, hdr_m0, overwrite=True)

out_m1 = input_cube.replace('.fits', '_Mom1.fits')
hdr_m1 = hdr_2d.copy()
hdr_m1['BUNIT'] = 'km/s'
hdr_m1.add_history(f'Moment 1 calculated with > {sigma_threshold} sigma threshold')
fits.writeto(out_m1, moment_1, hdr_m1, overwrite=True)

out_m2 = input_cube.replace('.fits', '_Mom2.fits')
hdr_m2 = hdr_2d.copy()
hdr_m2['BUNIT'] = 'km/s'
hdr_m2.add_history(f'Moment 2 calculated with > {sigma_threshold} sigma threshold')
fits.writeto(out_m2, moment_2, hdr_m2, overwrite=True)

print(f"...Saved FITS files.")

# Generating Figure
print("...Generating PNG figure")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

origin = 'lower'

# --- Plot Moment 0 ---
im0 = axes[0].imshow(moment_0, origin=origin, cmap='inferno')
axes[0].set_title('Moment 0 - LOS: Y')
axes[0].set_xlabel('X (pixels)')
axes[0].set_ylabel('Z (pixels)')
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label('Integrated $T_B$ (K km s$^{-1}$)')

# --- Plot Moment 1 ---
im1 = axes[1].imshow(moment_1, origin=origin, cmap='jet')
axes[1].set_title('Moment 1 (Velocity Field) - LOS: Y')
axes[1].set_xlabel('X (pixels)')
axes[1].set_ylabel('Z (pixels)')
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label('Velocity (km s$^{-1}$)')

# --- Plot Moment 2 ---
im2 = axes[2].imshow(moment_2, origin=origin, cmap='viridis')
axes[2].set_title('Moment 2 (Dispersion) - LOS: Y')
axes[2].set_xlabel('X (pixels)')
axes[2].set_ylabel('Z (pixels)')
cbar2 = fig.colorbar(im2, ax=axes[2])
cbar2.set_label('Velocity Dispersion (km s$^{-1}$)')

plt.tight_layout()

out_png = input_cube.replace('.fits', '_Moments_map.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"...Saved figure : {out_png}")

hdul.close()
print("\nDone.")