import numpy as np
from astropy.io import fits
import scipy.ndimage as ndimage

# USER PARAMETERS: SIMULATION & DISTANCE
box_size_pc = 40.0         # Physical size of the box in pc
n_pixels = 128             # Resolution
distance_pc = 200.0        # Distance to the cloud

# OBSERVATIONAL PARAMETERS (EBHIS 21cm Survey)
ebhis_beam_fwhm_arcmin = 10.8  
ebhis_vel_res_kms = 1.44       
ebhis_rms_noise_K = 0.09       

pixel_size_pc = box_size_pc / n_pixels
pixel_scale_rad = pixel_size_pc / distance_pc
pixel_scale_arcmin = pixel_scale_rad * (180.0 * 60.0 / np.pi)

print(f"...Simulation parameters: Box={box_size_pc}pc, Res={n_pixels}^3, Dist={distance_pc}pc")
print(f"...Pixel physical size: {pixel_size_pc:.4f} pc")
print(f"...Pixel angular scale: {pixel_scale_arcmin:.4f} arcmin/pixel")

file = "default_setup/cubes_default_setup/"
input_filename = file + "cube_ppv_Tb_LOS_Z.fits"
output_filename = f"cube_ppv_Tb_LOS_Z_EBHIS_{int(distance_pc)}pc.fits"

print(f"\n...Opening theoretical PPV cube: '{input_filename}'")
try:
    hdul = fits.open(input_filename)
except FileNotFoundError:
    print(f"Error: {input_filename} cannot be found.")
    exit()

cube_Tb = hdul[0].data
header = hdul[0].header

# Read original velocity resolution from header (CDELT3 is in cm/s)
dv_cms = header.get('CDELT3', 58647.26)
dv_kms = np.abs(dv_cms) / 1e5

print(f"...Original velocity resolution: {dv_kms:.3f} km/s")
print(f"...Target EBHIS resolution: {ebhis_vel_res_kms} km/s")

# 1. SPECTRAL SMOOTHING
print("\n...Applying spectral smoothing.")
if ebhis_vel_res_kms > dv_kms:
    fwhm_conv_kms = np.sqrt(ebhis_vel_res_kms**2 - dv_kms**2)
    sigma_v_pixels = (fwhm_conv_kms / dv_kms) / 2.3548
    
    # Apply 1D Gaussian filter along the velocity axis (axis 0)
    cube_Tb = ndimage.gaussian_filter1d(cube_Tb, sigma=sigma_v_pixels, axis=0, mode='nearest')
else:
    print("...Skipping spectral smoothing: original resolution is lower than target.")

# 2. SPATIAL SMOOTHING (INSTRUMENTAL BEAM)
print("\n...Applying spatial smoothing.")
# Calculate Gaussian sigma for the beam in pixels
sigma_beam_arcmin = ebhis_beam_fwhm_arcmin / 2.3548
sigma_beam_pixels = sigma_beam_arcmin / pixel_scale_arcmin

print(f"...Beam size in pixels: {sigma_beam_pixels:.2f} pixels (sigma)")

for i in range(cube_Tb.shape[0]):
    cube_Tb[i, :, :] = ndimage.gaussian_filter(
        cube_Tb[i, :, :], 
        sigma=sigma_beam_pixels, 
        mode='wrap'
    )

# 3. ADDING INSTRUMENTAL NOISE
print(f"\n...Injecting Gaussian white noise (RMS = {ebhis_rms_noise_K * 1000} mK).")
noise_cube = np.random.normal(loc=0.0, scale=ebhis_rms_noise_K, size=cube_Tb.shape)
cube_Tb_noisy = cube_Tb + noise_cube

# SAVE MOCK OBSERVATION
print(f"\n...Saving mock EBHIS observation to: '{output_filename}'")
hdu_out = fits.PrimaryHDU(cube_Tb_noisy, header=header)
hdr = hdu_out.header

hdr['BUNIT'] = 'K'
hdr['BMAJ'] = ebhis_beam_fwhm_arcmin / 60.0  
hdr['BMIN'] = ebhis_beam_fwhm_arcmin / 60.0  

hdr['CDELT1'] = -pixel_scale_arcmin / 60.0 
hdr['CDELT2'] = pixel_scale_arcmin / 60.0
hdr['CUNIT1'] = 'deg'
hdr['CUNIT2'] = 'deg'

hdr.add_history(f'EBHIS mock: Dist={distance_pc}pc. Spatial and spectral smoothing applied.')
hdr.add_history(f'Noise added: RMS = {ebhis_rms_noise_K} K.')

hdu_out.writeto(output_filename, overwrite=True)
hdul.close()

print("...Mock observation complete.")