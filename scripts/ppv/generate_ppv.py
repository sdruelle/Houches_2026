"""Génère un cube PPV HI et ses trois cartes de moments."""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from ppv_core import make_ppv, moments

INPUT_CUBE = "cube_info_00051.fits"
LOS_AXIS = "Z"


def main():
    result = make_ppv(INPUT_CUBE, axis=LOS_AXIS)
    brightness_temperature = result["brightness_temperature"]
    velocity = result["v_grid"]
    moment_0, moment_1, moment_2 = moments(brightness_temperature, velocity)

    header = fits.Header()
    header["BUNIT"] = "K"
    header["CTYPE1"] = "X" if LOS_AXIS in {"Y", "Z"} else "Y"
    header["CTYPE2"] = "Y" if LOS_AXIS == "Z" else "Z"
    header["CTYPE3"] = "VRAD"
    header["CRVAL3"] = 0.0
    header["CDELT3"] = velocity[1] - velocity[0]
    header["CRPIX3"] = len(velocity) // 2
    header["CUNIT3"] = "cm/s"
    output_cube = f"cube_ppv_Tb_LOS_{LOS_AXIS}.fits"
    fits.writeto(output_cube, brightness_temperature, header, overwrite=True)
    print(f"Saved '{output_cube}'.")

    labels = [(moment_0, "inferno", "Integrated $T_B$ (K km s$^{-1}$)"),
              (moment_1, "jet", "Velocity (km s$^{-1}$)"),
              (moment_2, "viridis", "Velocity dispersion (km s$^{-1}$)")]
    figure, axes = plt.subplots(1, 3, figsize=(20, 6))
    for index, (data, cmap, colorbar_label) in enumerate(labels):
        image = axes[index].imshow(data, origin="lower", cmap=cmap)
        figure.colorbar(image, ax=axes[index], label=colorbar_label)
        axes[index].set_title(f"Moment {index} — LOS {LOS_AXIS}")
    figure.tight_layout()
    figure.savefig(f"Moments_map_LOS_{LOS_AXIS}.png")


if __name__ == "__main__":
    main()
