"""Trace les PDF des incréments de vitesse d'un cube PPV."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from ppv_core import make_ppv, moments

INPUT_CUBE = "cube_info_00051.fits"
LOS_AXIS = "Z"
LAGS = [1, 4, 8, 16, 32]
LAGS_TO_FIT = {1, 32}


def main():
    result = make_ppv(INPUT_CUBE, axis=LOS_AXIS)
    brightness_temperature = result["brightness_temperature"]
    moment_0, moment_1, _ = moments(brightness_temperature, result["v_grid"])
    figure, axis = plt.subplots(figsize=(8, 6))
    for lag in LAGS:
        increments = np.concatenate((moment_1[:, lag:] - moment_1[:, :-lag], moment_1[lag:, :] - moment_1[:-lag, :])).ravel()
        increments = increments[np.isfinite(increments)]
        if increments.size:
            histogram, edges = np.histogram(increments, bins=150, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            line = axis.plot(centers, histogram, drawstyle="steps-mid", label=f"l = {lag} px")
            if lag in LAGS_TO_FIT:
                mean, standard_deviation = np.mean(increments), np.std(increments)
                if standard_deviation > 0:
                    x_fit = np.linspace(centers.min(), centers.max(), 500)
                    gaussian = np.exp(-0.5 * ((x_fit - mean) / standard_deviation) ** 2)
                    gaussian /= standard_deviation * np.sqrt(2.0 * np.pi)
                    axis.plot(x_fit, gaussian, ":", color=line[0].get_color(), label=f"Gaussian fit, l = {lag} px")
    axis.set_yscale("log")
    axis.set_ylim(bottom=1e-4)
    axis.set(xlabel="Velocity increment (km s$^{-1}$)", ylabel="Probability density", title=f"Velocity increments — LOS {LOS_AXIS}")
    axis.legend()
    axis.grid(True, which="both", ls="--", alpha=0.4)
    figure.tight_layout()
    figure.savefig(f"PDF_Velocity_Increments_LOS_{LOS_AXIS}.png", dpi=200)
    header_0 = fits.Header({"BUNIT": "K km/s"})
    header_1 = fits.Header({"BUNIT": "km/s"})
    fits.writeto(f"moment_0_map_LOS_{LOS_AXIS}.fits", moment_0, header_0, overwrite=True)
    fits.writeto(f"moment_1_map_LOS_{LOS_AXIS}.fits", np.nan_to_num(moment_1), header_1, overwrite=True)


if __name__ == "__main__":
    main()
