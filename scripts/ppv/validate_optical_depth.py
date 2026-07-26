"""Vérifie l'approximation optiquement mince pour un cube PPV HI."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppv_core import make_ppv, moments

INPUT_CUBE = "cube_info_00051.fits"
LOS_AXIS = "Z"


def main():
    result = make_ppv(INPUT_CUBE, axis=LOS_AXIS, with_tau=True)
    brightness_temperature = result["brightness_temperature"]
    velocity = result["v_grid"]
    tau = result["tau"]
    moment_0, _, _ = moments(brightness_temperature, velocity)
    true_column_density = np.sum(result["n_total"], axis=result["axis_index"]) * result["ds_cm"]
    inferred_column_density = 1.823e18 * moment_0
    tau_max = np.max(tau, axis=0)
    tau_total = np.trapezoid(tau, x=velocity / 1e5, axis=0)
    ratio = np.full_like(true_column_density, np.nan)
    valid = (tau_max < 1.0) & (true_column_density > 0)
    ratio[valid] = inferred_column_density[valid] / true_column_density[valid]

    panels = [
        (np.log10(true_column_density + 1e-10), "True column density", "viridis"),
        (tau_max, "Maximum optical depth", "magma"),
        (tau_total, "Integrated optical depth", "magma"),
        (np.log10(inferred_column_density + 1e-10), "Inferred column density", "viridis"),
        (ratio, "Inferred / true ($\\tau < 1$)", "RdBu_r"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(20, 12))
    for axis, (data, title, cmap) in zip(axes.flat, panels):
        image = axis.imshow(data, origin="lower", cmap=cmap)
        figure.colorbar(image, ax=axis)
        axis.set_title(title)
    axes[1, 2].axis("off")
    figure.tight_layout()
    figure.savefig(f"Optical_Thickness_Validation_LOS_{LOS_AXIS}.png")


if __name__ == "__main__":
    main()
