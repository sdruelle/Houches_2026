"""Fonctions communes pour les diagnostics de cubes PPV HI à 21 cm."""

import numpy as np
from astropy import constants as const
from astropy.io import fits

K_B = const.k_B.cgs.value
M_H = const.m_p.cgs.value
C = const.c.cgs.value
H = const.h.cgs.value
MU = 1.4
NU_UL = 1.420405751e9
A_UL = 2.8843e-15
G_U, G_L = 3.0, 1.0


def _los_data(hdul, axis):
    """Retourne le champ vitesse et les dimensions correspondant à la LOS."""
    density = hdul["DENSITY"].data
    choices = {
        "Z": ("VELOCITY_Z", "CDELT3", 0),
        "Y": ("VELOCITY_Y", "CDELT2", 1),
        "X": ("VELOCITY_X", "CDELT1", 2),
    }
    try:
        velocity_name, spacing_name, axis_index = choices[axis.upper()]
    except KeyError as exc:
        raise ValueError("axis must be 'X', 'Y', or 'Z'") from exc
    return density, hdul["PRESSURE"].data, hdul[velocity_name].data, hdul[0].header.get(spacing_name, 1.0), axis_index


def make_ppv(filename, axis="Z", n_velocity=200, sigma_turb=0.0, with_tau=False):
    """Calcule un cube PPV et les champs physiques utiles aux diagnostics."""
    with fits.open(filename) as hdul:
        density, pressure, velocity, ds_pc, axis_index = _los_data(hdul, axis)

    temperature = pressure * MU * M_H / (density * K_B)
    n_total = density / (MU * M_H)
    n_lower = 0.25 * n_total
    velocity_axis = axis_index
    velocity_width = np.sqrt(K_B * np.max(temperature) / M_H)
    v_max = np.max(np.abs(velocity)) + 5.0 * (velocity_width + sigma_turb)
    v_grid = np.linspace(-v_max, v_max, n_velocity)
    nu_grid = NU_UL * (1.0 - v_grid / C)[:, None, None]
    transverse_shape = tuple(size for i, size in enumerate(density.shape) if i != axis_index)
    intensity = np.zeros((n_velocity, *transverse_shape))
    tau = np.zeros_like(intensity) if with_tau else None

    for index in range(density.shape[axis_index]):
        slice_index = [slice(None)] * 3
        slice_index[axis_index] = index
        slice_index = tuple(slice_index)
        t = np.maximum(temperature[slice_index][None, :, :], 1e-10)
        v = velocity[slice_index][None, :, :]
        n_l = n_lower[slice_index][None, :, :]
        sigma_nu = NU_UL / C * np.sqrt(K_B * t / M_H + sigma_turb**2)
        nu_center = NU_UL * (1.0 - v / C)
        profile = np.exp(-0.5 * ((nu_grid - nu_center) / sigma_nu) ** 2) / (sigma_nu * np.sqrt(2.0 * np.pi))
        opacity = C**2 / (8.0 * np.pi * NU_UL**2) * A_UL * n_l * (G_U / G_L)
        opacity *= 1.0 - np.exp(-H * NU_UL / (K_B * t))
        delta_tau = opacity * profile * ds_pc * const.pc.cgs.value
        source = (2.0 * H * NU_UL**3 / C**2) / np.expm1(H * NU_UL / (K_B * t))
        intensity = intensity * np.exp(-delta_tau) + source * (1.0 - np.exp(-delta_tau))
        if tau is not None:
            tau += delta_tau

    brightness_temperature = intensity * C**2 / (2.0 * K_B * NU_UL**2)
    return {
        "brightness_temperature": brightness_temperature,
        "v_grid": v_grid,
        "tau": tau,
        "n_total": n_total,
        "axis_index": axis_index,
        "ds_cm": ds_pc * const.pc.cgs.value,
    }


def moments(brightness_temperature, v_grid):
    """Calcule les moments 0, 1 et 2, avec un masque de faible émission."""
    velocity_kms = v_grid / 1e5
    moment_0 = np.trapezoid(brightness_temperature, x=velocity_kms, axis=0)
    mask = moment_0 > 1e-3 * np.nanmax(moment_0)
    weighted_v = np.trapezoid(brightness_temperature * velocity_kms[:, None, None], x=velocity_kms, axis=0)
    weighted_v2 = np.trapezoid(brightness_temperature * velocity_kms[:, None, None] ** 2, x=velocity_kms, axis=0)
    moment_1 = np.full_like(moment_0, np.nan)
    moment_2 = np.full_like(moment_0, np.nan)
    moment_1[mask] = weighted_v[mask] / moment_0[mask]
    moment_2[mask] = np.sqrt(np.maximum(weighted_v2[mask] / moment_0[mask] - moment_1[mask] ** 2, 0.0))
    return moment_0, moment_1, moment_2
