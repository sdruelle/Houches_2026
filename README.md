# Houches 2026 — turbulence analysis

This repository contains Python scripts to analyze RAMSES astrophysical turbulence simulations and FITS cubes.

## Organization

```text
scripts/
  simulation/      # RAMSES export → FITS cube
  ppv/             # radiative transfer and PPV diagnostics
  observations/    # instrumental effect and moment maps
  analysis/        # statistical analyses and power spectra
archive/           # preserved prototype, outside the main pipeline
```

## Recommended PPV pipeline

Run the commands from the root of the repository, after adapting the data paths defined in the scripts:

```bash
# 1. Export RAMSES to FITS
python scripts/simulation/export_fits.py

# 2. Position-position-velocity cube generation
python scripts/ppv/generate_ppv.py

# 3. EBHIS observation simulation
python scripts/observations/apply_ebhis.py

# 4. Computation of moments 0, 1, and 2
python scripts/observations/compute_moments.py
```

Additional PPV diagnostics:

```bash
python scripts/ppv/validate_optical_depth.py
python scripts/ppv/velocity_pdfs.py
```

## Available analyses

- `scripts/analysis/compute_power_spectra.py`: computation of power spectra.
- `scripts/analysis/plot_power_spectra.py`: plotting of existing spectra.
- `scripts/analysis/plot_field_gradient_alignment.py`: magnetic field–density gradient alignment.
- `scripts/analysis/nongaussian_segmentation.py`: non-Gaussian segmentation.

## Dependencies

For the FITS pipeline:

```bash
pip install numpy scipy matplotlib astropy
```

Depending on the scripts used, `yt`, `h5py`, `pspec`, and `pywavan` may also be required.
