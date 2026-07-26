# Houches 2026 — analyse de turbulence

Ce dépôt contient des scripts Python pour analyser des simulations de turbulence
astrophysique RAMSES et des cubes FITS.

## Organisation

```text
scripts/
  simulation/      # export RAMSES → cube FITS
  ppv/             # transfert radiatif et diagnostics PPV
  observations/    # effet instrumental et cartes de moments
  analysis/        # analyses statistiques et spectres de puissance
archive/           # prototype conservé, hors pipeline courant
```

## Pipeline PPV recommandé

Lancer les commandes depuis la racine du dépôt, après avoir adapté les chemins
de données définis dans les scripts :

```bash
# 1. Export RAMSES vers FITS
python scripts/simulation/export_fits.py

# 2. Génération du cube position-position-vitesse
python scripts/ppv/generate_ppv.py

# 3. Simulation de l'observation EBHIS
python scripts/observations/apply_ebhis.py

# 4. Calcul des moments 0, 1 et 2
python scripts/observations/compute_moments.py
```

Diagnostics PPV complémentaires :

```bash
python scripts/ppv/validate_optical_depth.py
python scripts/ppv/velocity_pdfs.py
```

## Analyses disponibles

- `scripts/analysis/compute_power_spectra.py` : calcul des spectres de puissance.
- `scripts/analysis/plot_power_spectra.py` : tracé de spectres existants.
- `scripts/analysis/plot_field_gradient_alignment.py` : alignement champ magnétique–gradient de densité.
- `scripts/analysis/nongaussian_segmentation.py` : segmentation non gaussienne.

## Dépendances

Pour le pipeline FITS :

```bash
pip install numpy scipy matplotlib astropy
```

Selon les scripts utilisés, `yt`, `h5py`, `pspec` et `pywavan` peuvent aussi
être nécessaires.

## Archive

`archive/line_of_sight_intensity_prototype.py` est un prototype conservé pour
référence. Le pipeline actuel utilise `scripts/ppv/generate_ppv.py`.
