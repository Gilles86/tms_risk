"""Rebuild the subject-level cvR2 map by averaging the per-fold maps already on disk.

`fit_regression_nprf_cv.py` writes one cvR2 NIfTI per left-out run and *then* averages
them into `sub-XX_desc-cvr2.optim_space-T1w_pars.nii.gz`. Until 2026-08-06 that last
line used `groupby(..., axis=0)`, removed in pandas 2.x, so every job crashed there --
after all six folds had been fitted and written. The fits are therefore intact and only
the average is missing; this recovers it in seconds instead of re-running hours of GPU
time per subject.

Also useful whenever a CV array is killed partway: it simply skips subjects that do not
yet have the full set of folds.

    python -m tms_risk.modeling.scripts.aggregate_cv_r2 --model_label 1 \
        --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk
"""
import argparse
import re
from pathlib import Path

import nibabel as nb
import numpy as np


def main(bids_folder, model_label, n_runs, overwrite):
    root = (Path(bids_folder) / 'derivatives'
            / f'encoding_model2.model-{model_label}.smoothed.cv')
    if not root.exists():
        raise SystemExit(f'no such directory: {root}')

    made, skipped, already = 0, [], 0
    for sub_dir in sorted(root.glob('sub-*')):
        sub = sub_dir.name.replace('sub-', '')
        out = sub_dir / f'sub-{sub}_desc-cvr2.optim_space-T1w_pars.nii.gz'
        folds = sorted(sub_dir.glob(f'sub-{sub}_run-*_desc-cvr2.optim_space-T1w_pars.nii.gz'),
                       key=lambda p: int(re.search(r'run-(\d+)', p.name).group(1)))
        if out.exists() and not overwrite:
            already += 1
            continue
        if len(folds) < n_runs:
            skipped.append((sub, len(folds)))
            continue

        imgs = [nb.load(str(f)) for f in folds]
        data = np.stack([i.get_fdata() for i in imgs])
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(data, axis=0)
        img = nb.Nifti1Image(mean.astype(np.float32), imgs[0].affine, imgs[0].header)
        # NIfTI dtype trap: inherit nothing from a uint8 mask, and kill any scl_slope
        # so the parameter is not quantised to 256 levels across the brain.
        img.set_data_dtype(np.float32)
        img.header.set_slope_inter(slope=1, inter=0)
        img.to_filename(str(out))
        made += 1

    print(f'model-{model_label}: wrote {made}, already present {already}, '
          f'incomplete {len(skipped)}')
    for sub, n in skipped:
        print(f'  sub-{sub}: only {n}/{n_runs} folds')
    if made:
        chk = nb.load(str(out))
        d = chk.get_fdata()
        print(f'  sanity on the last written: dtype {chk.get_data_dtype()}, '
              f'{len(np.unique(np.round(d[np.isfinite(d)], 3)))} unique values, '
              f'mean {np.nanmean(d):.5f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--model_label', default=1, type=int)
    p.add_argument('--n_runs', default=6, type=int)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    main(a.bids_folder, a.model_label, a.n_runs, a.overwrite)
