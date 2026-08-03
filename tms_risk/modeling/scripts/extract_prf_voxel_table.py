"""Per-voxel nPRF parameter table for the numerosity ROIs, all subjects.

Serves the tuning-width analyses and the amplitude/dispersion trade-off check. Reads
only parameter and cvR2 maps -- no GLM betas, so it is fast.

Under m1 (`amplitude` is the only per-session regressor) `mu` and `sd` are shared
across sessions by construction, so they are written once per voxel. `amplitude` and
`baseline` are written per stimulation arm. m2's per-session `mu`/`sd`/`amplitude` are
written alongside for the trade-off check.

Units: the paradigm is x = log(n1), so `mu` is log preferred numerosity and `sd` is a
tuning width IN LOG UNITS (a Weber-like coefficient, not a numerosity). Preferred
numerosity in natural units is exp(mu).

    python -m tms_risk.modeling.scripts.extract_prf_voxel_table \
        --bids_folder /data/ds-tmsrisk --out notes/data/prf_voxel_table.tsv
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from tms_risk.utils.data import Subject, get_all_behavior

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31,
            34, 35, 36, 37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
ROIS = ['NPC12r', 'NPCr', 'NPCl', 'NF1', 'NTO', 'NPCr2cm-cluster', 'NPCr2cm-surface']
SESSIONS = [2, 3]


def main(bids_folder, out_fn, subjects):
    bids = Path(bids_folder)
    deriv = bids / 'derivatives'
    beh = get_all_behavior(bids_folder=bids)
    cond = (beh.reset_index()[['subject', 'session', 'stimulation_condition']]
               .drop_duplicates())

    frames = []
    for i, s in enumerate(subjects):
        t0 = time.time()
        sid = f'{s:02d}'
        try:
            sub = Subject(sid, bids_folder=bids)
            masker = NiftiMasker(mask_img=sub.get_volume_mask(session=1, roi=None,
                                                             epi_space=True))
            masker.fit()
        except Exception as e:
            logging.error(f'sub-{sid}: {e}')
            continue

        c = cond[cond.subject == s]
        cmap = {int(k): v for k, v in zip(c.session.astype(int), c.stimulation_condition)
                if int(k) in SESSIONS}
        if set(cmap.values()) != {'ips', 'vertex'}:
            logging.warning(f'sub-{sid}: arms {cmap}, skipping')
            continue

        roi_flags = {}
        for roi in ROIS:
            try:
                m = sub.get_volume_mask(session=1, roi=roi, epi_space=True)
                roi_flags[roi] = masker.transform(m).squeeze() > 0
            except Exception:
                pass
        if 'NPC12r' not in roi_flags:
            logging.warning(f'sub-{sid}: no NPC12r, skipping')
            continue
        keep = np.any(np.stack(list(roi_flags.values())), axis=0)

        def rd(p):
            return masker.transform(str(p)).squeeze()

        d = {'subject': s, 'voxel': np.where(keep)[0]}
        for roi, v in roi_flags.items():
            d[f'in_{roi}'] = v[keep]

        for m in [0, 1, 2]:
            root = deriv / f'encoding_model2.model-{m}.smoothed' / f'sub-{sid}'
            cvfn = (deriv / f'encoding_model2.model-{m}.smoothed.cv' / f'sub-{sid}'
                    / f'sub-{sid}_desc-cvr2.optim_space-T1w_pars.nii.gz')
            try:
                d[f'r2_m{m}'] = rd(root / f'sub-{sid}_desc-r2.optim_space-T1w_pars.nii.gz')[keep]
                d[f'cvr2_m{m}'] = rd(cvfn)[keep]
            except Exception as e:
                logging.warning(f'sub-{sid} m{m} r2/cvr2: {e}')
            for ses in SESSIONS:
                arm = cmap[ses]
                for p in ['mu', 'sd', 'amplitude', 'baseline']:
                    fn = (root / f'ses-{ses}'
                          / f'sub-{sid}_ses-{ses}_desc-{p}.optim_space-T1w_pars.nii.gz')
                    try:
                        d[f'{p}_m{m}_{arm}'] = rd(fn)[keep]
                    except Exception as e:
                        logging.warning(f'sub-{sid} m{m} {p} ses-{ses}: {e}')

        frames.append(pd.DataFrame(d))
        print(f'[{i+1}/{len(subjects)}] sub-{sid} {keep.sum()} voxels '
              f'({time.time()-t0:.1f}s)', flush=True)

    out = pd.concat(frames, ignore_index=True)
    # m1: mu/sd identical across arms by construction -- collapse and verify
    for p in ['mu', 'sd']:
        a, b = out[f'{p}_m1_ips'], out[f'{p}_m1_vertex']
        bad = float(np.nanmax(np.abs(a - b)))
        print(f'CHECK m1 {p}: max |ips - vertex| = {bad:.3e} (must be 0 by construction)')
        out[f'{p}_m1'] = a
        out.drop(columns=[f'{p}_m1_ips', f'{p}_m1_vertex'], inplace=True)
    out['pref_n_m1'] = np.exp(out.mu_m1)
    Path(out_fn).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_fn, sep='\t', index=False)
    print(f'wrote {out.shape} to {out_fn}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--out', default='notes/data/prf_voxel_table.tsv')
    p.add_argument('--subjects', default='')
    a = p.parse_args()
    subs = [int(x) for x in a.subjects.split(',')] if a.subjects else SUBJECTS
    main(a.bids_folder, a.out, subs)
