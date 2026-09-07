"""Per-voxel nPRF parameters by stimulation condition, for every model label.

Runs where the fits are (the cluster) and writes one small TSV to pull back, per the
repo's two-stage rule. One row per (model, subject, voxel) with each parameter given
per arm, plus the main fit's in-sample r2 so the plotting side can pick a signal
criterion without needing the NIfTIs.

    python -m tms_risk.modeling.scripts.extract_prf_params_by_condition \
        --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk --tree encoding_model2.refit2026 \
        --roi NPCr2cm-cluster --out_tsv /data/prf_params_by_condition.tsv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from tms_risk.utils.data import Subject, get_all_behavior

warnings.filterwarnings('ignore')

SUBS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31, 34, 35, 36,
        37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
PARAMS = ['mu', 'sd', 'amplitude', 'baseline']


def main(bids_folder, tree, roi, models, out_tsv):
    bids = Path(bids_folder)
    deriv = bids / 'derivatives'
    beh = get_all_behavior(bids_folder=bids).reset_index()
    arm = {(int(r.subject), int(r.session)): r.stimulation_condition
           for r in beh[['subject', 'session', 'stimulation_condition']]
           .drop_duplicates().itertuples()}

    frames = []
    for s in SUBS:
        sid = f'{s:02d}'
        try:
            sub = Subject(sid, bids_folder=bids)
            masker = NiftiMasker(mask_img=sub.get_volume_mask(session=1, roi=None,
                                                              epi_space=True))
            masker.fit()
            keep = masker.transform(sub.get_volume_mask(session=1, roi=roi,
                                                        epi_space=True)).squeeze() > 0
        except Exception as e:
            print(f'  sub-{sid}: {type(e).__name__}'); continue
        sessions = {ses: arm.get((s, ses)) for ses in (2, 3)}
        if set(sessions.values()) != {'ips', 'vertex'}:
            continue

        for m in models:
            root = deriv / f'{tree}.model-{m}.smoothed' / f'sub-{sid}'
            if not root.exists():
                continue
            d = {'model': m, 'subject': s, 'roi': roi,
                 'voxel': np.where(keep)[0]}
            try:
                d['r2'] = masker.transform(
                    str(root / f'sub-{sid}_desc-r2.optim_space-T1w_pars.nii.gz')
                ).squeeze()[keep]
                for ses, a in sessions.items():
                    for par in PARAMS:
                        fn = (root / f'ses-{ses}'
                              / f'sub-{sid}_ses-{ses}_desc-{par}.optim_space-T1w_pars.nii.gz')
                        d[f'{par}_{a}'] = masker.transform(str(fn)).squeeze()[keep]
            except Exception as e:
                print(f'  sub-{sid} m{m}: {type(e).__name__}'); continue
            frames.append(pd.DataFrame(d))
        print(f'  sub-{sid} done', flush=True)

    out = pd.concat(frames, ignore_index=True)
    for a in ('ips', 'vertex'):
        out[f'pref_n_{a}'] = np.exp(out[f'mu_{a}'])
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out.shape} to {out_tsv}')
    # sanity: pooled parameters must be bit-identical across arms
    for m in sorted(out.model.unique()):
        g = out[out.model == m]
        same = [p for p in PARAMS
                if np.nanmax(np.abs(g[f'{p}_ips'] - g[f'{p}_vertex'])) < 1e-9]
        print(f'  m{m}: identical across arms -> {same or "none"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--tree', default='encoding_model2.refit2026')
    p.add_argument('--roi', default='NPCr2cm-cluster')
    p.add_argument('--models', default='0,1,2,3,4,5')
    p.add_argument('--out_tsv', default='/data/prf_params_by_condition.tsv')
    a = p.parse_args()
    main(a.bids_folder, a.tree, a.roi, [int(x) for x in a.models.split(',')], a.out_tsv)
