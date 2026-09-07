"""The empirical NULL cvR2: predict the per-voxel TRAINING-set mean.

No encoding model was ever fitted for this. The pipeline uses cvR2 > 0 as the
"non-noise" threshold, but cvR2 is computed by braincoder's get_rsq as

    1 - sum((y - yhat)^2) / sum((y - y_test.mean())^2)

i.e. the denominator uses the HELD-OUT fold's own mean. So cvR2 = 0 is "as good as
knowing the test fold's mean", which is a slightly optimistic reference: a real null
predictor only has the TRAINING mean, and train-vs-test mean drift makes its cvR2
slightly negative. This measures how far below zero that actually sits, under exactly
the same leave-one-run-out scheme the CV fits use.
"""
import warnings
import numpy as np, pandas as pd
from pathlib import Path
from nilearn.maskers import NiftiMasker
from tms_risk.utils.data import Subject
warnings.filterwarnings('ignore')

BIDS = Path('/data/ds-tmsrisk')
DERIV = BIDS / 'derivatives'
SUBS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31, 34, 35, 36,
        37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]


def rsq(data, pred):
    ssr = ((data - pred) ** 2).sum(0)
    sst = ((data - data.mean(0)) ** 2).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = 1 - ssr / sst
    out[sst == 0] = np.nan
    return out


rows = []
for s in SUBS:
    sid = f'{s:02d}'
    try:
        sub = Subject(sid, bids_folder=BIDS)
        par = sub.get_paradigm().reset_index('session')[['log(n1)', 'session']]
        masker = NiftiMasker(mask_img=sub.get_volume_mask(session=1, roi=None,
                                                          epi_space=True))
        d = []
        for ses in [2, 3]:
            fn = (DERIV / 'glm_stim1.denoise.smoothed' / f'sub-{sid}' / f'ses-{ses}'
                  / 'func' / f'sub-{sid}_ses-{ses}_task-task_space-T1w_desc-stims1_pe.nii.gz')
            d.append(masker.fit_transform(str(fn)))
        data = np.vstack(d).astype(np.float32)
        roi = masker.transform(sub.get_volume_mask(session=1, roi='NPC12r',
                                                   epi_space=True)).squeeze() > 0
    except Exception as e:
        print(f'  sub-{sid}: {type(e).__name__}'); continue

    runs = par.index.get_level_values('run')
    folds = []
    for r in sorted(np.unique(runs)):
        te, tr = runs == r, runs != r
        # the null predictor: the training folds' mean for that voxel
        pred = np.repeat(data[tr].mean(0)[None, :], te.sum(), axis=0)
        folds.append(rsq(data[te], pred))
    null = np.nanmean(np.stack(folds), axis=0)
    rows.append(dict(subject=s,
                     null_wholebrain=np.nanmean(null),
                     null_npc12r=np.nanmean(null[roi]),
                     frac_gt0_npc12r=np.nanmean(null[roi] > 0),
                     q95_npc12r=np.nanpercentile(null[roi], 95)))
    print(f'  sub-{sid}: null cvR2 NPC12r {rows[-1]["null_npc12r"]:+.5f}', flush=True)

R = pd.DataFrame(rows)
print('\n=== EMPIRICAL NULL (predict training-set mean), leave-one-run-out ===')
print(f'n = {len(R)} subjects')
print(f'  mean null cvR2, NPC12r     : {R.null_npc12r.mean():+.5f} '
      f'(SD {R.null_npc12r.std():.5f})')
print(f'  mean null cvR2, whole brain: {R.null_wholebrain.mean():+.5f}')
print(f'  fraction of NPC12r voxels where the NULL alone exceeds 0: '
      f'{R.frac_gt0_npc12r.mean():.4f}')
print(f'  95th pct of null cvR2 within NPC12r: {R.q95_npc12r.mean():+.5f}')
print('\nFor comparison, m1 (10000 it) mean cvR2 in NPC12r is about -0.0059 and '
      '34.6% of voxels exceed 0.')
R.to_csv('notes/data/null_cvr2.tsv', sep='\t', index=False)
print('wrote notes/data/null_cvr2.tsv')
