"""Correlate a per-participant cTBS model contrast with the cTBS nPRF amplitude
change -- once per posterior draw, so the answer is a posterior over r.

A per-participant estimate from a hierarchical model is a posterior, not a
number. Correlating its posterior MEAN with a neural measure and putting a
bootstrap CI on the result asks the wrong question twice over: it throws away
the per-participant uncertainty, and then reconstructs an interval by resampling
35 shrunken point estimates. What the model actually supports is

    for each posterior draw d:  r_d = corr( Δ_{·,d} , amplitude change )

whose spread across draws already contains the measurement error in Δ. No
attenuation correction is needed and none is applied: a parameter that is
poorly identified per participant simply produces a wide posterior on r, which
is the honest answer. `notes/data/subject_reliability.*.tsv` says how much of
that width is measurement error, but the interval here stands on its own.

Reports the median r, the 95% credible interval, and P(r > 0) -- Pearson and
Spearman -- for every parameter of the model that carries a cTBS regressor.

    python -m tms_risk.behavior.scripts.anchor_brain_behavior_posterior \\
        --label log-power-perc.mapjitter.klw \\
        --trace_dir /shares/.../derivatives/cogmodels.anchor
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
#: the individually defined stimulation-site mask, cvR2-positive voxels --
#: the same neural measure as the model-free brain-behaviour analysis
MASK, SELECTION, AMP = 'NPCr2cm-cluster', 'cvr2pos', 'd_amp_median'


def _rank(a):
    order = np.argsort(a, axis=-1, kind='stable')
    r = np.empty_like(order, dtype=float)
    np.put_along_axis(r, order,
                      np.broadcast_to(np.arange(a.shape[-1], dtype=float),
                                      a.shape).copy(), axis=-1)
    return r


def _corr(x, y):
    """corr of each row of x (n_draw, n_sub) with the vector y (n_sub,)."""
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean()
    den = np.sqrt((x ** 2).sum(-1) * (y ** 2).sum())
    return (x * y).sum(-1) / np.where(den > 0, den, np.nan)


def main(label, trace_dir, neural_tsv, mask, selection, amp_col, out_tsv):
    import xarray as xr
    neu = pd.read_csv(neural_tsv, sep='\t')
    neu = neu[(neu['mask'] == mask) & (neu['selection'] == selection)]
    neu = neu.set_index('subject')[amp_col].dropna()

    ds = xr.open_dataset(Path(trace_dir) / f'model-{label}_trace.netcdf',
                         group='posterior')
    subj = np.array([int(s) for s in ds['subject'].values])
    keep = np.array([s in neu.index for s in subj])
    y = neu.loc[subj[keep]].values
    rows = []
    for p in ds.attrs['tms_risk_parameters'].split(','):
        rdim = f'{p}_regressors'
        if p not in ds or ds.sizes.get(rdim, 1) < 2:
            continue
        coef = (ds[p].stack(sample=('chain', 'draw'))
                .transpose('subject', 'sample', rdim).values)
        d = (-coef[..., 1])[keep].T                     # (n_draw, n_subject)
        for kind, x in (('pearson', d), ('spearman', _rank(d))):
            r = _corr(x, y if kind == 'pearson' else _rank(y[None])[0])
            rows.append(dict(
                label=label, parameter=p, kind=kind, n=int(keep.sum()),
                r=float(np.median(r)),
                lo=float(np.quantile(r, .025)), hi=float(np.quantile(r, .975)),
                p_gt0=float((r > 0).mean()), p_lt0=float((r < 0).mean()),
                r_posterior_mean_only=float(_corr(d.mean(0)[None], y)[0])))
    ds.close()
    out = pd.DataFrame(rows)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep='\t', index=False)
    pd.set_option('display.width', 220)
    print(f'{mask} / {selection} / {amp_col}, n = {int(keep.sum())}')
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nwrote {out_tsv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='log-power-perc.mapjitter.klw')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--neural_tsv', default=str(REPO / 'notes/data/bb_neural.tsv'))
    ap.add_argument('--mask', default=MASK)
    ap.add_argument('--selection', default=SELECTION)
    ap.add_argument('--amp_col', default=AMP)
    ap.add_argument('--out_tsv', default=None)
    a = ap.parse_args()
    main(a.label, a.trace_dir, a.neural_tsv, a.mask, a.selection, a.amp_col,
         a.out_tsv or str(REPO / f'notes/data/bb_posterior.{a.label}.tsv'))
