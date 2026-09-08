"""How much of the between-participant spread in a cTBS contrast is real?

A per-participant estimate from a hierarchical model carries its own posterior
uncertainty. If that uncertainty is large relative to the spread across
participants, most of the apparent individual differences are measurement
error, and no correlation with an external measure (nPRF amplitude, risk
attitude) can exceed the attenuation ceiling however real the association is.

Three estimators, because the obvious one is wrong in a specific way.

1. `reliability_var` -- the classical variance correction,

       (Var_s[E_d Δ_s] − median_s Var_d[Δ_s]) / Var_s[E_d Δ_s]

   This is the one to distrust here. Its numerator subtracts the within-subject
   variance from the spread of the POSTERIOR MEANS, but partial pooling has
   already shrunk those means towards the group: the denominator is a shrunken
   quantity and the correction is applied a second time. It therefore
   understates reliability, sometimes to zero, and gets it wrong most exactly
   when shrinkage is strongest -- i.e. when it matters.

2. `reliability_draw` -- the correlation between two INDEPENDENT posterior
   draws of the whole subject vector,

       E_{d ≠ d'} corr( Δ_{·,d}, Δ_{·,d'} )

   If the posterior across subjects is (conditionally) independent, this
   expectation equals σ²_true / (σ²_true + σ²_within): exactly the reliability,
   with no shrinkage double-count and no way to go negative. Two draws are two
   equally valid readings of the same data, so their agreement is a genuine
   internal-consistency coefficient.

3. `rank_stability` -- the same thing with Spearman instead of Pearson. This is
   what actually licenses a statement like "the same participants come out at
   the top", which is what a correlation with an external measure needs.

`ceiling_*` is sqrt of the corresponding reliability: the largest correlation
this measure could show with a PERFECTLY measured external variable. An observed
r should be read against it, not against 1.

    # on the cluster, where the traces are
    python -m tms_risk.behavior.scripts.anchor_subject_reliability \\
        --label log-power-perc.mapjitter.klw \\
        --trace_dir /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]
N_PAIRS = 2000          # draw pairs averaged over; SE of the mean r is <0.005

def ctbs_contrast(coef, colnames):
    """IPS - vertex from a regressor axis, whatever the contrast coding.

    Treatment coding (`stimulation_condition`) puts the IPS cell in the
    intercept and vertex - IPS in column 1, so the contrast is -coef[1]. Sum
    coding (`C(stimulation_condition, Sum)`) puts the grand mean in the
    intercept and (IPS - grand mean) in column 1, with vertex = -that, so the
    contrast is 2*coef[1]. Getting this wrong is silent -- same shape, same
    sign in some models -- so it is read off the column NAME, never assumed.
    """
    name = str(colnames[1])
    if 'Sum)' in name or '[S.' in name:
        return 2.0 * coef[..., 1]
    if 'T.vertex' in name:
        return -coef[..., 1]
    raise ValueError(f'unrecognised cTBS regressor column {name!r}')



def _rankdata(a):
    """Average-rank transform along the last axis (avoids a scipy import)."""
    order = np.argsort(a, axis=-1, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    np.put_along_axis(ranks, order,
                      np.broadcast_to(np.arange(a.shape[-1], dtype=float),
                                      a.shape).copy(), axis=-1)
    return ranks


def _paired_corr(x, y):
    """Row-wise Pearson correlation of two (n_pairs, n_subjects) arrays."""
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    num = (x * y).sum(-1)
    den = np.sqrt((x ** 2).sum(-1) * (y ** 2).sum(-1))
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def from_trace(label, trace_dir, rng):
    """Per-parameter reliability from the (subject x sample) contrast draws."""
    import xarray as xr
    ds = xr.open_dataset(Path(trace_dir) / f'model-{label}_trace.netcdf',
                         group='posterior')
    names = ds.attrs['tms_risk_parameters'].split(',')
    rows = []
    for p in names:
        rdim = f'{p}_regressors'
        if p not in ds or ds.sizes.get(rdim, 1) < 2:
            continue                       # no cTBS term on this parameter
        coef = (ds[p].stack(sample=('chain', 'draw'))
                .transpose('subject', 'sample', rdim).values)
        # index 0 is the IPS intercept, index 1 the vertex OFFSET, so the
        # IPS - vertex contrast is minus the second coefficient
        d = ctbs_contrast(coef, ds[rdim].values)           # (subject, sample)
        rows.append(_reliability(label, p, d, rng))
    ds.close()
    return rows


def _reliability(label, par, d, rng):
    n_sub, n_draw = d.shape
    mean_s = d.mean(1)                                  # posterior mean/subject
    sd_within_s = d.std(1, ddof=1)                      # posterior SD/subject
    var_between_obs = float(np.var(mean_s, ddof=1))
    var_within = float(np.median(sd_within_s ** 2))
    rel_var = max(var_between_obs - var_within, 0.) / var_between_obs \
        if var_between_obs > 0 else 0.

    i = rng.integers(0, n_draw, N_PAIRS)
    j = rng.integers(0, n_draw, N_PAIRS)
    j = np.where(j == i, (j + 1) % n_draw, j)           # force d != d'
    a, b = d[:, i].T, d[:, j].T                         # (N_PAIRS, n_subjects)
    rel_draw = float(np.nanmean(_paired_corr(a, b)))
    rank = float(np.nanmean(_paired_corr(_rankdata(a), _rankdata(b))))

    # the between-subject SD the model itself implies, per draw -- unshrunk
    sd_between_model = float(np.mean(d.std(0, ddof=1)))
    return dict(
        label=label, parameter=par, n=n_sub,
        sd_within=float(np.sqrt(var_within)),
        sd_between_observed=float(np.sqrt(var_between_obs)),
        sd_between_model=sd_between_model,
        reliability_var=rel_var, ceiling_var=float(np.sqrt(rel_var)),
        reliability_draw=rel_draw,
        ceiling_draw=float(np.sqrt(max(rel_draw, 0.))),
        rank_stability=rank,
        ceiling_rank=float(np.sqrt(max(rank, 0.))))


def from_summary(label, data_dir, rng):
    """Fallback when only the summary TSV is at hand: no draws, so only the
    variance correction is available. Reported for continuity; prefer the
    trace."""
    f = Path(data_dir) / 'subject_params' / f'subject_params.{label}.tsv'
    d = pd.read_csv(f, **READ)
    rows = []
    for par, g in d[d.subject != 'GROUP'].groupby('parameter'):
        mid = g['mid'].values
        sd_within = (g['hi'].values - g['lo'].values) / (2 * 1.96)
        var_between_obs = float(np.var(mid, ddof=1))
        var_within = float(np.median(sd_within ** 2))
        rel = max(var_between_obs - var_within, 0.) / var_between_obs \
            if var_between_obs > 0 else 0.
        rows.append(dict(label=label, parameter=par, n=len(g),
                         sd_within=float(np.sqrt(var_within)),
                         sd_between_observed=float(np.sqrt(var_between_obs)),
                         sd_between_model=np.nan,
                         reliability_var=rel, ceiling_var=float(np.sqrt(rel)),
                         reliability_draw=np.nan, ceiling_draw=np.nan,
                         rank_stability=np.nan, ceiling_rank=np.nan))
    return rows


def main(label, trace_dir, data_dir, out_tsv, seed):
    rng = np.random.default_rng(seed)
    rows = (from_trace(label, trace_dir, rng) if trace_dir
            else from_summary(label, data_dir, rng))
    out = pd.DataFrame(rows).sort_values('rank_stability' if trace_dir
                                         else 'reliability_var',
                                         ascending=False)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep='\t', index=False)
    pd.set_option('display.width', 250)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nwrote {out_tsv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', '--model_label', dest='label',
                    default='log-power-perc.mapjitter.klw')
    ap.add_argument('--trace_dir', default=None,
                    help='read the posterior draws (preferred). Without it, '
                         'only the shrinkage-biased variance correction can be '
                         'computed from the summary TSV.')
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--out_tsv', default=None)
    ap.add_argument('--seed', default=0, type=int)
    a = ap.parse_args()
    main(a.label, a.trace_dir, a.data_dir,
         a.out_tsv or str(REPO / f'notes/data/subject_reliability.{a.label}.tsv'),
         a.seed)
