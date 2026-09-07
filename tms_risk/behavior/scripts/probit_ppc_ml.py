"""Figure 3B as a posterior predictive check, via ML probits on simulated choices.

Figure 3B's cTBS effect on the psychometric SLOPE comes from a hierarchical
probit. To ask whether the PMC model reproduces it, the same estimator has to be
applied to simulated choices -- and applied many times, because the comparison
is between the observed value and the PREDICTIVE DISTRIBUTION of that value.

Refitting the Bayesian model per draw is unnecessary: only a point estimate is
needed from each simulated dataset, since the predictive spread comes from
variation ACROSS draws, not from any single fit's posterior. A maximum-likelihood
probit with subject dummies recovers the published posterior means to within
0.04 in every cell, in half a second:

    cell                     ML       published (hierarchical)
    Risky second, low     -0.702      -0.667 [-1.043, -0.286]
    Risky second, high    -0.146      -0.148
    Risky first,  low     -0.230      -0.189
    Risky first,  high    +0.122      +0.125

so it is used here for both sides. Subject dummies stand in for the random
intercept; the shrinkage difference is smaller than the agreement above.

Reports, per (order, stake) cell, the cTBS effect on the slope and on the
risk-neutral probability, with the observed value, the predictive interval, and
a posterior predictive p-value.

    python -m tms_risk.behavior.scripts.probit_ppc_ml log-power-n1n2 \
        --trace_dir .../cogmodels.anchor
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

warnings.filterwarnings('ignore')

#: reference cell is IPS, low stake, risky SECOND, matching `fit_probit`
CELLS = {('Risky second', 'low'):  (0., 0.),      # (rf, hi)
         ('Risky second', 'high'): (0., 1.),
         ('Risky first', 'low'):   (1., 0.),
         ('Risky first', 'high'):  (1., 1.)}


def _term(p, *parts):
    """Coefficient for an interaction, whatever order patsy names it in."""
    from itertools import permutations
    for perm in permutations(parts):
        n = ':'.join(perm)
        if n in p.index:
            return float(p[n])
    return 0.0


def prep(bids_folder):
    from tms_risk.behavior.fit_probit import get_data
    d = get_data('probit_average_n_full', bids_folder).reset_index()
    d['chose_risky'] = d['chose_risky'].astype(float)
    d['hi'] = (d['average_n_bin'].astype(str) == 'high').astype(float)
    d['rf'] = d['risky_first'].astype(float)
    d['v'] = (d['stimulation_condition'] == 'vertex').astype(float)
    d['subject'] = d['subject'].astype(str)
    y, X = patsy.dmatrices('chose_risky ~ x*rf*v*hi + C(subject)', d,
                           return_type='dataframe')
    return d, X


def fit(X, y):
    """ML probit; returns {cell: (dslope, drnp)}, IPS minus vertex."""
    try:
        r = sm.GLM(y, X, family=sm.families.Binomial(
            sm.families.links.Probit())).fit()
    except Exception:                                       # noqa: BLE001
        return None
    p = r.params
    if not np.isfinite(p.values).all():
        return None
    # group intercept = mean over subjects of the dummy-coded intercepts; the
    # omitted reference subject contributes 0
    sub = [n for n in p.index if n.startswith('C(subject)')]
    b0 = float(p['Intercept']) + float(np.sum([p[n] for n in sub])) / (len(sub) + 1)

    def slope(rf, v, hi):
        return (_term(p, 'x') + rf * _term(p, 'x', 'rf') + v * _term(p, 'x', 'v')
                + hi * _term(p, 'x', 'hi')
                + rf * v * _term(p, 'x', 'rf', 'v')
                + rf * hi * _term(p, 'x', 'rf', 'hi')
                + v * hi * _term(p, 'x', 'v', 'hi')
                + rf * v * hi * _term(p, 'x', 'rf', 'v', 'hi'))

    def icpt(rf, v, hi):
        return (b0 + rf * _term(p, 'rf') + v * _term(p, 'v') + hi * _term(p, 'hi')
                + rf * v * _term(p, 'rf', 'v') + rf * hi * _term(p, 'rf', 'hi')
                + v * hi * _term(p, 'v', 'hi')
                + rf * v * hi * _term(p, 'rf', 'v', 'hi'))

    out = {}
    for cell, (rf, hi) in CELLS.items():
        s_i, s_v = slope(rf, 0., hi), slope(rf, 1., hi)
        # indifference at x* = -b0/b1; RNP = 1/frac* = exp(b0/b1)
        rnp = [np.exp(icpt(rf, v, hi) / sv) if sv > 0.3 else np.nan
               for v, sv in [(0., s_i), (1., s_v)]]
        out[cell] = (s_i - s_v, rnp[0] - rnp[1])
    return out


def main(label, trace_dir, bids_folder, out_dir, n_draws):
    import arviz as az
    import pymc as pm
    from tms_risk.behavior.fit_anchor import build_model
    from tms_risk.behavior.fit_model import get_data

    d, X = prep(bids_folder)
    obs = fit(X, d[['chose_risky']])
    print('observed:', {k: (round(v[0], 3), round(v[1], 3)) for k, v in obs.items()})

    df = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm')
    idata = az.from_netcdf(Path(trace_dir) / f'model-{label}_trace.netcdf')
    model = build_model(label, df)
    model.build_estimation_model(save_p_choice=True)
    nc = idata.posterior.sizes['chain']
    keep = np.linspace(0, idata.posterior.sizes['draw'] - 1,
                       max(1, n_draws // nc)).astype(int)
    det = pm.compute_deterministics(idata.posterior.isel(draw=keep),
                                    model=model.estimation_model,
                                    var_names=['p'], merge_dataset=False,
                                    progressbar=False)
    p2 = det['p'].stack(sample=('chain', 'draw')).values
    p2 = p2[:, np.isfinite(p2).all(0)]
    raw = df.reset_index()
    p_risky = np.where((~raw['risky_first']).values[:, None], p2, 1 - p2)
    gap = float(p_risky.mean() - raw['chose_risky'].astype(float).mean())
    print(f'{label}: {p_risky.shape[1]} draws, grand-mean gap {gap:+.4f}', flush=True)
    if abs(gap) > 0.02:
        raise SystemExit(f'grand mean off by {gap:+.3f}; graph/trace mismatch')
    if len(raw) != len(d):
        raise SystemExit(f'{len(raw)} trace rows vs {len(d)} probit rows')

    rng = np.random.default_rng(0)
    sims = {k: {'slope': [], 'rnp': []} for k in CELLS}
    for j in range(p_risky.shape[1]):
        yj = pd.DataFrame({'chose_risky': (rng.random(len(d)) < p_risky[:, j]
                                           ).astype(float)}, index=d.index)
        r = fit(X, yj)
        if r is None:
            continue
        for k, (ds, dr) in r.items():
            sims[k]['slope'].append(ds)
            sims[k]['rnp'].append(dr)
        if (j + 1) % 25 == 0:
            print(f'  {j + 1}/{p_risky.shape[1]}', flush=True)

    rows = []
    for k in CELLS:
        for i, par in enumerate(['slope', 'rnp']):
            v = np.array(sims[k][par], float)
            v = v[np.isfinite(v)]
            o = obs[k][i]
            rows.append(dict(label=label, parameter=par, order=k[0], stake=k[1],
                             observed=o, model=float(np.median(v)),
                             lo=float(np.percentile(v, 2.5)),
                             hi=float(np.percentile(v, 97.5)),
                             ppp=float(np.mean(v <= o)), n=len(v)))
    out = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    f = Path(out_dir) / f'probit_ppc_ml.{label}.tsv'
    out.to_csv(f, sep='\t', index=False)
    print(out.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print(f'wrote {f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-tmsrisk')
    ap.add_argument('--out_dir', default='probit_ppc_ml')
    ap.add_argument('--n_draws', default=200, type=int)
    a = ap.parse_args()
    main(a.label, a.trace_dir, a.bids_folder, a.out_dir, a.n_draws)
