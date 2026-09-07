"""Figure 3's two quantities as a posterior predictive check: 2 x 2, stake x order.

Published Figure 3B reports the parameters of a psychophysical (probit) model
fitted to the choices: the SLOPE, which is choice consistency and is what
representational noise controls, and the indifference point, reported as the
risk-neutral probability (RNP), which is risk attitude. This script asks whether
the PMC model reproduces both.

The obvious route -- derive slope and RNP in closed form from the PMC parameters
-- is wrong here, and `notes/PLAN.md` records why: nu depends on payoff and the
risky payoff is frac * n_safe, so the model's choice function is NOT a probit in
log(frac) and the closed form has to linearise at each cell's mean payoffs. The
error reaches 3.7 percentage points and 14% of the slope, concentrated exactly
where the paper makes its claim.

So do it the other way round. Simulate choices from the posterior, then fit the
SAME probit, with the SAME gates, to the simulated choices as to the real ones.
Whatever the model's true choice function is, both sides pass through an
identical estimator, and the comparison is valid without any approximation.

    per (subject, order, stake2, stimulation):  probit on log(frac) -> b0, b1
    slope = b1                 RNP = exp(b0 / b1)      (the p that equates EVs)
    then average over subjects, exactly as the observed analysis does

Output: one row per (order, stake2, stimulation) x parameter, with the observed
value and the 2.5/50/97.5 percentiles of the predictive distribution, plus a
posterior predictive p-value for the cTBS effect in each cell.

    python -m tms_risk.behavior.scripts.extract_anchor_probit_ppc log-power-n1n2 \
        --trace_dir .../cogmodels.anchor --n_draws 200
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SLOPE_MIN, SLOPE_MAX, RNP_LO, RNP_HI = 0.5, 20.0, 0.05, 5.0


def probit_newton(x, y, n_iter=60, tol=1e-9):
    """Two-parameter probit by Newton-Raphson. Returns (b0, b1) or (nan, nan).

    statsmodels is ~2 ms a call, and this is called 35 x 8 x n_draws times.
    Newton on a 2-parameter problem with six design points converges in a
    handful of steps and keeps the whole thing to a couple of minutes.
    """
    from scipy.stats import norm
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    X = np.column_stack([np.ones_like(x), x])
    b = np.array([0.0, 1.0])
    for _ in range(n_iter):
        eta = np.clip(X @ b, -8, 8)
        phi, Phi = norm.pdf(eta), np.clip(norm.cdf(eta), 1e-9, 1 - 1e-9)
        w = phi ** 2 / (Phi * (1 - Phi))
        z = (y - Phi) * phi / (Phi * (1 - Phi))
        H = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(H, X.T @ z)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        return np.nan, np.nan
    return float(b[0]), float(b[1])


def cell_stats(x, y):
    """(slope, rnp) for one cell, with the gates the observed analysis uses."""
    b0, b1 = probit_newton(x, y)
    if not np.isfinite([b0, b1]).all() or not (SLOPE_MIN < b1 < SLOPE_MAX):
        return np.nan, np.nan
    # indifference at log frac* = -b0/b1; the risk-neutral probability is the p
    # equating the two EVs there, p n_risky* = n_safe, i.e. 1/frac* = exp(b0/b1)
    rnp = float(np.exp(b0 / b1))
    return b1, (rnp if RNP_LO <= rnp <= RNP_HI else np.nan)


def fit_all(d, ycols, keys):
    """Fit every (cell x column of ycols). Returns (n_cells, n_y, 2) array."""
    groups = list(d.groupby(keys, sort=True))
    out = np.full((len(groups), ycols.shape[1], 2), np.nan)
    for i, (_, g) in enumerate(groups):
        x = g['x'].values
        Y = ycols[g.index.values]
        for j in range(Y.shape[1]):
            out[i, j] = cell_stats(x, Y[:, j])
    return [k for k, _ in groups], out


def main(label, trace_dir, bids_folder, out_dir, n_draws):
    import arviz as az
    import pymc as pm
    from tms_risk.behavior.fit_anchor import build_model
    from tms_risk.behavior.fit_model import get_data

    df = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm')
    idata = az.from_netcdf(Path(trace_dir) / f'model-{label}_trace.netcdf')
    model = build_model(label, df)
    model.build_estimation_model(save_p_choice=True)
    n_chain = idata.posterior.sizes['chain']
    keep = np.linspace(0, idata.posterior.sizes['draw'] - 1,
                       max(1, n_draws // n_chain)).astype(int)
    det = pm.compute_deterministics(idata.posterior.isel(draw=keep),
                                    model=model.estimation_model,
                                    var_names=['p'], merge_dataset=False,
                                    progressbar=False)
    p2 = det['p'].stack(sample=('chain', 'draw')).values
    p2 = p2[:, np.isfinite(p2).all(0)]

    d = df.reset_index().copy()
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['stim'] = d['stimulation_condition']
    d['stake'] = (d['n_safe'] + d['n_risky']) / 2
    d['stake2'] = (d.groupby('subject', group_keys=False)['stake']
                   .apply(lambda v: (v > v.median()).astype(int)))
    d['x'] = np.log(d['frac'])
    p_risky = np.where((~d['risky_first']).values[:, None], p2, 1 - p2)
    gap = float(p_risky.mean() - d['chose_risky'].astype(float).mean())
    print(f'{label}: {p_risky.shape[1]} draws, grand-mean gap {gap:+.4f}')
    if abs(gap) > 0.02:
        raise SystemExit(f'predicted grand mean off by {gap:+.3f}; graph/trace mismatch')

    rng = np.random.default_rng(0)
    sim = (rng.random(p_risky.shape) < p_risky).astype(float)
    obs = d['chose_risky'].astype(float).values[:, None]
    d = d.reset_index(drop=True)

    keys = ['subject', 'order', 'stake2', 'stim']
    print(f'fitting probits: {d.groupby(keys).ngroups} cells x '
          f'{sim.shape[1] + 1} datasets')
    gk, res = fit_all(d, np.hstack([obs, sim]), keys)
    idx = pd.MultiIndex.from_tuples(gk, names=keys)

    rows = []
    for pi, par in enumerate(['slope', 'rnp']):
        v = pd.DataFrame(res[:, :, pi], index=idx)
        # subject-average within each cell, exactly as Figure 3B does
        grp = v.groupby(['order', 'stake2', 'stim']).mean()
        for (order, s2, stim), r in grp.iterrows():
            draws = r.values[1:]
            draws = draws[np.isfinite(draws)]
            rows.append(dict(label=label, parameter=par, order=order,
                             stake2=int(s2), stim=stim, observed=r.values[0],
                             model=np.median(draws),
                             lo=np.percentile(draws, 2.5),
                             hi=np.percentile(draws, 97.5),
                             n_draws=len(draws)))
        # posterior predictive p-value on the cTBS effect within each cell
        d_ips = grp.xs('ips', level='stim')
        d_ver = grp.xs('vertex', level='stim')
        diff = d_ips - d_ver
        for (order, s2), r in diff.iterrows():
            draws = r.values[1:]
            draws = draws[np.isfinite(draws)]
            rows.append(dict(label=label, parameter=f'{par}_ctbs', order=order,
                             stake2=int(s2), stim='ips-vertex',
                             observed=r.values[0], model=np.median(draws),
                             lo=np.percentile(draws, 2.5),
                             hi=np.percentile(draws, 97.5),
                             ppp=float(np.mean(draws <= r.values[0])),
                             n_draws=len(draws)))
    out = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f'probit_ppc.{label}.tsv'
    out.to_csv(f, sep='\t', index=False)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'wrote {f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-tmsrisk')
    ap.add_argument('--out_dir', default='probit_ppc')
    ap.add_argument('--n_draws', default=200, type=int)
    a = ap.parse_args()
    main(a.label, a.trace_dir, a.bids_folder, a.out_dir, a.n_draws)
