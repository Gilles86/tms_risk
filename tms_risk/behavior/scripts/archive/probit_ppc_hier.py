"""Does the PMC model reproduce Figure 3B? The hierarchical probit, run as a PPC.

Figure 3B's slope effect comes from a hierarchical (random-intercept) probit
fitted to the choices. Three other estimators were tried and none of them can
stand in for it:

* a probit fitted to the SUBJECT-AVERAGED curve mixes between-subject spread in
  the indifference point into the slope, and gives −11.1% on the data against
  +3.8% on the model -- both artefacts;
* per-subject ML probits are unbiased but far too noisy at ~60 trials and six
  ladder rungs per subject-cell: +11.0% ± 15.2, median −18.6%, p = 0.82;
* the model's analytic slope w_R / diff_sd is a model quantity with no observed
  counterpart computed the same way.

So run the paper's own estimator on both sides. For each posterior draw of the
PMC model: simulate choices for the trials participants actually saw, fit the
SAME hierarchical probit, and record the group-level cTBS effect on the slope.
The result is a predictive distribution that the observed value either sits
inside or does not.

The design matrix is rebuilt here from `fit_model.get_data`, not loaded from
`fit_probit`, so the simulated choices and the regressors are guaranteed to
refer to the same trials in the same order. `x` is log(risky/safe), matching
`fit_probit`, and `stim_v`/`rf`/`stake_hi` use the coding of
`fit_probit_randomslopes` so the coefficients mean what they mean there.

    python -m tms_risk.behavior.scripts.probit_ppc_hier log-power-n1n2 \
        --trace_dir .../cogmodels.anchor --draw_start 0 --n_sim 5
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

FORMULA = 'y ~ x*rf*stim_v*stake_hi + (1|subject)'
#: slope for a cell, from bambi's treatment coding. stim_v = 1 is VERTEX, so
#: the cTBS effect (IPS - vertex) on the slope is minus the stim_v terms.
CELLS = {('Risky second', 'low'):  ['x:stim_v'],
         ('Risky second', 'high'): ['x:stim_v', 'x:stim_v:stake_hi'],
         ('Risky first', 'low'):   ['x:stim_v', 'x:rf:stim_v'],
         ('Risky first', 'high'):  ['x:stim_v', 'x:rf:stim_v',
                                    'x:stim_v:stake_hi', 'x:rf:stim_v:stake_hi']}


def design(bids_folder):
    from tms_risk.behavior.fit_model import get_data
    d = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    d['x'] = np.log(d['frac'])
    d['rf'] = d['risky_first'].astype(float)
    d['stim_v'] = (d['stimulation_condition'] == 'vertex').astype(float)
    d['average_n'] = (d['n_safe'] + d['n_risky']) / 2.
    d['stake_hi'] = (d.groupby('subject')['average_n']
                     .transform(lambda v: pd.qcut(v, 2, labels=[0, 1],
                                                  duplicates='drop'))
                     .astype(float))
    d['subject'] = d['subject'].astype(str)
    return d.dropna(subset=['x', 'stake_hi']).reset_index(drop=True)


def fit_once(d, y, draws, tune, chains):
    """Fit the hierarchical probit to one choice vector; return cell d-slopes."""
    import bambi
    dd = d.copy()
    dd['y'] = y
    m = bambi.Model(FORMULA, dd, link='probit', family='bernoulli')
    idata = m.fit(draws=draws, tune=tune, chains=chains, cores=chains,
                  target_accept=0.9, random_seed=0, progressbar=False)
    post = idata.posterior
    out = {}
    for (order, stake), terms in CELLS.items():
        v = np.zeros(post.sizes['chain'] * post.sizes['draw'])
        for t in terms:
            v = v + post[t].values.ravel()
        # stim_v == 1 is vertex, so IPS - vertex is the negative of the sum
        out[(order, stake)] = -v
    return out


def main(label, trace_dir, bids_folder, out_dir, draw_start, n_sim,
         draws, tune, chains, observed):
    d = design(bids_folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if observed:
        res = fit_once(d, d['chose_risky'].astype(float).values, draws, tune, chains)
        rows = [dict(label='observed', order=o, stake=s, sim=-1,
                     mean=float(v.mean()),
                     lo=float(np.percentile(v, 2.5)),
                     hi=float(np.percentile(v, 97.5)))
                for (o, s), v in res.items()]
        f = out_dir / 'probit_hier.observed.tsv'
        pd.DataFrame(rows).to_csv(f, sep='\t', index=False)
        print(pd.DataFrame(rows).to_string(index=False,
                                           float_format=lambda v: f'{v:+.3f}'))
        print(f'wrote {f}')
        return

    import arviz as az
    import pymc as pm
    from tms_risk.behavior.fit_anchor import build_model
    from tms_risk.behavior.fit_model import get_data

    df = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm')
    idata = az.from_netcdf(Path(trace_dir) / f'model-{label}_trace.netcdf')
    model = build_model(label, df)
    model.build_estimation_model(save_p_choice=True)
    nc = idata.posterior.sizes['chain']
    keep = np.linspace(0, idata.posterior.sizes['draw'] - 1,
                       max(1, 200 // nc)).astype(int)
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
    # `design` drops rows with a missing bin; align the simulation to what survived
    keepmask = raw.index.isin(d.index) if len(d) == len(raw) else None
    if len(d) != len(raw):
        raise SystemExit(f'design has {len(d)} rows, trace data {len(raw)}')

    rng = np.random.default_rng(1000 + draw_start)
    rows = []
    for k in range(n_sim):
        j = (draw_start + k) % p_risky.shape[1]
        y = (rng.random(p_risky.shape[0]) < p_risky[:, j]).astype(float)
        res = fit_once(d, y, draws, tune, chains)
        for (o, s), v in res.items():
            rows.append(dict(label=label, order=o, stake=s, sim=j,
                             mean=float(v.mean()),
                             lo=float(np.percentile(v, 2.5)),
                             hi=float(np.percentile(v, 97.5))))
        print(f'  sim {k + 1}/{n_sim} (draw {j}) done', flush=True)
    f = out_dir / f'probit_hier.{label}.{draw_start:04d}.tsv'
    pd.DataFrame(rows).to_csv(f, sep='\t', index=False)
    print(f'wrote {f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', default=None)
    ap.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-tmsrisk')
    ap.add_argument('--out_dir', default='probit_hier')
    ap.add_argument('--draw_start', default=0, type=int)
    ap.add_argument('--n_sim', default=5, type=int)
    ap.add_argument('--draws', default=500, type=int)
    ap.add_argument('--tune', default=500, type=int)
    ap.add_argument('--chains', default=2, type=int)
    ap.add_argument('--observed', action='store_true',
                    help='fit the real choices instead, for the comparison value')
    a = ap.parse_args()
    main(a.label, a.trace_dir, a.bids_folder, a.out_dir, a.draw_start, a.n_sim,
         a.draws, a.tune, a.chains, a.observed)
