"""The observer's fitted magnitude priors, next to the payoffs actually shown.

The PMC's whole mechanism is shrinkage of a noisy percept toward a prior, so
whether that prior resembles the distribution the participant was actually
exposed to is a first-order question -- and one nothing in the model forces.

Writes two TSVs: the group-level prior (median and 95% CrI on mu and sd, per
model) and the empirical payoff distribution the participants saw.

Note the SD parameters are sampled on a log scale (`log_safe_prior_sd = -0.71`
means a spread of exp(-0.71) = 0.49 log units), so they are exponentiated here.

    python -m tms_risk.behavior.scripts.extract_anchor_priors \\
        --trace_dir .../cogmodels.anchor --out_dir /home/gdehol/priors
"""
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def label_from_path(path, attrs):
    """Label a trace by its FILENAME, not by its stamp.

    A variant refit (`...pathfinder_trace.netcdf`) carries the base label in
    `tms_risk_label`, so trusting the stamp collapses the variant and the
    original into one row.
    """
    stem = Path(path).name.replace('model-', '').replace('_trace.netcdf', '')
    return stem or attrs.get('tms_risk_label')


def main(trace_dir, out_dir, bids_folder):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(glob(str(Path(trace_dir) / 'model-*_trace.netcdf'))):
        ds = xr.open_dataset(path, group='posterior')
        a = ds.attrs
        label = label_from_path(path, a)
        if label is None:
            ds.close()
            continue
        space = a['tms_risk_space']
        for which in ('risky', 'safe'):
            out = {}
            for kind in ('mu', 'sd'):
                v = f'{space}_{which}_prior_{kind}_mu'      # group mean of the
                if v not in ds:                             # subject-level coef
                    continue
                rdim = f'{space}_{which}_prior_{kind}_regressors'
                x = ds[v].isel({rdim: 0}).stack(s=('chain', 'draw')).values
                # the SD parameter is sampled as log(spread)
                x = np.exp(x) if kind == 'sd' else x
                out[kind] = np.quantile(x, [.025, .5, .975])
            if len(out) == 2:
                rows.append(dict(
                    label=label, space=space, form=a['tms_risk_noise_form'],
                    placement=a['tms_risk_placement'], which=which,
                    mu_lo=out['mu'][0], mu=out['mu'][1], mu_hi=out['mu'][2],
                    sd_lo=out['sd'][0], sd=out['sd'][1], sd_hi=out['sd'][2]))
        ds.close()
    pd.DataFrame(rows).to_csv(out_dir / 'anchor_priors.tsv', sep='\t', index=False)
    print(f'wrote anchor_priors.tsv ({len(rows)} rows)')

    # ...and the same priors split by stimulation condition, for the models that
    # let a prior differ between sessions. Figure 5's panel d needs this and
    # nothing else wrote it, which is why a new label rendered an empty panel.
    # `mu` is reported in CHF here (the pooled table above keeps it in LOG CHF)
    # -- do not log it twice.
    crows = []
    for path in sorted(glob(str(Path(trace_dir) / 'model-*_trace.netcdf'))):
        ds = xr.open_dataset(path, group='posterior')
        a = ds.attrs
        label = label_from_path(path, a)
        if label is None:
            ds.close()
            continue
        space = a['tms_risk_space']
        for which in ('risky', 'safe'):
            for kind in ('mu', 'sd'):
                v = f'{space}_{which}_prior_{kind}_mu'
                if v not in ds:
                    continue
                rdim = f'{space}_{which}_prior_{kind}_regressors'
                nreg = ds.sizes[rdim]
                icpt = ds[v].isel({rdim: 0}).stack(s=('chain', 'draw')).values
                # index 0 is the IPS intercept; index 1 the vertex offset
                off = (ds[v].isel({rdim: 1}).stack(s=('chain', 'draw')).values
                       if nreg > 1 else np.zeros_like(icpt))
                per = {'ips': icpt, 'vertex': icpt + off}
                for cond, x in per.items():
                    y = np.exp(x) if kind == 'sd' else (
                        np.exp(x) if space == 'log' else x)
                    q = np.quantile(y, [.025, .5, .975])
                    crows.append(dict(label=label, which=which, kind=kind,
                                      condition=cond, lo=q[0], mid=q[1],
                                      hi=q[2], varies=int(nreg > 1),
                                      p_gt0=np.nan))
                if nreg > 1:
                    d_ = -off        # IPS - vertex, on the sampled scale
                    q = np.quantile(d_, [.025, .5, .975])
                    crows.append(dict(label=label, which=which, kind=kind,
                                      condition='delta', lo=q[0], mid=q[1],
                                      hi=q[2], varies=1,
                                      p_gt0=float((d_ > 0).mean())))
        ds.close()
    pd.DataFrame(crows).to_csv(out_dir / 'anchor_priors_by_condition.tsv',
                               sep='\t', index=False)
    print(f'wrote anchor_priors_by_condition.tsv ({len(crows)} rows)')

    from tms_risk.behavior.fit_model import get_data
    df = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    pay = pd.concat([df[['subject', 'n_safe']].rename(columns={'n_safe': 'payoff'})
                     .assign(which='safe'),
                     df[['subject', 'n_risky']].rename(columns={'n_risky': 'payoff'})
                     .assign(which='risky')])
    (pay.groupby(['which', 'subject', 'payoff']).size().rename('n').reset_index()
        .to_csv(out_dir / 'anchor_payoffs.tsv', sep='\t', index=False))
    print(f'wrote anchor_payoffs.tsv ({len(pay)} option presentations)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_dir', default='priors')
    ap.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-tmsrisk')
    a = ap.parse_args()
    main(a.trace_dir, a.out_dir, a.bids_folder)
