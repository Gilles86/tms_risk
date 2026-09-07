"""ELPD (PSIS-LOO) for one anchor trace, written as a one-row TSV.

Kept separate from the PPC extraction and run per-label so the ~800 MB
log_likelihood group of a single trace is the only thing in memory at a time.
Also records the fraction of Pareto-k above 0.7, which is what decides whether
the ELPD is trustworthy at all.

    python -m tms_risk.behavior.scripts.extract_anchor_loo log-affine-percmem \\
        --trace_dir .../cogmodels.anchor --out_dir /home/gdehol/loo_anchor
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


def main(label, trace_dir, out_dir):
    path = Path(trace_dir) / f'model-{label}_trace.netcdf'
    idata = az.from_netcdf(path)
    if 'log_likelihood' not in idata.groups():
        raise SystemExit(f'{label}: no log_likelihood group')
    a = idata.posterior.attrs
    loo = az.loo(idata, pointwise=True)
    k = np.asarray(loo.pareto_k)
    row = dict(label=label, space=a['tms_risk_space'], form=a['tms_risk_noise_form'],
               placement=a['tms_risk_placement'],
               memory_model=('independent'
                             if a['tms_risk_placement'] in ('nullind', 'n1', 'n2', 'n1n2')
                             else 'shared_perceptual_noise'),
               n_par=len(a['tms_risk_parameters'].split(',')),
               elpd_loo=float(loo.elpd_loo), se=float(loo.se), p_loo=float(loo.p_loo),
               frac_k_gt_07=float((k > 0.7).mean()), max_k=float(k.max()))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_dir / f'loo.{label}.tsv', sep='\t', index=False)
    # pointwise, for proper pairwise comparison later
    np.save(out_dir / f'looi.{label}.npy', np.asarray(loo.loo_i))
    print(f'{label}: elpd {row["elpd_loo"]:.1f} +/- {row["se"]:.1f}  '
          f'p_loo {row["p_loo"]:.1f}  k>0.7 {row["frac_k_gt_07"]:.3%}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_dir', default='loo_anchor')
    args = ap.parse_args()
    main(args.label, args.trace_dir, args.out_dir)
