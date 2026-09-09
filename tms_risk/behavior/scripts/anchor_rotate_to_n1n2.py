"""Rotate a `sum_difference` trace back into n1/n2 coordinates.

`sum_difference` fits theta_total and theta_split instead of theta_n1 and
theta_n2, where

    theta_n1 = theta_total + theta_split / 2
    theta_n2 = theta_total - theta_split / 2

It is a REPARAMETERISATION, not a different model: same likelihood, same
degrees of freedom, chosen because the (level, ratio) coordinates cut the ridge
that stops the n1/n2 coordinates sampling (group means correlate at r = 0.97 as
(n1, n2), -0.39 to -0.64 as (level, ratio); condition number 231).

So the n1/n2 answer is already in the trace and needs no refit. The transform is
LINEAR, so it passes through the design matrix untouched -- the cTBS
coefficients rotate exactly as the intercepts do -- and applying it PER DRAW
gives the exact posterior over n1 and n2 rather than a summary of a summary.

This writes a new trace whose `split`/`total` variables are replaced by `n1`/`n2`
ones, with `tms_risk_parameters` and the placement attribute rewritten to match.
Every downstream extraction script then works on it untouched, with no special
case for the rotated family anywhere.

    python -m tms_risk.behavior.scripts.anchor_rotate_to_n1n2 \\
        --trace_dir <bids>/derivatives/cogmodels.anchor \\
        --label log-power-sd.mapjitter.klw

writes `model-log-power-sd.mapjitter.klw.asn1n2_trace.netcdf` alongside it.
"""
import argparse
import re
from pathlib import Path

SPLIT, TOTAL = 'split', 'total'


def rotated_name(name):
    """`log_split_power_sd7` -> `log_n1_power_sd7` (and total -> n2 slot)."""
    return (re.sub(rf'_{SPLIT}_', '_n1_', name),
            re.sub(rf'_{SPLIT}_', '_n2_', name))


def main(trace_dir, label, out_label, overwrite):
    import arviz as az
    import xarray as xr

    src = Path(trace_dir) / f'model-{label}_trace.netcdf'
    dst = Path(trace_dir) / f'model-{out_label}_trace.netcdf'
    if dst.exists() and not overwrite:
        raise SystemExit(f'{dst} exists; pass --overwrite to replace it')

    idata = az.from_netcdf(str(src))
    post = idata.posterior
    splits = [v for v in post.data_vars if f'_{SPLIT}_' in v]
    if not splits:
        raise SystemExit(
            f'{label} has no `{SPLIT}` variables -- it is not a sum_difference '
            f'fit. Variables: {sorted(post.data_vars)}')

    new = {}
    for sname in splits:
        tname = sname.replace(f'_{SPLIT}_', f'_{TOTAL}_')
        if tname not in post:
            raise SystemExit(f'{sname} has no matching {tname}')
        sp, to = post[sname], post[tname]
        # the two channels share a design matrix, so the regressor axes match
        # elementwise and the sum is well defined; rename the axis to the new
        # variable's own so nothing downstream reads a stale coord name
        n1_name, n2_name = rotated_name(sname)
        for out_name, expr in ((n1_name, to + .5 * sp),
                               (n2_name, to - .5 * sp)):
            d = expr.rename({f'{sname}_regressors': f'{out_name}_regressors'}) \
                if f'{sname}_regressors' in expr.dims else expr
            new[out_name] = d
        print(f'  {tname} +/- 0.5*{sname}  ->  {n1_name}, {n2_name}')

    post = post.drop_vars(splits + [s.replace(f'_{SPLIT}_', f'_{TOTAL}_')
                                    for s in splits])
    drop_dims = [d for d in post.dims if f'_{SPLIT}_' in d or f'_{TOTAL}_' in d]
    post = post.drop_dims(drop_dims, errors='ignore')
    post = post.assign(new)

    pars = [p for p in post.attrs.get('tms_risk_parameters', '').split(',')
            if p and SPLIT not in p and TOTAL not in p]
    post.attrs['tms_risk_parameters'] = ','.join(
        pars + ['n1_evidence_sd', 'n2_evidence_sd'])
    post.attrs['tms_risk_memory_model'] = 'independent'
    post.attrs['tms_risk_rotated_from'] = label
    idata.posterior = post
    idata.to_netcdf(str(dst))
    print(f'\nwrote {dst}')
    print('  This is the SAME posterior in n1/n2 coordinates -- not a refit. '
          'Every extraction script now works on it with no special case.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out_label', default=None,
                    help='default: <label>.asn1n2')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    main(a.trace_dir, a.label, a.out_label or f'{a.label}.asn1n2', a.overwrite)
