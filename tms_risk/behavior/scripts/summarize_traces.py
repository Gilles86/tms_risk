"""Convergence + model-comparison summary over a directory of PMC traces.

One row per trace: sampler geometry, the convergence gate (max r_hat / min ESS /
divergences), and the provenance stamps written by `fit_pmc_noisefix`
(bauer commit, family, whether priors were constrained). With `--loo` it also
runs an ArviZ LOO comparison over every trace that carries a `log_likelihood`
group -- which is how the nested flexible1 variants (full / first-only /
second-only / null) are compared.

    python -m tms_risk.behavior.scripts.summarize_traces \\
        --trace_dir /data/ds-tmsrisk/derivatives/cogmodels.overnight --loo
"""
import argparse
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# skip the per-subject offsets: 35 subjects x ~20 parameters swamps the gate with
# parameters nobody reads, and their ESS is bounded by the group-level ones anyway
SUBJECT_SUFFIX = ('_offset', '_subjectwise', '_sd_untransformed')


def group_vars(post):
    return [v for v in post.data_vars
            if 'subject' not in post[v].dims and not v.endswith(SUBJECT_SUFFIX)]


def summarize_one(path, all_params=False):
    idata = az.from_netcdf(str(path))
    post = idata.posterior
    vs = list(post.data_vars) if all_params else group_vars(post)
    s = az.summary(post, var_names=vs, kind='diagnostics')
    row = {
        'trace': path.name.replace('model-', '').replace('_trace.netcdf', ''),
        'chains': post.sizes['chain'], 'draws': post.sizes['draw'],
        'n_par': len(vs),
        'max_rhat': float(s['r_hat'].max()),
        'min_ess_bulk': float(s['ess_bulk'].min()),
        'min_ess_tail': float(s['ess_tail'].min()),
        'worst_par': str(s['r_hat'].idxmax()),
        'has_loglik': 'log_likelihood' in idata.groups(),
        'size_gb': path.stat().st_size / 2**30,
    }
    if 'sample_stats' in idata.groups() and 'diverging' in idata.sample_stats:
        row['divergences'] = int(idata.sample_stats['diverging'].values.sum())
    for k in ['tms_risk_bauer_commit', 'tms_risk_family', 'tms_risk_constrained']:
        row[k.replace('tms_risk_', '')] = post.attrs.get(k, '-')
    row['ok'] = bool(row['max_rhat'] <= 1.01 and row['min_ess_bulk'] >= 400)
    return row, idata


def main(trace_dirs, pattern, do_loo, out_tsv, all_params):
    paths = sorted({p for d in trace_dirs
                    for p in Path(d).glob(f'model-{pattern}_trace.netcdf')})
    if not paths:
        raise SystemExit(f'no traces matching model-{pattern}_trace.netcdf in {trace_dirs}')

    rows, loadables = [], {}
    for p in paths:
        print(f'reading {p} ...', flush=True)
        try:
            row, idata = summarize_one(p, all_params)
        except Exception as e:                                  # noqa: BLE001
            print(f'  FAILED: {type(e).__name__}: {e}')
            rows.append({'trace': p.name, 'ok': False, 'worst_par': f'{type(e).__name__}'})
            continue
        rows.append(row)
        if do_loo and row['has_loglik']:
            loadables[row['trace']] = idata
        else:
            del idata

    tab = pd.DataFrame(rows)
    cols = ['trace', 'ok', 'max_rhat', 'min_ess_bulk', 'divergences', 'chains',
            'draws', 'n_par', 'worst_par', 'family', 'constrained',
            'bauer_commit', 'size_gb']
    tab = tab.reindex(columns=[c for c in cols if c in tab])
    print('\n=== convergence (gate: r_hat <= 1.01 and ess_bulk >= 400) ===')
    with pd.option_context('display.width', 200, 'display.max_colwidth', 30):
        print(tab.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    bad = tab[~tab.ok.astype(bool)]
    print(f'\n{len(tab) - len(bad)}/{len(tab)} traces pass.'
          + (f'  FAILING: {", ".join(bad.trace)}' if len(bad) else ''))

    if do_loo and len(loadables) > 1:
        print('\n=== LOO comparison ===')
        cmp = az.compare(loadables, ic='loo', method='stacking')
        print(cmp.to_string())
        if out_tsv:
            cmp.to_csv(Path(out_tsv).with_suffix('.loo.tsv'), sep='\t')
    elif do_loo:
        print('\n(LOO skipped: fewer than two traces carry a log_likelihood group)')

    if out_tsv:
        tab.to_csv(out_tsv, sep='\t', index=False)
        print(f'\nwrote {out_tsv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_dir', nargs='+',
                        default=['/data/ds-tmsrisk/derivatives/cogmodels.overnight'])
    parser.add_argument('--pattern', default='*',
                        help='glob for the label part, e.g. "flexible1_noisefix*"')
    parser.add_argument('--loo', action='store_true')
    parser.add_argument('--out_tsv', default=None)
    parser.add_argument('--all_params', action='store_true',
                        help='include per-subject parameters in the gate')
    args = parser.parse_args()
    main(args.trace_dir, args.pattern, args.loo, args.out_tsv, args.all_params)
