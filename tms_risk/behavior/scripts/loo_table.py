"""Table 1: an ELPD/LOO comparison across an arbitrary set of refitted traces.

Differs from `summarize_traces --loo` in one way that matters here: that script keeps
every InferenceData alive so it can hand them all to `az.compare`, which costs ~1.2 GB
per trace and will not fit a twelve-model table next to a running fit. This one loads
one trace at a time, reduces it to its pointwise ELPD (8335 floats), frees it, and
compares the ELPDData objects at the end.

Nothing here recomputes a likelihood, so no bauer version enters the picture: it reads
the `log_likelihood` group each fit stored at sampling time, which is by construction
the one its own posterior was drawn against.

    python -m tms_risk.behavior.scripts.loo_table \\
        --trace_dir /data/ds-tmsrisk/derivatives/cogmodels.overnight \\
        --pattern 'flexible[12]_noisefix*' --out_stem /data/table1_flexible

Writes <out_stem>.tsv (full arviz comparison) and <out_stem>.md (the paper's column
layout: ELPD, effective number of parameters, difference in ELPD, SE, dSE).
"""
import argparse
import gc
import re
from pathlib import Path

import arviz as az
import pandas as pd

# How each label should read in the paper. Matched in order; first hit wins.
NAMES = [
    (r'weber([12])_noisefix$',          'Weber PMC (cTBS on {both})'),
    (r'weber2_noisefix_perception$',    'Weber PMC (cTBS on perceptual noise only)'),
    (r'weber2_noisefix_memory$',        'Weber PMC (cTBS on memory noise only)'),
    (r'weber1_noisefix_first$',         'Weber PMC (cTBS on first-presented option only)'),
    (r'weber1_noisefix_second$',        'Weber PMC (cTBS on second-presented option only)'),
    (r'weber([12])_noisefix_null$',     'Weber PMC null model'),
    (r'flexible([12])(\.\d)?_noisefix$', 'Flexible PMC{df} (cTBS on {both})'),
    (r'flexible2(\.\d)?_noisefix_perception$',
     'Flexible PMC{df} (cTBS on perceptual noise only)'),
    (r'flexible2(\.\d)?_noisefix_memory$',
     'Flexible PMC{df} (cTBS on memory noise only)'),
    (r'flexible1(\.\d)?_noisefix_first$',
     'Flexible PMC{df} (cTBS on first-presented option only)'),
    (r'flexible1(\.\d)?_noisefix_second$',
     'Flexible PMC{df} (cTBS on second-presented option only)'),
    (r'flexible([12])(\.\d)?_noisefix_null$', 'Flexible PMC{df} null model'),
]
BOTH = {1: 'first- and second-option noise', 2: 'perceptual and memory noise'}


def label_of(path):
    """`model-<label>.<variant>_trace.netcdf` -> `<label>`."""
    m = re.fullmatch(r'model-(.+?)(?:\.\w+)?_trace\.netcdf', path.name)
    return m.group(1) if m else path.stem


def pretty(label, family, spline_order, spline_degree):
    for pat, tmpl in NAMES:
        if re.fullmatch(pat, label):
            df = ''
            if spline_order and (spline_order, spline_degree) != (5, 3):
                df = f' [{spline_order} splines, degree {spline_degree}]'
            return tmpl.format(both=BOTH.get(family, 'both noise terms'), df=df)
    return label


def main(trace_dirs, patterns, out_stem, gate):
    paths = sorted({p for d in trace_dirs for pat in patterns
                    for p in Path(d).glob(f'model-{pat}_trace.netcdf')})
    if len(paths) < 2:
        raise SystemExit(f'need >=2 traces; found {len(paths)} in {trace_dirs}')

    loos, meta = {}, {}
    for p in paths:
        label = label_of(p)
        if label in loos:      # same label, different --trace_dir (e.g. objective prior)
            label = f'{label}@{p.parent.name}'
        print(f'reading {p.name} ...', flush=True)
        idata = az.from_netcdf(str(p))
        if 'log_likelihood' not in idata:
            print('  no log_likelihood group -- skipped')
            del idata
            gc.collect()
            continue
        a = idata.posterior.attrs
        summ = az.summary(idata, var_names=[v for v in idata.posterior.data_vars
                                            if v.endswith('_mu')], hdi_prob=.95)
        meta[label] = {
            'label': label,
            'name': pretty(label, int(a.get('tms_risk_family', 0)),
                           int(a.get('tms_risk_spline_order', 0)),
                           int(a.get('tms_risk_spline_degree', 3))),
            'max_rhat': float(summ.r_hat.max()),
            'min_ess': float(summ.ess_bulk.min()),
            'divergences': int(idata.sample_stats['diverging'].sum())
            if 'diverging' in getattr(idata, 'sample_stats', {}) else -1,
            'bauer_commit': a.get('tms_risk_bauer_commit', '?')[:7],
            'prior': a.get('tms_risk_prior_estimate', 'full'),
        }
        meta[label]['converged'] = (meta[label]['max_rhat'] <= gate[0]
                                    and meta[label]['min_ess'] >= gate[1])
        loos[label] = az.loo(idata, pointwise=True)
        del idata
        gc.collect()

    if len(loos) < 2:
        raise SystemExit('fewer than two traces carry a log_likelihood group')

    cmp = az.compare(loos, ic='loo', method='stacking')
    info = pd.DataFrame(meta).T.set_index('label')
    tab = cmp.join(info)
    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(f'{out_stem}.tsv', sep='\t')

    # The paper's column layout. `elpd_diff` is reported relative to the best model,
    # so the sign convention matches Table 1 (negative = worse than reference).
    lines = ['| Model | ELPD (LOO) | Eff. no. params | Diff. in ELPD | SE | dSE | r&#770; | ESS |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for lab, r in tab.iterrows():
        flag = '' if r['converged'] else ' ⚠'
        lines.append(
            f"| {r['name']}{flag} | {r.elpd_loo:.1f} | {r.p_loo:.1f} | "
            f"{-r.elpd_diff:.2f} | {r.se:.1f} | {r.dse:.1f} | "
            f"{r['max_rhat']:.3f} | {r['min_ess']:.0f} |")
    bad = [r['name'] for _, r in tab.iterrows() if not r['converged']]
    if bad:
        lines += ['', f'⚠ failed the convergence gate (r̂ ≤ {gate[0]}, ESS ≥ {gate[1]:.0f}): '
                      + '; '.join(bad) + '.']
    md = '\n'.join(lines)
    Path(f'{out_stem}.md').write_text(md + '\n')
    print('\n' + md)
    print(f'\nwrote {out_stem}.tsv and {out_stem}.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_dir', nargs='+',
                        default=['/data/ds-tmsrisk/derivatives/cogmodels.overnight'])
    parser.add_argument('--pattern', nargs='+', default=['*'],
                        help="glob(s) for the label part, e.g. 'flexible[12]_noisefix*'")
    parser.add_argument('--out_stem', default='/data/table1')
    parser.add_argument('--rhat', default=1.01, type=float)
    parser.add_argument('--ess', default=400., type=float)
    args = parser.parse_args()
    main(args.trace_dir, args.pattern, args.out_stem, (args.rhat, args.ess))
