"""Supplementary Table 1: model comparison, converged models only.

Replaces the v10 table, which ranked sixteen spline variants including fits
whose posteriors never mixed. Every row here passes the convergence gate
(r_hat <= 1.01 and ESS >= 400 on the group-level parameters); models that fail
it are listed at the bottom as excluded, because an ELPD computed on a posterior
that never mixed is not a number to rank.

ELPD differences are PAIRED and reported against the model the paper reports,
with the standard error of the difference (dSE), which accounts for the
correlation between models evaluated on the same trials.

`checks` counts how many of the seven targeted posterior predictive statistics
fall inside the model's own 95% predictive interval. It is the criterion that
separates the families; ELPD alone mostly reflects the bulk choice curve.

    python -m tms_risk.behavior.scripts.make_supp_table1
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]

#: label -> (noise function, what cTBS is allowed to change)
ROWS = [
    # WHERE the cTBS effect acts (power law throughout)
    ('log-power-n1n2',     'Power', 'Noise on both options'),
    ('log-power-percmem',  'Power', 'Perceptual + memory noise'),
    ('log-power-perc',     'Power', 'Perceptual noise only'),
    ('log-power-n2',       'Power', 'Noise on 2nd-presented option'),
    ('log-power-n1',       'Power', 'Noise on 1st-presented option'),
    ('log-power-mem',      'Power', 'Memory noise only'),
    ('log-power-nullind',  'Power', 'No cTBS effect'),
    # WHAT SHAPE the noise function takes (cTBS on both options throughout)
    ('log-spl3-n1n2',      'Spline, 3 knots', 'Noise on both options'),
    ('log-spl5-n1n2',      'Spline, 5 knots', 'Noise on both options'),
    ('log-genweber-n1n2',  'Generalised Weber', 'Noise on both options'),
    ('log-weber-n1n2',     'Weber, constant ν', 'Noise on both options'),
    ('log-spl3-nullind',   'Spline, 3 knots', 'No cTBS effect'),
    ('log-weber-nullind',  'Weber, constant ν', 'No cTBS effect'),
]


def main(data_dir, reference, out_stem):
    # `resolve` maps a bare model label to the KLW fit of it, and REFUSES to
    # fall back to a raw-rule file -- the two choice rules are not on the same
    # footing and a table mixing them is meaningless. Shared with the ladder so
    # the figure and the table can never disagree about which fit is "the"
    # power/n1n2 model.
    from tms_risk.behavior.scripts.plot_elpd_ladder import resolve
    dd = Path(data_dir)
    _c = dd / 'all_klw_check.tsv'
    chk = pd.read_csv(_c if _c.exists() else dd / 'all_anchor_check.tsv',
                      **READ).set_index('trace')
    ld = dd / 'loo_anchor'
    piw = lambda l: (np.load(ld / f'looi.{l}.npy')
                     if (ld / f'looi.{l}.npy').exists() else None)
    reference = resolve(reference.split('.')[0], ld, chk) or reference
    print(f'reference: {reference}')
    ref = piw(reference)
    out, excluded = [], []
    for base, form, what in ROWS:
        lab = resolve(base, ld, chk)
        if lab is None:
            print(f'  no KLW fit for {base}, skipped')
            continue
        f = ld / f'loo.{lab}.tsv'
        if not f.exists():
            continue
        loo = pd.read_csv(f, **READ).iloc[0]
        row = dict(Model=form, cTBS_affects=what,
                   ELPD=float(loo.elpd_loo),
                   p_loo=float(loo.get('p_loo', np.nan)))
        b = piw(lab)
        if ref is not None and b is not None and b.shape == ref.shape and lab != reference:
            d_ = b - ref
            row['dELPD'] = float(d_.sum())
            row['dSE'] = float(np.std(d_, ddof=1) * np.sqrt(len(d_)))
        else:
            row['dELPD'], row['dSE'] = (0.0, 0.0) if lab == reference else (np.nan, np.nan)
        sf = dd / 'ppc_anchor' / f'ppc_stats.{lab}.tsv'
        if sf.exists():
            st = pd.read_csv(sf, **READ)
            row['checks'] = f'{int(st.covered.sum())}/{len(st)}'
        else:
            row['checks'] = '--'
        if lab in chk.index:
            row['rhat'] = float(chk.loc[lab, 'max_rhat'])
            row['ess'] = int(chk.loc[lab, 'min_ess_bulk'])
            ok = bool(chk.loc[lab, 'ok'])
        else:
            row['rhat'], row['ess'], ok = np.nan, -1, False
        row['label'] = lab
        (out if ok else excluded).append(row)

    T = pd.DataFrame(out).sort_values('ELPD', ascending=False)
    T.to_csv(f'{out_stem}.tsv', sep='\t', index=False)

    def fmt(df, ref_lab):
        lines = ['| Noise function | cTBS affects | ELPD | ΔELPD (dSE) | p_loo | PPC | r̂ | ESS |',
                 '|---|---|---:|---:|---:|:---:|---:|---:|']
        for _, r in df.iterrows():
            d = ('reference' if r.label == ref_lab
                 else ('--' if not np.isfinite(r.dELPD)
                       else f'{r.dELPD:+.1f} ({r.dSE:.1f})'))
            star = ' **' if r.label == ref_lab else ' '
            lines.append(f'|{star}{r.Model}{star.strip()} | {r.cTBS_affects} | '
                         f'{r.ELPD:.1f} | {d} | {r.p_loo:.0f} | {r.checks} | '
                         f'{r.rhat:.3f} | {r.ess} |')
        return '\n'.join(lines)

    md = ['# Supplementary Table 1. Model comparison', '',
          'Expected log predictive density (ELPD, leave-one-out; Vehtari et al., '
          '2017) for every candidate model that met the convergence criterion '
          '(r̂ ≤ 1.01 and effective sample size ≥ 400 on all '
          'group-level parameters). Every model holds the magnitude '
          'priors fixed across stimulation sessions. ΔELPD is the PAIRED '
          'difference against '
          'the model reported in the main text, with the standard error of that '
          'difference (dSE) in brackets; positive values favour the alternative. '
          '`p_loo` is the effective number of parameters. `PPC` is the number of '
          'the seven targeted posterior predictive statistics that fall inside '
          "the model's own 95% predictive interval.", '',
          fmt(T, reference)]
    if excluded:
        md += ['', '**Excluded for non-convergence.** '
               'These models were fitted but did not meet the criterion above, '
               'so their ELPD is not interpretable and they are not ranked: '
               + ', '.join(f'`{r["label"]}` (r̂ = {r["rhat"]:.2f})'
                           for r in excluded) + '.']
    Path(f'{out_stem}.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\nwrote {out_stem}.tsv and {out_stem}.md')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--reference', default='log-power-n1n2')
    ap.add_argument('--out_stem', default=str(REPO / 'notes/supp_table1'))
    a = ap.parse_args()
    main(a.data_dir, a.reference, a.out_stem)
