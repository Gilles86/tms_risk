"""Reproduce the manuscript's by-stake probit numbers from the stored trace.

The v9 paragraph ("... median split on average stake size ...") and Supplementary
Figure 1 come from `model-probit_average_n_full_trace.netcdf` (fit 2024-11,
`fit_probit.py`: `chose_risky ~ x*risky_first*stimulation_condition*C(average_n_bin)
+ (1|subject)`, probit link). The scratchpad scripts that first verified these numbers
(`check1_interaction_triple.py`, see notes/checks_20260803.md) are gone; this script is
the permanent replacement, so the paragraph stays reproducible from disk.

It rebuilds, per posterior draw, the slope on x for every (order, stake bin,
stimulation) cell by summing the named fixed-effect terms (reference levels: IPS,
low stake, risky_first = False i.e. SAFE FIRST), and prints:
  - the eight cell slopes with 95% CrI,
  - per-cell IPS-vertex p (P of the delta crossing 0),
  - the named triple-interaction coefficient x:stimulation_condition:C(average_n_bin)
    -- the manuscript's p = 0.0153 (this coefficient IS the stake x stimulation
    interaction *within safe-first trials*, because risky_first = 0 there),
  - the same interaction within risky-first trials and averaged over orders (the
    alternative readings; the pooled one is the notebook contrast that gives ~0.053).

    python -m tms_risk.behavior.scripts.reproduce_stake_probit
"""
import argparse

import arviz as az
import numpy as np
import pandas as pd


def main(trace_path, out_tsv):
    post = az.from_netcdf(trace_path).posterior.stack(sample=('chain', 'draw'))

    def term(name, level=None):
        v = post[name]
        if level is not None:
            dim = [d for d in v.dims if d != 'sample'][0]
            v = v.sel({dim: level})
        return v.values

    x = term('x')
    x_rf = term('x:risky_first')
    x_v = term('x:stimulation_condition', 'vertex')
    x_h = term('x:C(average_n_bin)', 'high')
    x_rf_v = term('x:risky_first:stimulation_condition', 'vertex')
    x_rf_h = term('x:risky_first:C(average_n_bin)', 'high')
    x_v_h = term('x:stimulation_condition:C(average_n_bin)', 'vertex, high')
    x_rf_v_h = term('x:risky_first:stimulation_condition:C(average_n_bin)',
                    'vertex, high')

    def slope(rf, vertex, high):
        return (x + rf * x_rf + vertex * x_v + high * x_h
                + rf * vertex * x_rf_v + rf * high * x_rf_h
                + vertex * high * x_v_h + rf * vertex * high * x_rf_v_h)

    rows = []
    print('Cell slopes (posterior mean [95% CrI]); order names follow the repo '
          'convention risky_first=True -> "Risky first":')
    for rf, order in [(0, 'Risky second (safe first)'), (1, 'Risky first')]:
        for high, stake in [(0, 'low'), (1, 'high')]:
            sv, si = slope(rf, 1, high), slope(rf, 0, high)
            d = si - sv
            p_less = (d < 0).mean()
            print(f'  {order:26s} {stake:4s} stake: '
                  f'vertex {sv.mean():.3f} [{np.quantile(sv, .025):.2f}, '
                  f'{np.quantile(sv, .975):.2f}] -> ips {si.mean():.3f} '
                  f'[{np.quantile(si, .025):.2f}, {np.quantile(si, .975):.2f}]   '
                  f'P(ips<vertex) = {p_less:.4f}')
            rows.append(dict(order=order, stake=stake,
                             vertex_mean=sv.mean(), ips_mean=si.mean(),
                             vertex_lo=np.quantile(sv, .025),
                             vertex_hi=np.quantile(sv, .975),
                             ips_lo=np.quantile(si, .025),
                             ips_hi=np.quantile(si, .975),
                             p_ips_less=p_less))

    print('\nStake x stimulation interaction on the slope, three readings:')
    named = x_v_h
    print(f'  named coefficient x:stim:C(bin) [vertex, high] '
          f'(= interaction WITHIN safe-first trials): '
          f'{named.mean():.4f} [{np.quantile(named, .025):.4f}, '
          f'{np.quantile(named, .975):.4f}], P(>0) = {(named > 0).mean():.4f}'
          f'   <- the manuscript\'s 0.0153')
    within_rf1 = x_v_h + x_rf_v_h
    print(f'  within risky-first trials (+4-way term): '
          f'{within_rf1.mean():.4f}, P(>0) = {(within_rf1 > 0).mean():.4f}')
    pooled = x_v_h + 0.5 * x_rf_v_h
    print(f'  averaged over both orders: {pooled.mean():.4f}, '
          f'P(>0) = {(pooled > 0).mean():.4f}'
          f'   <- the notebook cell-contrast (~0.053)')

    pd.DataFrame(rows).to_csv(out_tsv, sep='\t', index=False)
    print(f'\nwrote {out_tsv}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trace', default='/data/ds-tmsrisk/derivatives/cogmodels/'
                   'model-probit_average_n_full_trace.netcdf')
    p.add_argument('--out_tsv', default='notes/data/probit_stake_cells_published.tsv')
    a = p.parse_args()
    main(a.trace, a.out_tsv)
