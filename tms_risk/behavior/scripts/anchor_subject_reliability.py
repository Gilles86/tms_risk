"""How much of the between-participant spread in a cTBS contrast is real?

A per-participant estimate from a hierarchical cognitive model carries its own
posterior uncertainty. If that uncertainty is large relative to the spread of
the point estimates across participants, most of the apparent individual
differences are measurement error, and no correlation with an external measure
can exceed the attenuation ceiling however real the underlying association is.

Estimator, matching `probit_subject_reliability.py` so the two are comparable:

    var_between_obs = Var_s( median_d Δ_s )        spread of the point estimates
    var_within      = median_s Var_d( Δ_s )        posterior variance per subject
    reliability     = (var_between_obs − var_within) / var_between_obs
    ceiling         = sqrt(reliability)

`var_between_obs` is inflated by measurement error, so subtracting the mean
within-participant variance is the standard correction. The ceiling is the
largest correlation this measure could show with a perfectly measured external
variable. Reported alongside the cruder 1/sqrt(1 + within/between) form, which
assumes the observed spread is all signal and is therefore optimistic.

    python -m tms_risk.behavior.scripts.anchor_subject_reliability \\
        --model_label log-power-n1n2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]


def main(data_dir, label, out_tsv):
    f = Path(data_dir) / 'subject_params' / f'subject_params.{label}.tsv'
    d = pd.read_csv(f, **READ)
    rows = []
    for par, g in d[d.subject != 'GROUP'].groupby('parameter'):
        mid = g['mid'].values
        # the per-subject 95% CrI is 2*1.96 posterior SDs wide
        sd_within = (g['hi'].values - g['lo'].values) / (2 * 1.96)
        var_between_obs = float(np.var(mid, ddof=1))
        var_within = float(np.median(sd_within ** 2))
        var_between_true = max(var_between_obs - var_within, 0.0)
        rel = var_between_true / var_between_obs if var_between_obs > 0 else 0.0
        rows.append(dict(
            label=label, parameter=par, n=len(g),
            sd_within=float(np.sqrt(var_within)),
            sd_between_observed=float(np.sqrt(var_between_obs)),
            sd_between_corrected=float(np.sqrt(var_between_true)),
            reliability=rel,
            ceiling_corrected=float(np.sqrt(rel)),
            ceiling_naive=float(1 / np.sqrt(1 + var_within / var_between_obs))))
    out = pd.DataFrame(rows).sort_values('reliability', ascending=False)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep='\t', index=False)
    pd.set_option('display.width', 200)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nwrote {out_tsv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--out_tsv', default=None)
    a = ap.parse_args()
    main(a.data_dir, a.model_label,
         a.out_tsv or str(REPO / f'notes/data/subject_reliability.{a.model_label}.tsv'))
