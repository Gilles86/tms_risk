"""Per-subject cTBS shift in the noise functions, for brain-behaviour correlation.

The group-level curves say cTBS raises shared perceptual noise. The *individualised*
prediction is stronger: subjects whose stimulated voxels lost more nPRF amplitude
should show a bigger noise increase. That needs one number per subject, which is
what this writes.

Subject-level regression coefficients live in `<term>_spline<i>` (dims chain, draw,
subject, regressors), regressors = [Intercept, stimulation]. IPS is the reference
level, so vertex = Intercept + stimulation.

Everything about the basis and the memory/perceptual rotation follows
`noise_curve_inference` -- in particular the knots come from the paradigm columns
(n1 for memory, n2 for perceptual), not from the evaluation grid.

    python -m tms_risk.behavior.scripts.extract_subject_noise_shift \
        --label flexible2_noisefix.head --tag flexible2nf \
        --trace_dir /data/ds-tmsrisk/derivatives/cogmodels.overnight
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from tms_risk.behavior.scripts.noise_curve_inference import (basis, paradigm_columns,
                                                             softplus)


def main(bids_folder, label, spline_order, out_dir, lower, upper, payoffs,
         trace_dir=None, tag=None):
    out_dir = Path(out_dir)
    tdir = Path(trace_dir) if trace_dir else Path(bids_folder) / 'derivatives' / 'cogmodels'
    idata = az.from_netcdf(tdir / f'model-{label}_trace.netcdf')
    post = idata.posterior
    label = tag or label
    family = 2 if 'perceptual_noise_sd_spline1_mu' in post else 1

    subjects = post.coords['subject'].values
    n1, n2 = paradigm_columns(bids_folder)
    xs = np.asarray(payoffs, dtype=float)
    B = {'memory': basis(xs, spline_order, lower, upper, anchor=n1),
         'perceptual': basis(xs, spline_order, lower, upper, anchor=n2)}

    src = ({'perceptual': 'n2_evidence_sd', 'memory': 'n1_evidence_sd'} if family == 1
           else {'perceptual': 'perceptual_noise_sd', 'memory': 'memory_noise_sd'})

    def coefs(term):
        """(sample, subject, regressor, spline) subject-level spline coefficients."""
        c = np.stack([post[f'{term}_spline{i}'].values for i in range(1, spline_order + 1)], -1)
        return c.reshape(-1, *c.shape[-3:])

    eta = {}
    for t, term in src.items():
        c = coefs(term)                                     # sample, subj, reg, spline
        ips, vertex = c[:, :, 0, :], c[:, :, 0, :] + c[:, :, 1, :]
        eta[(t, 'ips')] = ips @ B[t].T                      # sample, subj, payoff
        eta[(t, 'vertex')] = vertex @ B[t].T
    if family == 1:
        for cond in ['ips', 'vertex']:
            eta[('memory', cond)] = eta[('memory', cond)] - eta[('perceptual', cond)]

    rows = []
    for t in ['perceptual', 'memory']:
        d = softplus(eta[(t, 'ips')]) - softplus(eta[(t, 'vertex')])     # sample,subj,payoff
        for j, x in enumerate(xs):
            for i, s in enumerate(subjects):
                rows.append({'subject': s, 'term': t, 'payoff': x,
                             'd_nu': float(d[:, i, j].mean()),
                             'lo': float(np.quantile(d[:, i, j], .025)),
                             'hi': float(np.quantile(d[:, i, j], .975)),
                             'p_pos': float((d[:, i, j] > 0).mean())})
        # localisation: is this subject's effect bigger at the bottom of the range?
        slope = d[:, :, 0] - d[:, :, -1]
        for i, s in enumerate(subjects):
            rows.append({'subject': s, 'term': t, 'payoff': -1,     # -1 = the contrast
                         'd_nu': float(slope[:, i].mean()),
                         'lo': float(np.quantile(slope[:, i], .025)),
                         'hi': float(np.quantile(slope[:, i], .975)),
                         'p_pos': float((slope[:, i] > 0).mean())})

    df = pd.DataFrame(rows)
    out = out_dir / f'subject_noise_shift.{label}.tsv'
    df.to_csv(out, sep='\t', index=False)
    print(f'wrote {out}  ({len(subjects)} subjects, payoffs {list(xs)}; '
          f'payoff=-1 rows are d_nu({xs[0]:.0f}) - d_nu({xs[-1]:.0f}))')
    grp = df[df.payoff == xs[0]].groupby('term').d_nu
    print('\nper-subject cTBS shift at the lowest payoff:')
    print(f'  {"term":12s} {"mean":>8} {"sd":>8} {"n>0":>6}')
    for t, g in grp:
        print(f'  {t:12s} {g.mean():8.3f} {g.std():8.3f} {int((g > 0).sum()):>4}/{len(g)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--label', default='flexible1')
    parser.add_argument('--spline_order', default=5, type=int)
    parser.add_argument('--out_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--lower', default=7.0, type=float)
    parser.add_argument('--upper', default=112.0, type=float)
    parser.add_argument('--payoffs', default='7,10,14,20,28', type=str)
    parser.add_argument('--trace_dir', default=None)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()
    main(args.bids_folder, args.label, args.spline_order, args.out_dir,
         args.lower, args.upper, [float(x) for x in args.payoffs.split(',')],
         args.trace_dir, args.tag)
