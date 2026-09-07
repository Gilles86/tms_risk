"""The probit slope and indifference point, derived from the PMC parameters.

Fitting a probit to simulated choices adds sampling noise to a quantity the
model already determines exactly. In log space the choice rule is

    P(choose 2) = Phi( (threshold - diff_mu) / diff_sd ),  threshold = log(p2/p1)
    post_k = w_k log n_k + (1-w_k) mu_k,   w_k = sd_k^2 / (sd_k^2 + nu_k^2)

so writing log n_R = log n_S + log(frac) for the risky and safe options and
collecting terms, the probit index is affine in log(frac):

    index = [ w_R log frac + (w_R - w_S) log n_S
              + (1-w_R) mu_R - (1-w_S) mu_S + log p_R ] / diff_sd

giving closed forms for both published Figure-3 quantities:

    slope       = w_R / diff_sd
    log frac*   = -[ (w_R - w_S) log n_S + (1-w_R) mu_R - (1-w_S) mu_S
                     + log p_R ] / w_R
    rnp         = p_R * exp(log frac*)          (risk-neutral probability)

Two things fall straight out of the algebra, and neither is visible in a
simulated-data probit:

* Under the raw-evidence rule diff_sd = sqrt(nu_R^2 + nu_S^2), so the prior
  enters the SLOPE directly through w_R. Under the KLW-consistent rule
  diff_sd = sqrt((w_R nu_R)^2 + (w_S nu_S)^2) and w largely cancels, leaving the
  slope a function of nu alone -- which is why prior width cannot change
  consistency under the correct rule.
* The indifference point carries a (w_R - w_S) log n_S term, so it MOVES WITH
  STAKE whenever the two options are unequally noisy. The stake dependence is
  not an extra assumption; it is a consequence of noise asymmetry.

nu depends on payoff, so the true psychometric function is not exactly probit;
each cell is evaluated at its own mean payoffs, which is the same approximation
the fitted probit makes.

    python -m tms_risk.behavior.scripts.extract_anchor_probit log-power-n1n2 \\
        --trace_dir .../cogmodels.anchor
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from bauer.models.anchor_noise import NOISE_FORMS, AnchorNoiseMixin

NAME_RE = re.compile(r'^(log|chf)_(perc|mem|n1|n2)_'
                     r'(weber|affine|power|genweber|spl3|spl5|spl7|spl9|cspl3|cspl5|cspl7)_sd(\d*)$')
BY = 'stake2'
SHARED = ('null', 'perc', 'mem', 'percmem', 'spsd', 'spmusd', 'percpsd',
          'percmempsd', 'percx', 'percmemx')


class _Curve(AnchorNoiseMixin):
    def __init__(self, anchors, noise_form):
        self.noise_form = noise_form
        self.n_anchors, self.anchor_link = NOISE_FORMS[noise_form]
        self._anchors = np.asarray(anchors, dtype=float)


def channel_curve(names, chan):
    """A `_Curve` for ONE channel, built from that channel's own parameter names.

    With split forms (`spl5+affine`) the two channels have different forms AND
    different anchor sets, and the trace stamps only the primary form's anchors.
    But every parameter name carries both -- `log_mem_affine_sd7`,
    `log_perc_spl5_sd13` -- so read them off the names rather than the stamp.
    Returns (curve, ordered parameter names), or (None, []) if absent.
    """
    ms = [(n, NAME_RE.match(n)) for n in names]
    hit = [(n, m) for n, m in ms if m and m.group(2) == chan]
    if not hit:
        return None, []
    form = hit[0][1].group(3)
    if NOISE_FORMS[form][0] == 1:                 # weber: one flat value
        anchors, order = [1.0], [hit[0][0]]
    else:
        pairs = sorted((float(m.group(4)), n) for n, m in hit)
        anchors = [a for a, _ in pairs]
        order = [n for _, n in pairs]
    return _Curve(anchors, form), order


def channel(ds, names, chan, cond, x, curve):
    """(subject, sample, len(x)) noise for one channel under one condition."""
    curve, pars = channel_curve(names, chan)
    B = curve.interp_matrix(np.asarray(x, float))
    th = []
    for p in pars:
        rdim = f'{p}_regressors'
        coef = (ds[p].stack(sample=('chain', 'draw'))
                .transpose('subject', 'sample', rdim).values)
        icpt = coef[..., 0]
        th.append(icpt if cond == 'ips'
                  else icpt + (coef[..., 1] if coef.shape[-1] > 1 else 0.0))
    th = np.stack(th, axis=-1)
    return np.exp(th @ B.T) if curve.anchor_link == 'log' else np.exp(th) @ B.T


def prior(ds, space, which, kind, cond):
    """(subject, sample) prior mean or SD for a role under one condition."""
    v = f'{space}_{which}_prior_{kind}'
    rdim = f'{v}_regressors'
    coef = ds[v].stack(sample=('chain', 'draw')).transpose('subject', 'sample',
                                                           rdim).values
    val = coef[..., 0]
    if cond == 'vertex' and coef.shape[-1] > 1:
        val = val + coef[..., 1]
    return np.exp(val) if kind == 'sd' else val


def main(label, trace_dir, bids_folder, out_dir):
    from tms_risk.behavior.fit_model import get_data
    d = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['stake'] = (d['n_safe'] + d['n_risky']) / 2
    d['stake2'] = (d.groupby('subject', group_keys=False)['stake']
                   .apply(lambda v: (v > v.median()).astype(int)))
    # p_R is the RISKY option's win probability, which is the smaller of the
    # two on every trial -- p1 alone gives 1.0 on risky-second trials, and the
    # resulting log p_R = 0 instead of log 0.55 shifted the derived indifference
    # point by a factor of two (rnp came out above 1, which is impossible).
    d['p_risky'] = np.minimum(d['p1'].values, d['p2'].values)
    # The cell key must be the grouping variable itself. Grouping by
    # ('subject','order','stake2') and only *labelling* the result with BY gave
    # 46 near-duplicate 'n_safe' cells at each subject's mean payoff (9.5, 9.6,
    # 10.2, ...) instead of the five design levels 7/10/14/20/28.
    d['cell'] = d[BY]                # a copy, so BY='n_safe' does not collide
    cells = (d.groupby(['subject', 'order', 'cell'])
               .agg(n_safe=('n_safe', 'mean'), n_risky=('n_risky', 'mean'),
                    p_risky=('p_risky', 'min')).reset_index())

    path = Path(trace_dir) / f'model-{label}_trace.netcdf'
    ds = xr.open_dataset(path, group='posterior')
    a = ds.attrs
    space = a['tms_risk_space']
    placement = a['tms_risk_placement']
    consistent = a.get('tms_risk_choice_noise', 'raw_evidence_sd') == 'consistent'
    # only a placeholder now: every channel builds its own curve from its own
    # parameter names, which is the only thing that works when the two channels
    # have different forms ('spl5+affine') and therefore different anchor sets
    curve = _Curve([float(v) for v in a['tms_risk_anchors'].split(',')],
                   a['tms_risk_noise_form'].split('+')[0])
    names = a['tms_risk_parameters'].split(',')
    shared = placement in SHARED
    subjects = list(ds['subject'].values)

    rows, srows = [], []
    for (order, sb), g in cells.groupby(['order', 'cell']):
        g = g.set_index('subject').reindex(subjects)
        nS, nR = g.n_safe.values, g.n_risky.values
        pR = g.p_risky.values
        risky_first = order == 'Risky first'
        # nR / nS are the RISKY and SAFE payoffs; x1 / x2 are the payoffs of
        # whatever sits in position 1 and position 2. Conflating the two was a
        # bug: on risky-second trials it evaluated the risky channel at the safe
        # payoff and vice versa, which put the derived indifference point in the
        # wrong place (rnp > 1) for exactly those cells.
        x1, x2 = (nR, nS) if risky_first else (nS, nR)
        for cond in ('ips', 'vertex'):
            def sig(pos, x):
                if shared:
                    perc = channel(ds, names, 'perc', cond, x, curve)
                    mem = channel(ds, names, 'mem', cond, x, curve)
                    return (perc + mem) if pos == 1 else perc
                return channel(ds, names, f'n{pos}', cond, x, curve)
            # each subject's own cell payoffs: take the diagonal
            s1 = np.stack([sig(1, [x1[i]])[i, :, 0]
                           for i in range(len(subjects))])
            s2 = np.stack([sig(2, [x2[i]])[i, :, 0]
                           for i in range(len(subjects))])
            nu_R, nu_S = (s1, s2) if risky_first else (s2, s1)
            sd_R = prior(ds, space, 'risky', 'sd', cond)
            sd_S = prior(ds, space, 'safe', 'sd', cond)
            mu_R = prior(ds, space, 'risky', 'mu', cond)
            mu_S = prior(ds, space, 'safe', 'mu', cond)
            wR = sd_R ** 2 / (sd_R ** 2 + nu_R ** 2)
            wS = sd_S ** 2 / (sd_S ** 2 + nu_S ** 2)
            diff_sd = (np.sqrt((wR * nu_R) ** 2 + (wS * nu_S) ** 2) if consistent
                       else np.sqrt(nu_R ** 2 + nu_S ** 2))
            slope = wR / diff_sd
            logfrac_star = -((wR - wS) * np.log(nS)[:, None]
                             + (1 - wR) * mu_R - (1 - wS) * mu_S
                             + np.log(pR)[:, None]) / wR
            rnp = pR[:, None] * np.exp(logfrac_star)
            # per-subject posterior summaries, for the subject-wise PPC: does
            # the model predict WHICH people show the biggest effect, not just
            # the group mean?
            for name, arr in [('slope', slope), ('rnp', rnp),
                              ('logfrac_star', logfrac_star)]:
                for i, subj in enumerate(subjects):
                    srows.append(dict(label=label, subject=int(subj),
                                      order=order, cell=float(sb), by=BY,
                                      stimulation_condition=cond,
                                      parameter=name,
                                      median=float(np.median(arr[i])),
                                      lo=float(np.quantile(arr[i], .025)),
                                      hi=float(np.quantile(arr[i], .975))))
            for name, arr in [('slope', slope), ('rnp', rnp),
                              ('logfrac_star', logfrac_star)]:
                m = np.nanmean(arr, axis=0)          # across subjects, per draw
                rows.append(dict(label=label, order=order, cell=float(sb),
                                 by=BY,
                                 stimulation_condition=cond, parameter=name,
                                 mean=float(np.median(m)),
                                 lo=float(np.quantile(m, .025)),
                                 hi=float(np.quantile(m, .975))))
                rows[-1]['draws'] = ','.join(f'{v:.6g}' for v in m[:2000])
    ds.close()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f'probit_derived.{label}.tsv'
    pd.DataFrame(rows).to_csv(f, sep='\t', index=False)
    fs = out_dir / f'probit_subject.{label}.tsv'
    pd.DataFrame(srows).to_csv(fs, sep='\t', index=False)
    print(f'wrote {f} and {fs}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-tmsrisk')
    ap.add_argument('--out_dir', default='probit_derived')
    ap.add_argument('--by', default='stake2', choices=['stake2', 'n_safe'],
                    help='cell definition: stake median split, or safe payoff')
    a = ap.parse_args()
    globals()['BY'] = a.by
    main(a.label, a.trace_dir, a.bids_folder,
         a.out_dir + ('.safe' if a.by == 'n_safe' else ''))
