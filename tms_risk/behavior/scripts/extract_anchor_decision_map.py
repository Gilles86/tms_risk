"""The model evaluated over the whole decision space, for Figure 5.

Figure 5 asks where in the (safe payoff x risky/safe ratio) plane cTBS changes
behaviour, and answers it with maps rather than with the trials that happen to
have been presented. Everything here is closed form -- the same algebra as
`extract_anchor_probit`, evaluated on a grid instead of on the observed cells --
so no PyMC graph is built and no simulation noise enters.

Per (order, safe payoff, ratio), computed per subject x draw and then averaged
across subjects WITHIN draw:

    rel_risky, rel_safe   perceived value of each option under cTBS, % change
    ratio_shift           perceived risky/safe ratio, IPS / vertex
    leverage              dP(risky) / d log(ratio) at vertex -- where choice is
                          actually sensitive to the ratio
    p_vertex, p_ips, dp   choice probability and the cTBS effect on it

    python -m tms_risk.behavior.scripts.extract_anchor_decision_map \\
        log-power-n1n2 --trace_dir .../cogmodels.anchor
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

from bauer.models.anchor_noise import NOISE_FORMS, AnchorNoiseMixin

NAME_RE = re.compile(r'^(log|chf)_(perc|mem|n1|n2)_'
                     r'(weber|affine|power|genweber|spl3|spl5|spl7|spl9|cspl3|cspl5|cspl7)_sd(\d*)$')
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


#: When True, every parameter is read from its GROUP-LEVEL mean instead of the
#: per-subject values -- "the average subject". Not the same object as the
#: average OVER subjects: for a nonlinear model the two differ, and only the
#: average subject satisfies the chain rule that the causal-chain figure draws
#: (leverage x d log ratio reproduces dP to 7%, against a 3.4x failure for the
#: average over subjects). See notes/audit_dp_magnitude.md, audit 2.
AVERAGE_SUBJECT = False


def _coef(ds, v):
    """(subject, sample, regressor), or a single pseudo-subject at the group mean."""
    rdim = f'{v}_regressors'
    if AVERAGE_SUBJECT:
        c = ds[f'{v}_mu'].stack(sample=('chain', 'draw')).transpose(
            'sample', rdim).values
        return c[None, ...]
    return (ds[v].stack(sample=('chain', 'draw'))
            .transpose('subject', 'sample', rdim).values)


def sigma_at(ds, names, chan, cond, x, curve=None):
    """(subject, sample, len(x)) noise for one channel under one condition."""
    curve, pars = channel_curve(names, chan)
    B = curve.interp_matrix(np.asarray(x, float))
    th = []
    for p in pars:
        coef = _coef(ds, p)
        icpt = coef[..., 0]
        th.append(icpt if cond == 'ips'
                  else icpt + (coef[..., 1] if coef.shape[-1] > 1 else 0.0))
    th = np.stack(th, axis=-1)
    return np.exp(th @ B.T) if curve.anchor_link == 'log' else np.exp(th) @ B.T


def prior_par(ds, space, which, kind, cond):
    v = f'{space}_{which}_prior_{kind}'
    coef = _coef(ds, v)
    val = coef[..., 0]
    if cond == 'vertex' and coef.shape[-1] > 1:
        val = val + coef[..., 1]
    return np.exp(val) if kind == 'sd' else val


def main(label, trace_dir, out_dir, n_ratio, thin):
    ds = xr.open_dataset(Path(trace_dir) / f'model-{label}_trace.netcdf',
                         group='posterior')
    a = ds.attrs
    space, placement = a['tms_risk_space'], a['tms_risk_placement']
    consistent = a.get('tms_risk_choice_noise', 'raw_evidence_sd') == 'consistent'
    # only a placeholder now: every channel builds its own curve from its own
    # parameter names, which is the only thing that works when the two channels
    # have different forms ('spl5+affine') and therefore different anchor sets
    curve = _Curve([float(v) for v in a['tms_risk_anchors'].split(',')],
                   a['tms_risk_noise_form'].split('+')[0])
    names = a['tms_risk_parameters'].split(',')
    shared = placement in SHARED
    n_sub = ds.sizes['subject']

    safes = np.array([7., 10., 14., 20., 28.])
    ratios = np.exp(np.linspace(np.log(1.2), np.log(3.5), n_ratio))
    p_risky = 0.55
    rows = []

    for order in ('Risky first', 'Risky second'):
        rf = order == 'Risky first'
        for nS in safes:
            nR = ratios * nS
            # position 1 holds the risky option on risky-first trials
            x1, x2 = (nR, np.full_like(nR, nS)) if rf else (np.full_like(nR, nS), nR)
            cache = {}
            for cond in ('ips', 'vertex'):
                if shared:
                    p1 = sigma_at(ds, names, 'perc', cond, x1, curve)
                    m1 = sigma_at(ds, names, 'mem', cond, x1, curve)
                    s1 = p1 + m1
                    s2 = sigma_at(ds, names, 'perc', cond, x2, curve)
                else:
                    s1 = sigma_at(ds, names, 'n1', cond, x1, curve)
                    s2 = sigma_at(ds, names, 'n2', cond, x2, curve)
                nu_R, nu_S = (s1, s2) if rf else (s2, s1)
                sd_R = prior_par(ds, space, 'risky', 'sd', cond)[..., None]
                sd_S = prior_par(ds, space, 'safe', 'sd', cond)[..., None]
                mu_R = prior_par(ds, space, 'risky', 'mu', cond)[..., None]
                mu_S = prior_par(ds, space, 'safe', 'mu', cond)[..., None]
                wR = sd_R ** 2 / (sd_R ** 2 + nu_R ** 2)
                wS = sd_S ** 2 / (sd_S ** 2 + nu_S ** 2)
                diff_sd = (np.sqrt((wR * nu_R) ** 2 + (wS * nu_S) ** 2)
                           if consistent else np.sqrt(nu_R ** 2 + nu_S ** 2))
                post_R = wR * np.log(nR) + (1 - wR) * mu_R
                post_S = wS * np.log(nS) + (1 - wS) * mu_S
                index = (post_R - post_S + np.log(p_risky)) / diff_sd
                cache[cond] = dict(P=norm.cdf(index), post_R=post_R,
                                   post_S=post_S, index=index, diff_sd=diff_sd,
                                   wR=wR)
            v, i = cache['vertex'], cache['ips']
            # leverage: dP/d log(ratio) at vertex = phi(index) * wR / diff_sd
            lev = norm.pdf(v['index']) * v['wR'] / v['diff_sd']
            q = {'rel_risky': 100 * np.expm1(i['post_R'] - v['post_R']),
                 'rel_safe': 100 * np.expm1(i['post_S'] - v['post_S']),
                 'ratio_shift': np.exp((i['post_R'] - i['post_S'])
                                       - (v['post_R'] - v['post_S'])),
                 'leverage': lev, 'p_vertex': v['P'], 'p_ips': i['P'],
                 'dp': i['P'] - v['P']}
            for name, arr in q.items():
                g = arr.mean(axis=0)[::thin]            # subjects within draw
                lo, mid, hi = np.quantile(g, [.025, .5, .975], axis=0)
                for k, rr in enumerate(ratios):
                    rows.append(dict(label=label, order=order, n_safe=float(nS),
                                     ratio=float(rr), n_risky=float(nR[k]),
                                     quantity=name, lo=lo[k], mid=mid[k],
                                     hi=hi[k]))
    ds.close()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f'decision_map.{label}.tsv'
    pd.DataFrame(rows).to_csv(f, sep='\t', index=False)
    print(f'wrote {f} ({len(rows)} rows, {n_sub} subjects)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_dir', default='decision_map')
    ap.add_argument('--n_ratio', default=25, type=int)
    ap.add_argument('--thin', default=4, type=int)
    ap.add_argument('--average_subject', action='store_true',
                    help='evaluate at the group-level parameters instead of '
                         'averaging per-subject quantities')
    a = ap.parse_args()
    if a.average_subject:
        globals()['AVERAGE_SUBJECT'] = True
    main(a.label, a.trace_dir, a.out_dir + ('.avg' if a.average_subject else ''),
         a.n_ratio, a.thin)
