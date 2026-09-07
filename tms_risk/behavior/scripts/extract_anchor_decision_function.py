"""The model's decision function itself, and the two channels cTBS moves.

In this observer the choice index is a ratio,

    index = (post_R - post_S + log p_R) / diff_sd

and cTBS moves BOTH parts of it. Raising nu shifts the numerator (percepts are
pulled further toward the prior) and simultaneously inflates the denominator
(the decision function flattens). On choice probability those two work against
each other, which a plot of the noise function or of the perceived ratio alone
cannot show.

So evaluate the two counterfactuals explicitly, on the same trials:

    P_vertex = Phi(num_v / den_v)      P_full  = Phi(num_i / den_i)
    P_bias   = Phi(num_i / den_v)      P_scale = Phi(num_v / den_i)

`P_bias - P_vertex` is what cTBS would do if it only moved the percepts;
`P_scale - P_vertex` is what it would do if it only flattened the decision
function. Their sum is the full effect up to an interaction term, which is
reported rather than assumed away.

Writes a grid over (safe payoff x risky/safe ratio) for both presentation
orders, everything closed form from the posterior.

    python -m tms_risk.behavior.scripts.extract_anchor_decision_function \\
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


def sigma_at(ds, names, chan, cond, x, curve=None):
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


def prior_par(ds, space, which, kind, cond):
    v = f'{space}_{which}_prior_{kind}'
    coef = (ds[v].stack(sample=('chain', 'draw'))
            .transpose('subject', 'sample', f'{v}_regressors').values)
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

    safes = np.array([7., 10., 14., 20., 28.])
    ratios = np.exp(np.linspace(np.log(1.2), np.log(3.5), n_ratio))
    p_risky = 0.55
    rows = []

    for order in ('Risky first', 'Risky second'):
        rf = order == 'Risky first'
        for nS in safes:
            nR = ratios * nS
            x1, x2 = (nR, np.full_like(nR, nS)) if rf else (np.full_like(nR, nS), nR)
            num, den = {}, {}
            for cond in ('ips', 'vertex'):
                if shared:
                    s1 = (sigma_at(ds, names, 'perc', cond, x1, curve)
                          + sigma_at(ds, names, 'mem', cond, x1, curve))
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
                den[cond] = (np.sqrt((wR * nu_R) ** 2 + (wS * nu_S) ** 2)
                             if consistent else np.sqrt(nu_R ** 2 + nu_S ** 2))
                num[cond] = (wR * np.log(nR) + (1 - wR) * mu_R
                             - wS * np.log(nS) - (1 - wS) * mu_S
                             + np.log(p_risky))
            P = {'p_vertex': norm.cdf(num['vertex'] / den['vertex']),
                 'p_full': norm.cdf(num['ips'] / den['ips']),
                 'p_bias_only': norm.cdf(num['ips'] / den['vertex']),
                 'p_scale_only': norm.cdf(num['vertex'] / den['ips'])}
            P['dp_full'] = P['p_full'] - P['p_vertex']
            P['dp_bias'] = P['p_bias_only'] - P['p_vertex']
            P['dp_scale'] = P['p_scale_only'] - P['p_vertex']
            P['dp_interaction'] = P['dp_full'] - P['dp_bias'] - P['dp_scale']
            P['num_shift'] = num['ips'] - num['vertex']
            P['den_ratio'] = den['ips'] / den['vertex']
            for name, arr in P.items():
                g = arr.mean(axis=0)[::thin]
                lo, mid, hi = np.quantile(g, [.025, .5, .975], axis=0)
                for k, rr in enumerate(ratios):
                    rows.append(dict(label=label, order=order, n_safe=float(nS),
                                     ratio=float(rr), quantity=name,
                                     lo=lo[k], mid=mid[k], hi=hi[k]))
    ds.close()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f'decision_function.{label}.tsv'
    pd.DataFrame(rows).to_csv(f, sep='\t', index=False)
    print(f'wrote {f} ({len(rows)} rows)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_dir', default='decision_function')
    ap.add_argument('--n_ratio', default=40, type=int)
    ap.add_argument('--thin', default=4, type=int)
    a = ap.parse_args()
    main(a.label, a.trace_dir, a.out_dir, a.n_ratio, a.thin)
