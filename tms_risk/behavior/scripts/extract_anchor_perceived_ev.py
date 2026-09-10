"""Perceived expected value against objective expected value, per option.

The reviewer's question in one picture: the observer never acts on the payoff it
was shown, but on a percept shrunk toward a prior. Plotting the perceived EV
against the objective EV puts the whole distortion on one pair of axes -- the
identity line is a veridical observer, and everything the model does is the
departure from it.

    perceived EV = p * exp(w * log n + (1 - w) * mu),   w = sd_p^2/(sd_p^2+nu^2)
    objective EV = p * n

Evaluated for both options, both presentation orders and both stimulation
conditions, since the noise depends on which POSITION an option occupied and the
prior on which ROLE it played.

    python -m tms_risk.behavior.scripts.extract_anchor_perceived_ev \\
        log-power-n1n2 --trace_dir .../cogmodels.anchor
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from bauer.models.anchor_noise import NOISE_FORMS, AnchorNoiseMixin

NAME_RE = re.compile(r'^(log|chf)_(perc|mem|n1|n2)_'
                     r'(' + '|'.join(sorted(NOISE_FORMS, key=len,
                                              reverse=True))
                     + r')_sd(\d*)$')
SHARED = ('null', 'perc', 'mem', 'percmem', 'spsd', 'spmusd', 'percpsd',
          'percmempsd', 'percx', 'percmemx')
P_RISKY = 0.55


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


def main(label, trace_dir, out_dir, n_grid, thin):
    ds = xr.open_dataset(Path(trace_dir) / f'model-{label}_trace.netcdf',
                         group='posterior')
    a = ds.attrs
    space, placement = a['tms_risk_space'], a['tms_risk_placement']
    # only a placeholder now: every channel builds its own curve from its own
    # parameter names, which is the only thing that works when the two channels
    # have different forms ('spl5+affine') and therefore different anchor sets
    curve = _Curve([float(v) for v in a['tms_risk_anchors'].split(',')],
                   a['tms_risk_noise_form'].split('+')[0])
    names = a['tms_risk_parameters'].split(',')
    shared = placement in SHARED

    # each option is only ever shown over its own payoff range
    ranges = {'risky': np.exp(np.linspace(np.log(7), np.log(112), n_grid)),
              'safe': np.exp(np.linspace(np.log(7), np.log(28), n_grid))}
    rows = []
    for order in ('Risky first', 'Risky second'):
        rf = order == 'Risky first'
        for role in ('risky', 'safe'):
            n = ranges[role]
            # which position does this role occupy in this order?
            pos = 1 if (role == 'risky') == rf else 2
            p_k = P_RISKY if role == 'risky' else 1.0
            for cond in ('ips', 'vertex'):
                if shared:
                    sig = sigma_at(ds, names, 'perc', cond, n, curve)
                    if pos == 1:                       # first option is recalled
                        sig = sig + sigma_at(ds, names, 'mem', cond, n, curve)
                else:
                    sig = sigma_at(ds, names, f'n{pos}', cond, n, curve)
                sd = prior_par(ds, space, role, 'sd', cond)[..., None]
                mu = prior_par(ds, space, role, 'mu', cond)[..., None]
                w = sd ** 2 / (sd ** 2 + sig ** 2)
                perceived = p_k * np.exp(w * np.log(n) + (1 - w) * mu)
                g = perceived.mean(axis=0)[::thin]
                lo, mid, hi = np.quantile(g, [.025, .5, .975], axis=0)
                for k, nn in enumerate(n):
                    rows.append(dict(label=label, order=order, role=role,
                                     position=pos, stimulation_condition=cond,
                                     payoff=float(nn),
                                     objective_ev=float(p_k * nn),
                                     lo=lo[k], perceived_ev=mid[k], hi=hi[k]))
    ds.close()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f'perceived_ev.{label}.tsv'
    pd.DataFrame(rows).to_csv(f, sep='\t', index=False)
    print(f'wrote {f} ({len(rows)} rows)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_dir', default='perceived_ev')
    ap.add_argument('--n_grid', default=40, type=int)
    ap.add_argument('--thin', default=4, type=int)
    a = ap.parse_args()
    main(a.label, a.trace_dir, a.out_dir, a.n_grid, a.thin)
