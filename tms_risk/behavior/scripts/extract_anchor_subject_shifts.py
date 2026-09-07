"""Per-subject cTBS shift in sigma_n1 and sigma_n2 at chosen payoffs.

The group-level curves say n2 moves and n1 does not, but a hierarchical model
shrinks; this asks the same question one participant at a time. Computed per
draw from the subject-level coefficients -- sigma(x) = B(x) @ exp(theta) for the
value link, exp(B(x) @ theta) for the log link -- so no PyMC graph is needed and
the credible interval is on the shift itself, not on a ratio of two summaries.

    python -m tms_risk.behavior.scripts.extract_anchor_subject_shifts \\
        log-spl3-n1n2 --trace_dir .../cogmodels.anchor --payoffs 7 28
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
COMPOSE = {'shared_perceptual_noise': {'n1': ('perc', 'mem'), 'n2': ('perc',)},
           'independent': {'n1': ('n1',), 'n2': ('n2',)}}
SHARED = ('null', 'perc', 'mem', 'percmem')


class _Curve(AnchorNoiseMixin):
    def __init__(self, anchors, noise_form):
        self.noise_form = noise_form
        self.n_anchors, self.anchor_link = NOISE_FORMS[noise_form]
        self._anchors = np.asarray(anchors, dtype=float)


def channel_curve(names, chan):
    """A `_Curve` for ONE channel, built from that channel's own parameter names.

    With split forms ('weber+affine') the two channels have different forms AND
    different anchor sets, while the trace stamps only the primary form's
    anchors. Every parameter name carries both -- `log_n1_weber_sd`,
    `log_n2_affine_sd112` -- so read them off the names.
    """
    ms = [(n, NAME_RE.match(n)) for n in names]
    hit = [(n, m) for n, m in ms if m and m.group(2) == chan]
    if not hit:
        return None, []
    form = hit[0][1].group(3)
    if NOISE_FORMS[form][0] == 1:
        return _Curve([1.0], form), [hit[0][0]]
    pairs = sorted((float(m.group(4)), n) for n, m in hit)
    return _Curve([a for a, _ in pairs], form), [n for _, n in pairs]


def main(label, trace_dir, out_dir, payoffs):
    ds = xr.open_dataset(Path(trace_dir) / f'model-{label}_trace.netcdf',
                         group='posterior')
    a = ds.attrs
    form, placement = a['tms_risk_noise_form'], a['tms_risk_placement']
    anchors = [float(v) for v in a['tms_risk_anchors'].split(',')]
    names = a['tms_risk_parameters'].split(',')
    memory = ('shared_perceptual_noise' if placement in SHARED else 'independent')
    curve = _Curve(anchors, form.split('+')[0])
    x = np.asarray(payoffs, float)
    B = curve.interp_matrix(x)                                   # (n_x, K)
    subjects = ds['subject'].values

    def channel_sigma(chan, cond):
        """(n_subject, n_sample, n_x) noise for one channel under one condition.

        The curve is built from THIS channel's own parameter names, so a split
        form ('weber+affine') gets the right form and anchors on each channel.
        """
        ccur, pars = channel_curve(names, chan)
        B = ccur.interp_matrix(x)                                # (n_x, K)
        th = []
        for p in pars:
            rdim = f'{p}_regressors'
            coef = (ds[p].stack(sample=('chain', 'draw'))
                    .transpose('subject', 'sample', rdim).values)
            icpt = coef[..., 0]                                  # Intercept = IPS
            th.append(icpt if cond == 'ips'
                      else icpt + (coef[..., 1] if coef.shape[-1] > 1 else 0.0))
        th = np.stack(th, axis=-1)                               # (subj, samp, K)
        if ccur.anchor_link in ('log', 'logcubic'):
            return np.exp(th @ B.T)
        return np.exp(th) @ B.T

    sig = {}
    free = sorted({NAME_RE.match(n).group(2) for n in names if NAME_RE.match(n)})
    for chan in free:
        for cond in ('ips', 'vertex'):
            sig[(chan, cond)] = channel_sigma(chan, cond)
    for comp, parts in COMPOSE[memory].items():
        if comp not in free:
            for cond in ('ips', 'vertex'):
                sig[(comp, cond)] = sum(sig[(p, cond)] for p in parts)

    rows = []
    for chan in ('n1', 'n2'):
        v, i = sig[(chan, 'vertex')], sig[(chan, 'ips')]
        rel = 100.0 * (i / v - 1.0)                              # per draw
        for j, xv in enumerate(x):
            for k, subj in enumerate(subjects):
                r = rel[k, :, j]
                rows.append(dict(
                    label=label, channel=chan, x=float(xv), subject=int(subj),
                    shift_pct=float(np.median(r)),
                    lo=float(np.quantile(r, .025)),
                    hi=float(np.quantile(r, .975)),
                    p_gt0=float((r > 0).mean()),
                    sigma_vertex=float(np.median(v[k, :, j]))))
    ds.close()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f'subject_shifts.{label}.tsv'
    pd.DataFrame(rows).to_csv(f, sep='\t', index=False)
    print(f'wrote {f} ({len(subjects)} subjects x {len(x)} payoffs x 2 channels)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_dir', default='subject_shifts')
    ap.add_argument('--payoffs', nargs='+', type=float, default=[7, 28])
    args = ap.parse_args()
    main(args.label, args.trace_dir, args.out_dir, args.payoffs)
