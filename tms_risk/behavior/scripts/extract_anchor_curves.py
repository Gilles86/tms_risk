"""Noise functions sigma(payoff) from every anchor-parameterised trace.

Cheap by construction: the anchor parameterisation means the noise function is
``B(x) @ exp(theta)`` (or ``exp(B(x) @ theta)`` for the log link) with B a
closed-form partition of unity, so no PyMC graph is built and no likelihood is
evaluated. Only the posterior group means and the per-subject coefficients are
read, lazily, out of each ~600 MB netcdf.

Two families, one comparable output. Under ``shared_perceptual_noise`` the free
channels are perc/mem and the composition is sigma_n1 = sigma_perc + sigma_mem,
sigma_n2 = sigma_perc; under ``independent`` the free channels are n1/n2
directly. Both are written out as n1/n2 as well as in their own coordinates, so
the two halves of the grid can be plotted on the same axes.

Treatment coding: `stimulation_condition[T.vertex]` means the Intercept IS the
IPS (stimulated) condition and the slope carries vertex - IPS. The `delta` rows
are IPS minus vertex, i.e. the cTBS effect, positive = cTBS added noise.

    python -m tms_risk.behavior.scripts.extract_anchor_curves \\
        --trace_dir /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor \\
        --out_tsv /home/gdehol/anchor_curves.tsv
"""
import argparse
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from bauer.models.anchor_noise import NOISE_FORMS, AnchorNoiseMixin

#: composition of the two free channels into the two presented options
COMPOSE = {
    'shared_perceptual_noise': {'n1': ('perc', 'mem'), 'n2': ('perc',)},
    'independent':             {'n1': ('n1',),         'n2': ('n2',)},
}
PLACEMENT_MEMORY = {'null': 'shared_perceptual_noise',
                    'perc': 'shared_perceptual_noise',
                    'mem': 'shared_perceptual_noise',
                    'percmem': 'shared_perceptual_noise'}
NAME_RE = re.compile(r'^(log|chf)_(perc|mem|n1|n2)_'
                     r'(' + '|'.join(sorted(NOISE_FORMS, key=len,
                                              reverse=True))
                     + r')_sd(\d*)$')


class _Curve(AnchorNoiseMixin):
    """Just enough of the mixin to call `interp_matrix` — the real interpolation
    code, not a reimplementation of it."""

    def __init__(self, anchors, noise_form):
        self.noise_form = noise_form
        self.n_anchors, self.anchor_link = NOISE_FORMS[noise_form]
        # `anchors` is a read-only property that derives them from a paradigm;
        # here they come from the trace's own stamp, so set the backing field.
        self._anchors = np.asarray(anchors, dtype=float)


def channel_curve(names, chan):
    """A `_Curve` for ONE channel, built from that channel's own parameter names.

    With split forms ('spl5+affine') the two channels have different forms AND
    different anchor sets, while the trace stamps only the primary form's
    anchors. Every parameter name carries both -- `log_mem_affine_sd7`,
    `log_perc_spl5_sd13` -- so read them off the names.
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


def channel_params(names, channel):
    """Anchor parameter names for one channel, ordered by anchor payoff."""
    out = []
    for n in names:
        m = NAME_RE.match(n)
        if m and m.group(2) == channel:
            out.append((float(m.group(4)) if m.group(4) else 0.0, n))
    return [n for _, n in sorted(out)]


def theta(ds, names, condition):
    """(sample, K) group-level log anchor values under one stimulation condition."""
    cols = []
    for n in names:
        v = ds[f'{n}_mu']
        rdim = f'{n}_regressors'
        coef = v.stack(sample=('chain', 'draw')).transpose('sample', rdim).values
        icpt = coef[:, 0]                              # Intercept == IPS
        if condition == 'ips':
            cols.append(icpt)
        elif condition == 'vertex':
            slope = coef[:, 1] if coef.shape[1] > 1 else 0.0
            cols.append(icpt + slope)                  # vertex = IPS + contrast
        else:
            raise ValueError(condition)
    return np.stack(cols, axis=1)


def sigma(curve, th, x):
    """sigma(x) for (sample, K) log anchor values -> (sample, len(x))."""
    B = curve.interp_matrix(x)                         # (len(x), K)
    if curve.anchor_link == 'log':
        return np.exp(th @ B.T)
    return np.exp(th) @ B.T


def q(a, axis=0):
    lo, mid, hi = np.quantile(a, [.025, .5, .975], axis=axis)
    return lo, mid, hi


def label_from_path(path, attrs):
    """Label a trace by its FILENAME, not by its stamp.

    A variant refit (`...pathfinder_trace.netcdf`) carries the base label in
    `tms_risk_label`, so trusting the stamp collapses the variant and the
    original into one row.
    """
    stem = Path(path).name.replace('model-', '').replace('_trace.netcdf', '')
    return stem or attrs.get('tms_risk_label')


def main(trace_dir, out_tsv, out_subject_tsv, n_grid):
    rows, srows = [], []
    for path in sorted(glob(str(Path(trace_dir) / 'model-*_trace.netcdf'))):
        ds = xr.open_dataset(path, group='posterior')
        a = ds.attrs
        label = label_from_path(path, a)
        if label is None or 'tms_risk_anchors' not in a:
            print(f'  skip {Path(path).name}: not an anchor trace')
            ds.close()
            continue
        space, form, placement = (a['tms_risk_space'], a['tms_risk_noise_form'],
                                  a['tms_risk_placement'])
        anchors = [float(v) for v in a['tms_risk_anchors'].split(',')]
        names = a['tms_risk_parameters'].split(',')
        memory = PLACEMENT_MEMORY.get(placement, 'independent')
        curve = _Curve(anchors, form.split('+')[0])
        # Grid over the PAYOFFS PRESENTED, not over the anchors: Weber has a
        # single anchor, so an anchor-spanned grid would collapse to one point
        # and its (perfectly well-defined, constant) noise function would not
        # plot at all.
        cd = xr.open_dataset(path, group='constant_data')
        pay = np.concatenate([np.asarray(cd['n1']), np.asarray(cd['n2'])])
        cd.close()
        x = np.exp(np.linspace(np.log(pay.min()), np.log(pay.max()), n_grid))

        free = [c for c in ('perc', 'mem', 'n1', 'n2') if channel_params(names, c)]
        sig = {}
        for chan in free:
            ccur, pars = channel_curve(names, chan)
            for cond in ('ips', 'vertex'):
                sig[(chan, cond)] = sigma(ccur, theta(ds, pars, cond), x)
            # per-subject anchor values, posterior mean, both conditions
            for p in pars:
                v = ds[p]
                rdim = f'{p}_regressors'
                coef = v.mean(('chain', 'draw')).transpose('subject', rdim).values
                for i, subj in enumerate(ds['subject'].values):
                    ips = coef[i, 0]
                    ver = ips + (coef[i, 1] if coef.shape[1] > 1 else 0.0)
                    srows.append(dict(label=label, space=space, form=form,
                                      placement=placement, channel=chan,
                                      parameter=p, subject=int(subj),
                                      sigma_ips=float(np.exp(ips)),
                                      sigma_vertex=float(np.exp(ver))))
        # composed n1 / n2, so both families land in the same coordinates
        for comp, parts in COMPOSE[memory].items():
            if all((p, 'ips') in sig for p in parts) and comp not in free:
                for cond in ('ips', 'vertex'):
                    sig[(comp, cond)] = sum(sig[(p, cond)] for p in parts)

        for (chan, cond), s in sig.items():
            lo, mid, hi = q(s)
            for j, xv in enumerate(x):
                rows.append(dict(label=label, space=space, form=form,
                                 placement=placement, memory_model=memory,
                                 channel=chan, condition=cond, x=float(xv),
                                 lo=lo[j], mid=mid[j], hi=hi[j]))
        # the cTBS effect itself, as a difference within draw (IPS - vertex),
        # and the same thing as a FRACTION of the vertex noise -- the scale the
        # psychophysical analyses work on, and the one a figure can compare
        # across payoffs without the growth of sigma itself doing the talking
        for chan in {c for c, _ in sig}:
            r = 100 * (sig[(chan, 'ips')] / sig[(chan, 'vertex')] - 1.0)
            lo, mid, hi = q(r)
            for j, xv in enumerate(x):
                rows.append(dict(label=label, space=space, form=form,
                                 placement=placement, memory_model=memory,
                                 channel=chan, condition='delta_pct', x=float(xv),
                                 lo=lo[j], mid=mid[j], hi=hi[j]))
            d = sig[(chan, 'ips')] - sig[(chan, 'vertex')]
            lo, mid, hi = q(d)
            pgt = (d > 0).mean(0)
            for j, xv in enumerate(x):
                rows.append(dict(label=label, space=space, form=form,
                                 placement=placement, memory_model=memory,
                                 channel=chan, condition='delta', x=float(xv),
                                 lo=lo[j], mid=mid[j], hi=hi[j], p_gt0=pgt[j]))
        ds.close()
        print(f'  {label:28s} {len(free)} free channels, anchors {anchors}')

    df = pd.DataFrame(rows)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv}  ({len(df)} rows, {df.label.nunique()} models)')
    sdf = pd.DataFrame(srows)
    sdf.to_csv(out_subject_tsv, sep='\t', index=False)
    print(f'wrote {out_subject_tsv}  ({len(sdf)} rows)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_tsv', default='anchor_curves.tsv')
    ap.add_argument('--out_subject_tsv', default='anchor_curves_subject.tsv')
    ap.add_argument('--n_grid', default=40, type=int)
    args = ap.parse_args()
    main(args.trace_dir, args.out_tsv, args.out_subject_tsv, args.n_grid)
