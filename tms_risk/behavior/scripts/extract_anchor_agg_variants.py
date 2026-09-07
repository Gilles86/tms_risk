"""How much does the AGGREGATION choice move the Figure-5 mechanism panels?

Panels e/f are built from `decision_function.<label>.tsv`, which takes the
posterior MEDIAN over draws of the subject-MEAN P, on a UNIFORM GRID of
(safe payoff x risky/safe ratio).  Panels h/i (the PPC) take the MEAN over
draws of simulated choices on the ACTUAL trials.  Those differ on two axes at
once, so this script evaluates the same closed form under every combination
and reports which axis is responsible for what.

Axes crossed:
  * where      : 'trials' (the real design) vs 'grid' (uniform ratios)
  * over_draws : 'mean' vs 'median' vs plug-in at the per-subject posterior
                 'plugin_mean' / 'plugin_median' parameter vector
  * over_subj  : 'mean' vs 'median' across the 35 participants

Writes one long TSV; nothing is plotted and nothing in the repo is touched.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

from bauer.models.anchor_noise import NOISE_FORMS, AnchorNoiseMixin
from tms_risk.behavior.fit_model import get_data

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
    hit = [(n, m) for n, m in ((n, NAME_RE.match(n)) for n in names)
           if m and m.group(2) == chan]
    if not hit:
        return None, []
    form = hit[0][1].group(3)
    if NOISE_FORMS[form][0] == 1:
        return _Curve([1.0], form), [hit[0][0]]
    pairs = sorted((float(m.group(4)), n) for n, m in hit)
    return _Curve([a for a, _ in pairs], form), [n for _, n in pairs]


def main(label, trace_dir, bids_folder, out_tsv, n_draws):
    ds = xr.open_dataset(Path(trace_dir) / f'model-{label}_trace.netcdf',
                         group='posterior')
    a = ds.attrs
    space, placement = a['tms_risk_space'], a['tms_risk_placement']
    consistent = a.get('tms_risk_choice_noise', 'raw_evidence_sd') == 'consistent'
    names = a['tms_risk_parameters'].split(',')
    shared = placement in SHARED
    subj = [str(s) for s in ds['subject'].values]
    S = ds.sizes['chain'] * ds.sizes['draw']
    keep = np.linspace(0, S - 1, n_draws).astype(int)
    print(f'{label}: {ds.sizes["subject"]} subjects, {S} draws -> {n_draws} kept, '
          f'shared={shared} consistent={consistent}')

    # ---- parameter tensors: (subject, draw, regressor), plus plug-in copies --
    raw = {}
    for p in set(names):
        rdim = f'{p}_regressors'
        raw[p] = (ds[p].stack(sample=('chain', 'draw'))
                  .transpose('subject', 'sample', rdim).values[:, keep, :])
    ds.close()

    # A plug-in variant collapses the draw axis of the COEFFICIENTS first; the
    # integrating variants keep it and collapse P at the very end.  That is the
    # whole Jensen question, so build them as literally parallel tensors.
    tensors = {
        'integrate':      raw,
        'plugin_mean':    {p: v.mean(1, keepdims=True) for p, v in raw.items()},
        'plugin_median':  {p: np.median(v, 1, keepdims=True) for p, v in raw.items()},
    }

    def sigma(T, chan, cond, x, sidx):
        curve, pars = channel_curve(names, chan)
        B = curve.interp_matrix(np.asarray(x, float))          # (nX, n_anchors)
        th = []
        for p in pars:
            c = T[p]
            t = c[..., 0] if cond == 'ips' else c[..., 0] + (
                c[..., 1] if c.shape[-1] > 1 else 0.0)
            th.append(t[sidx])                                  # (nX, ndraw)
        th = np.stack(th, axis=-1)                              # (nX, ndraw, nA)
        if curve.anchor_link == 'log':
            return np.exp(np.einsum('xda,xa->xd', th, B))
        return np.einsum('xda,xa->xd', np.exp(th), B)

    def prior(T, which, kind, cond, sidx):
        c = T[f'{space}_{which}_prior_{kind}']
        v = c[..., 0]
        if cond == 'vertex' and c.shape[-1] > 1:
            v = v + c[..., 1]
        v = v[sidx]
        return np.exp(v) if kind == 'sd' else v

    def choice_p(T, n1, n2, rf, sidx):
        """(nX, ndraw) P(choose risky) under each cTBS condition."""
        nR = np.where(rf, n1, n2)[:, None]
        nS = np.where(rf, n2, n1)[:, None]
        out = {}
        for cond in ('ips', 'vertex'):
            if shared:
                s1 = sigma(T, 'perc', cond, n1, sidx) + sigma(T, 'mem', cond, n1, sidx)
                s2 = sigma(T, 'perc', cond, n2, sidx)
            else:
                s1 = sigma(T, 'n1', cond, n1, sidx)
                s2 = sigma(T, 'n2', cond, n2, sidx)
            nu_R = np.where(rf[:, None], s1, s2)
            nu_S = np.where(rf[:, None], s2, s1)
            sd_R = prior(T, 'risky', 'sd', cond, sidx)
            sd_S = prior(T, 'safe', 'sd', cond, sidx)
            mu_R = prior(T, 'risky', 'mu', cond, sidx)
            mu_S = prior(T, 'safe', 'mu', cond, sidx)
            wR = sd_R ** 2 / (sd_R ** 2 + nu_R ** 2)
            wS = sd_S ** 2 / (sd_S ** 2 + nu_S ** 2)
            den = (np.sqrt((wR * nu_R) ** 2 + (wS * nu_S) ** 2) if consistent
                   else np.sqrt(nu_R ** 2 + nu_S ** 2))
            num = (wR * np.log(nR) + (1 - wR) * mu_R
                   - wS * np.log(nS) - (1 - wS) * mu_S + np.log(0.55))
            out[cond] = dict(P=norm.cdf(num / den), num=num, den=den,
                             postR=wR * np.log(nR) + (1 - wR) * mu_R,
                             postS=wS * np.log(nS) + (1 - wS) * mu_S)
        return out

    # ---------------- the real trials ---------------------------------------
    df = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    df['sub_s'] = df['subject'].astype(str)
    df = df[df.sub_s.isin(subj)].copy()
    sidx = np.array([subj.index(s) for s in df.sub_s])
    df['order'] = df['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    df['stake'] = (df['n_safe'] + df['n_risky']) / 2
    # exactly the PPC's binning, so the numbers are comparable cell by cell
    df['rung'] = (df.groupby(['subject', 'n_safe'], group_keys=False)['frac']
                  .rank(method='dense').astype(int))
    df['stake_grp'] = (df.groupby('subject', group_keys=False)['stake']
                       .apply(lambda v: (v > v.median()).astype(int)))
    n1 = df['n1'].values.astype(float)
    n2 = df['n2'].values.astype(float)
    rf = df['risky_first'].values.astype(bool)

    # ---------------- the uniform grid panels e/f use ------------------------
    # Same safe payoffs and ratio range as `extract_anchor_decision_function`,
    # every subject on every grid point (that is what the grid assumes).
    g_safe = np.array([7., 10., 14., 20., 28.])
    g_ratio = np.exp(np.linspace(np.log(1.2), np.log(3.5), 25))
    gs, gr, go = np.meshgrid(g_safe, g_ratio, [True, False], indexing='ij')
    gs, gr, go = gs.ravel(), gr.ravel(), go.ravel()
    nsub = len(subj)
    G = len(gs)
    g_sidx = np.repeat(np.arange(nsub), G)
    g_nS = np.tile(gs, nsub)
    g_ratio_f = np.tile(gr, nsub)
    g_rf = np.tile(go, nsub).astype(bool)
    g_nR = g_ratio_f * g_nS
    g_n1 = np.where(g_rf, g_nR, g_nS)
    g_n2 = np.where(g_rf, g_nS, g_nR)
    gdf = pd.DataFrame(dict(subject=np.repeat(np.arange(nsub), G),
                            order=np.where(g_rf, 'Risky first', 'Risky second'),
                            n_safe=g_nS, ratio=g_ratio_f))

    rows = []
    for tname, T in tensors.items():
        Ptr = None
        for where in ('trials', 'trials_safe', 'grid'):
            if where.startswith('trials'):
                if Ptr is None:
                    Ptr = choice_p(T, n1, n2, rf, sidx)
                P = Ptr
                base = df
                keys = (['subject', 'order', 'stake_grp', 'rung']
                        if where == 'trials' else ['subject', 'order', 'n_safe'])
            else:
                P = choice_p(T, g_n1, g_n2, g_rf, g_sidx)
                base, keys = gdf, ['subject', 'order', 'n_safe']
            # every quantity the mechanism panels draw, differenced the same way
            quant = {
                'dp':        P['ips']['P'] - P['vertex']['P'],
                'num_shift': P['ips']['num'] - P['vertex']['num'],
                'den_ratio': P['ips']['den'] / P['vertex']['den'],
                'd_postR':   P['ips']['postR'] - P['vertex']['postR'],
                'd_postS':   P['ips']['postS'] - P['vertex']['postS'],
                'p_vertex':  P['vertex']['P'],
            }
            idx = pd.MultiIndex.from_frame(base[keys])
            grp = [k for k in keys if k != 'subject']
            for qname, Q in quant.items():
                per_sub = pd.DataFrame(Q, index=idx).groupby(level=keys).mean()
                for osub in ('mean', 'median'):
                    cells = getattr(per_sub.groupby(level=grp), osub)()
                    draw_ops = ({'mean': cells.mean(axis=1),
                                 'median': cells.median(axis=1)}
                                if tname == 'integrate' else
                                {'point': cells.iloc[:, 0]})
                    for odraw, v in draw_ops.items():
                        variant = (f'{tname}/{odraw}' if tname == 'integrate'
                                   else tname)
                        for k, val in v.items():
                            r = dict(zip(grp, k if isinstance(k, tuple) else (k,)))
                            r.update(label=label, where=where, variant=variant,
                                     over_subjects=osub, quantity=qname,
                                     value=float(val))
                            rows.append(r)
    out = pd.DataFrame(rows)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv} ({len(out)} rows)')

    for where in ('trials', 'trials_safe', 'grid'):
        sub = out[(out['where'] == where) & (out['over_subjects'] == 'mean')
                  & (out['quantity'] == 'dp')]
        w = sub.pivot_table(index=[c for c in ('order', 'stake_grp', 'rung', 'n_safe')
                                  if c in sub.columns and sub[c].notna().any()],
                            columns='variant', values='value')
        print(f'\n--- {where}: mean dP by order ---')
        print(w.groupby(level='order').mean().round(4).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-tmsrisk')
    ap.add_argument('--out_tsv', required=True)
    ap.add_argument('--n_draws', default=400, type=int)
    a = ap.parse_args()
    main(a.label, a.trace_dir, a.bids_folder, a.out_tsv, a.n_draws)
