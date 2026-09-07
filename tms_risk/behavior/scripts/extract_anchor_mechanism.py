"""The four mechanism traces of Figure 5's panels f/g, integrated over draws.

Panels f/g show, per presentation order and safe payoff, what cTBS did to

    * the perceived value of the risky option,
    * the perceived value of the safe option,
    * their ratio -- the numerator of the decision variable, and
    * the decision SD it is divided by.

All four are NONLINEAR in the parameters (the shrinkage weight is
`w = sd^2 / (sd^2 + nu^2)`), so the value at a point estimate is not the mean
value.  The panels used to be rebuilt in the plotting script from the
per-subject MEDIAN tables (`anchor_priors_subject.tsv`,
`anchor_curves_subject.tsv`, both written with `np.median`), which is a plug-in
estimator; scored against the model's own simulated PPC it reaches r = 0.977
and, worse, flips the sign of the small risky-first effect (-0.0013 -> +0.0014)
while compressing the risky-second effect by 21%.  Integrating over draws
first -- computing the quantity per posterior sample and collapsing at the end
-- reaches r = 0.991 and reproduces the PPC to 0.004 in probability units.
`notes/analyses/aggregation_check.md` has the full comparison.

Two further rules that fall out of the same check and are enforced here:

* Evaluate on the trials people **actually saw**.  The risky/safe ratios are a
  per-subject calibrated ladder, not uniform in log-ratio, and averaging the
  same algebra over a uniform grid flips the sign of the risky-first effect.
* Average across participants with the **mean**, never the median: the observed
  data and the posterior predictive are both subject means, and the
  between-subject distribution is right-skewed enough that the median
  compresses the risky-second effect by 40%.

Writes one row per (order, safe payoff) with the across-subject mean, its SEM,
and the 95% credible interval of the group mean over draws.

    python -m tms_risk.behavior.scripts.extract_anchor_mechanism log-power-n1n2 \\
        --trace_dir .../cogmodels.anchor --out_dir notes/data
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from bauer.models.anchor_noise import NOISE_FORMS, AnchorNoiseMixin
from tms_risk.behavior.fit_model import get_data

NAME_RE = re.compile(r'^(log|chf)_(perc|mem|n1|n2)_'
                     r'(weber|affine|power|genweber|spl3|spl5|spl7|spl9|cspl3|cspl5|cspl7)_sd(\d*)$')
#: placements whose noise lives on perceptual/memory channels rather than on the
#: first/second-presented option directly
SHARED = ('null', 'perc', 'mem', 'percmem', 'spsd', 'spmusd', 'percpsd',
          'percmempsd', 'percx', 'percmemx')
QUANTITIES = ('risky', 'safe', 'ratio', 'noise')


class _Curve(AnchorNoiseMixin):
    def __init__(self, anchors, noise_form):
        self.noise_form = noise_form
        self.n_anchors, self.anchor_link = NOISE_FORMS[noise_form]
        self._anchors = np.asarray(anchors, dtype=float)


def channel_curve(names, chan):
    """A `_Curve` for ONE channel, read off that channel's parameter names.

    With split forms (`spl5+affine`) the two channels have different forms and
    different anchor sets while the trace stamps only the primary form's
    anchors, so the names -- `log_mem_affine_sd7`, `log_perc_spl5_sd13` -- are
    the only reliable source.
    """
    hit = [(n, m) for n, m in ((n, NAME_RE.match(n)) for n in names)
           if m and m.group(2) == chan]
    if not hit:
        return None, []
    form = hit[0][1].group(3)
    if NOISE_FORMS[form][0] == 1:            # weber: one flat value, no anchor
        return _Curve([1.0], form), [hit[0][0]]
    pairs = sorted((float(m.group(4)), n) for n, m in hit)
    return _Curve([a for a, _ in pairs], form), [n for _, n in pairs]


def main(label, trace_dir, bids_folder, out_dir, n_draws):
    ds = xr.open_dataset(Path(trace_dir) / f'model-{label}_trace.netcdf',
                         group='posterior')
    a = ds.attrs
    space, placement = a['tms_risk_space'], a['tms_risk_placement']
    consistent = a.get('tms_risk_choice_noise', 'raw_evidence_sd') == 'consistent'
    names = a['tms_risk_parameters'].split(',')
    shared = placement in SHARED
    subj = [str(s) for s in ds['subject'].values]
    S = ds.sizes['chain'] * ds.sizes['draw']
    keep = np.linspace(0, S - 1, min(n_draws, S)).astype(int)
    par = {}
    for p in set(names):
        par[p] = (ds[p].stack(sample=('chain', 'draw'))
                  .transpose('subject', 'sample', f'{p}_regressors')
                  .values[:, keep, :])
    ds.close()
    print(f'{label}: {len(subj)} subjects, {S} draws -> {len(keep)} kept, '
          f'shared={shared} consistent={consistent}')

    def sigma(chan, cond, x, sidx):
        curve, pars = channel_curve(names, chan)
        B = curve.interp_matrix(np.asarray(x, float))          # (nT, n_anchors)
        th = []
        for p in pars:
            c = par[p]
            t = c[..., 0] if cond == 'ips' else c[..., 0] + (
                c[..., 1] if c.shape[-1] > 1 else 0.0)
            th.append(t[sidx])
        th = np.stack(th, axis=-1)                              # (nT, nD, nA)
        if curve.anchor_link == 'log':
            return np.exp(np.einsum('tda,ta->td', th, B))
        return np.einsum('tda,ta->td', np.exp(th), B)

    def prior(which, kind, cond, sidx):
        c = par[f'{space}_{which}_prior_{kind}']
        v = c[..., 0]
        if cond == 'vertex' and c.shape[-1] > 1:
            v = v + c[..., 1]
        v = v[sidx]
        return np.exp(v) if kind == 'sd' else v

    df = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    df['sub_s'] = df['subject'].astype(str)
    df = df[df.sub_s.isin(subj)].copy()
    sidx = np.array([subj.index(s) for s in df.sub_s])
    df['order'] = df['risky_first'].map({True: 'Risky first',
                                         False: 'Risky second'})
    n1 = df['n1'].values.astype(float)
    n2 = df['n2'].values.astype(float)
    rf = df['risky_first'].values.astype(bool)
    nR = np.where(rf, n1, n2)[:, None]
    nS = np.where(rf, n2, n1)[:, None]

    post, den = {}, {}
    for cond in ('ips', 'vertex'):
        if shared:
            # sigma_n1 = perceptual + memory (the first option is also held in
            # memory); sigma_n2 = perceptual only
            s1 = sigma('perc', cond, n1, sidx) + sigma('mem', cond, n1, sidx)
            s2 = sigma('perc', cond, n2, sidx)
        else:
            s1, s2 = sigma('n1', cond, n1, sidx), sigma('n2', cond, n2, sidx)
        nu_R = np.where(rf[:, None], s1, s2)
        nu_S = np.where(rf[:, None], s2, s1)
        sd_R, sd_S = prior('risky', 'sd', cond, sidx), prior('safe', 'sd', cond, sidx)
        mu_R, mu_S = prior('risky', 'mu', cond, sidx), prior('safe', 'mu', cond, sidx)
        wR = sd_R ** 2 / (sd_R ** 2 + nu_R ** 2)
        wS = sd_S ** 2 / (sd_S ** 2 + nu_S ** 2)
        post[(cond, 'risky')] = wR * np.log(nR) + (1 - wR) * mu_R
        post[(cond, 'safe')] = wS * np.log(nS) + (1 - wS) * mu_S
        den[cond] = (np.sqrt((wR * nu_R) ** 2 + (wS * nu_S) ** 2) if consistent
                     else np.sqrt(nu_R ** 2 + nu_S ** 2))

    keys = ['subject', 'order', 'n_safe']
    idx = pd.MultiIndex.from_frame(df[keys])

    def cellmean(arr):
        return pd.DataFrame(arr, index=idx).groupby(level=keys).mean()

    dR = cellmean(post[('ips', 'risky')] - post[('vertex', 'risky')])
    dS = cellmean(post[('ips', 'safe')] - post[('vertex', 'safe')])
    dn = cellmean(den['ips']) / cellmean(den['vertex'])
    # percent change, which is what the axis is labelled in; expm1 because the
    # quantities are differences of logs
    Q = {'risky': 100 * np.expm1(dR), 'safe': 100 * np.expm1(dS),
         'ratio': 100 * np.expm1(dR - dS), 'noise': 100 * (dn - 1)}

    rows = {}
    for q, V in Q.items():
        # integrate over draws FIRST (one number per participant), then treat
        # participants as the sampling unit, exactly as the observed data are
        per_sub = V.mean(axis=1)
        grp = per_sub.groupby(level=['order', 'n_safe'])
        m, sd, n = grp.mean(), grp.std(ddof=1), grp.size()
        # and separately: the posterior of the GROUP mean, which is the honest
        # band for a model-derived curve
        gdraws = V.groupby(level=['order', 'n_safe']).mean()
        for k in m.index:
            r = rows.setdefault(k, dict(label=label, order=k[0], n_safe=k[1],
                                        n_sub=int(n[k])))
            r[q] = float(m[k])
            r[f'{q}_sem'] = float(sd[k] / np.sqrt(n[k]))
            r[f'{q}_sd'] = float(sd[k])
            r[f'{q}_lo'] = float(np.quantile(gdraws.loc[k].values, .025))
            r[f'{q}_hi'] = float(np.quantile(gdraws.loc[k].values, .975))

    out = pd.DataFrame(list(rows.values())).sort_values(['order', 'n_safe'])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f'anchor_mechanism.{label}.tsv'
    out.to_csv(f, sep='\t', index=False)
    print(f'wrote {f} ({len(out)} rows)')
    print(out[['order', 'n_safe'] + list(QUANTITIES)].round(2).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-tmsrisk')
    ap.add_argument('--out_dir', default='notes/data')
    ap.add_argument('--n_draws', default=800, type=int)
    a = ap.parse_args()
    main(a.label, a.trace_dir, a.bids_folder, a.out_dir, a.n_draws)
