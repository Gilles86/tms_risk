"""Where in the decision space cTBS moves behaviour, for one anchor model.

Same device as `decompose_pmc_channels`: rather than re-deriving the PMC algebra
by hand, let bauer build its own graph and tap the two functions where the
percepts and the choice distribution are produced. The paradigm is doubled --
every trial once as vertex, once as IPS -- so both counterfactuals exist for the
same trial and patsy still sees both levels of `stimulation_condition`.

Writes one long TSV, aggregated within draw across subjects (the aggregation the
data are summarized with), holding everything Figure 5 needs:

    rel_risky, rel_safe   perceived value of each option under cTBS, % change
    ratio_shift           perceived risky/safe ratio, IPS / vertex
    p_vertex, p_ips, dp   choice probability and the cTBS effect on it

    python -m tms_risk.behavior.scripts.extract_anchor_decision_space \\
        log-spl3-percmem --trace_dir .../cogmodels.anchor
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as ss
import bauer.core as bcore
import bauer.models.risky_choice as rc

from tms_risk.behavior.fit_anchor import build_model, parse_label

#: `log-affine-perc.pathfinder` names a variant refit; the grammar and the
#: model it builds are those of the part before the dot.
BASE = lambda lbl: lbl.split('.')[0]
from tms_risk.behavior.fit_model import get_data


#: The log-space observer inherits its choice rule from `BaseModel`, so the
#: calls live in bauer.core, not in bauer.models.risky_choice. Patching only the
#: latter -- which is what the older decompose script does -- leaves the graph
#: untouched and the tap silently produces nothing.
_TAP_MODULES = (bcore, rc)


def tap():
    """Expose the model's own percepts and choice distribution as Deterministics.
    `_get_choice_predictions` calls get_posterior for n1 then n2, so call order
    names them."""
    orig_diff = bcore.get_diff_dist
    orig_post = bcore.get_posterior
    calls = {'n': 0}

    def diff(mu1, sd1, mu2, sd2):
        dm, dsd = orig_diff(mu1, sd1, mu2, sd2)
        pm.Deterministic('diff_mu', dm)
        pm.Deterministic('diff_sd', dsd)
        return dm, dsd

    def post(*a):
        mu, sd = orig_post(*a)
        calls['n'] += 1
        if calls['n'] <= 2:
            pm.Deterministic(f'post_mu_{calls["n"]}', mu)
        return mu, sd

    for mod in _TAP_MODULES:
        mod.get_diff_dist, mod.get_posterior = diff, post


def main(label, bids_folder, trace_dir, out_dir, n_draws):
    parse_label(BASE(label))
    df = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm')
    idata = az.from_netcdf(Path(trace_dir) / f'model-{label}_trace.netcdf')

    doubled = pd.concat([df.assign(stimulation_condition=c) for c in
                         ('vertex', 'ips')])
    tap()
    model = build_model(BASE(label), doubled.copy())
    assert sorted(model.parameter_signature()['parameters']) == \
        sorted(idata.posterior.attrs['tms_risk_parameters'].split(',')), \
        'rebuilt parameter set differs from the stamped one'
    model.build_estimation_model(save_p_choice=True)
    missing = [v for v in ('diff_mu', 'diff_sd', 'post_mu_1', 'post_mu_2')
               if v not in model.estimation_model.named_vars]
    if missing:
        raise SystemExit(f'the tap did not take: {missing} absent from the graph')

    n_chain = idata.posterior.sizes['chain']
    keep = np.linspace(0, idata.posterior.sizes['draw'] - 1,
                       max(1, n_draws // n_chain)).astype(int)
    det = pm.compute_deterministics(
        idata.posterior.isel(draw=keep), model=model.estimation_model,
        var_names=['diff_mu', 'diff_sd', 'p', 'post_mu_1', 'post_mu_2'],
        merge_dataset=False, progressbar=False)

    def flat(k):
        return det[k].stack(sample=('chain', 'draw')).values     # (row, draw)

    # The log-space observer uses BaseModel's THRESHOLD choice rule:
    #     p = cumulative_normal(threshold, diff_mu, diff_sd)
    #       = Phi((log(p2/p1) - diff_mu) / diff_sd),
    # and get_diff_dist(mu1, .., mu2, ..) returns mu2 - mu1, called with n2
    # first, so diff_mu = post_n1 - post_n2. Reconstructing it as Phi(m/s)
    # -- the rule the natural-space models use -- is off by the whole
    # threshold, which is what the check below caught.
    dm, s, p = flat('diff_mu'), flat('diff_sd'), flat('p')
    mu1, mu2 = flat('post_mu_1'), flat('post_mu_2')
    thr = np.log(doubled['p2'].values / doubled['p1'].values)[:, None]
    m = thr - dm
    ok = np.isfinite(np.stack([m, s, p, mu1, mu2])).all(0).all(0)
    m, s, p, mu1, mu2 = (a[:, ok] for a in (m, s, p, mu1, mu2))
    err = np.abs(ss.norm.cdf(m / s) - p).max()
    print(f'reconstruction check: max |Phi(m/s) - p| = {err:.2e}')
    assert err < 1e-6, 'sign/scale convention does not reproduce the model p'

    n = len(df)
    d = df.reset_index().copy()
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    rf = d['risky_first'].values[:, None]
    # option 1 is the risky one exactly when the risky option came first
    risky_v, safe_v = np.where(rf, mu1[:n], mu2[:n]), np.where(rf, mu2[:n], mu1[:n])
    risky_i, safe_i = np.where(rf, mu1[n:], mu2[n:]), np.where(rf, mu2[n:], mu1[n:])
    p_v, p_i = p[:n], p[n:]
    # p is P(choose option 2); the risky option is option 2 when it came second
    pr_v = np.where(~rf, p_v, 1 - p_v)
    pr_i = np.where(~rf, p_i, 1 - p_i)

    # percepts live in log CHF for the log observer, so a difference of logs IS
    # the proportional change; for the natural observer take the plain ratio.
    log_space = idata.posterior.attrs['tms_risk_space'] == 'log'
    rel = ((lambda a, b: np.expm1(a - b)) if log_space
           else (lambda a, b: a / b - 1.0))
    q = dict(rel_risky=100 * rel(risky_i, risky_v),
             rel_safe=100 * rel(safe_i, safe_v),
             ratio_shift=(np.exp((risky_i - safe_i) - (risky_v - safe_v))
                          if log_space else (risky_i / safe_i) / (risky_v / safe_v)),
             p_vertex=pr_v, p_ips=pr_i, dp=pr_i - pr_v)

    d['rung'] = (d.groupby(['subject', 'n_safe'], group_keys=False)['frac']
                 .rank(method='dense').astype(int))
    grp = ['order', 'n_safe']
    keys = ['subject'] + grp
    idx = pd.MultiIndex.from_frame(d[keys])
    rows = []
    for name, arr in q.items():
        # subjects averaged WITHIN draw, then summarized over draws
        per_subj = pd.DataFrame(arr, index=idx).groupby(level=keys).mean()
        per = per_subj.groupby(grp).mean()
        # ALSO the median and between-subject SD. `rel_risky` is a PERCENTAGE
        # change of a quantity whose baseline varies a lot between people, and
        # its across-subject distribution is skewed +2.8: the mean is +9.5%
        # where the median is -0.4%, and three participants supply 86% of the
        # sum -- the three noisiest ones, whose choices carry the least
        # information. A figure that plots only the mean is reporting them.
        # See notes/audit_dp_magnitude.md, audit 2.
        med = per_subj.groupby(grp).median()
        sdb = per_subj.groupby(grp).std()
        lo, mid, hi = (per.quantile(.025, axis=1), per.median(axis=1),
                       per.quantile(.975, axis=1))
        # the median needs its OWN credible interval -- the mean's interval can
        # sit entirely on one side of it, which is what a skewed across-subject
        # distribution looks like
        r = pd.DataFrame({'lo': lo, 'mid': mid, 'hi': hi,
                          'subject_median': med.median(axis=1),
                          'median_lo': med.quantile(.025, axis=1),
                          'median_hi': med.quantile(.975, axis=1),
                          'subject_sd': sdb.median(axis=1)}).reset_index()
        r['quantity'] = name
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, 'label', label)

    # the observed cTBS effect, paired within subject, for the last panel
    o = (d.assign(y=d['chose_risky'].astype(float))
           .groupby(keys + ['stimulation_condition'])['y'].mean()
           .unstack('stimulation_condition'))
    o = (o['ips'] - o['vertex']).dropna().groupby(grp).agg(['mean', 'sem'])
    o.columns = ['observed', 'observed_sem']
    o = o.reset_index()
    o['label'], o['quantity'] = label, 'dp_observed'

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([out, o], ignore_index=True).to_csv(
        out_dir / f'decision_space.{label}.tsv', sep='\t', index=False)
    print(f'wrote {out_dir / f"decision_space.{label}.tsv"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-tmsrisk')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_dir', default='decision_space')
    ap.add_argument('--n_draws', default=400, type=int)
    a = ap.parse_args()
    main(a.label, a.bids_folder, a.trace_dir, a.out_dir, a.n_draws)
