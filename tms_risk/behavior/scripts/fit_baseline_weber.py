"""Is choice consistency independent of stake? The test Weber's law has to pass.

Weber's law -- scalar invariance, constant noise in log space -- predicts that
the slope of the psychometric function does not depend on the magnitudes being
compared. This fits that slope separately for low- and high-stake trials in the
BASELINE session, before any participant was stimulated, so the test is
independent of everything else in the paper. It also uses the full cohort
(73 participants), including the 38 who were never stimulated.

Probit with subject dummies, bootstrapped over participants for the interval.
Subject dummies matter: pooling participants with different indifference points
flattens the aggregate psychometric function and would bias the slope down.

    python -m tms_risk.behavior.scripts.fit_baseline_weber
"""
import argparse
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

warnings.filterwarnings('ignore')

FORMULA = 'y ~ x*rf*hi + C(subject)'


def prep(bids_folder, session):
    from tms_risk.utils.data import get_all_behavior
    d = get_all_behavior(bids_folder=bids_folder).reset_index()
    if session:
        d = d[d.session.astype(str).str.startswith(str(session))]
    d = d.dropna(subset=['chose_risky', 'frac', 'n_safe', 'n_risky'])
    d['x'] = np.log(d['frac'])
    d['rf'] = d['risky_first'].astype(float)
    d['avg'] = (d['n_safe'] + d['n_risky']) / 2
    d['hi'] = (d.groupby('subject')['avg']
               .transform(lambda v: pd.qcut(v, 2, labels=[0, 1],
                                            duplicates='drop')).astype(float))
    d['y'] = d['chose_risky'].astype(float)
    d['subject'] = d['subject'].astype(str)
    return d.dropna(subset=['hi']).reset_index(drop=True)


def slopes(d):
    y, X = patsy.dmatrices(FORMULA, d, return_type='dataframe')
    r = sm.GLM(y, X, family=sm.families.Binomial(
        sm.families.links.Probit())).fit()
    p = r.params

    def g(*parts):
        for perm in permutations(parts):
            n = ':'.join(perm)
            if n in p.index:
                return float(p[n])
        return 0.0

    out = {}
    for rf, order in [(0., 'Risky second'), (1., 'Risky first')]:
        for hi, stake in [(0., 'low'), (1., 'high')]:
            out[(order, stake)] = (g('x') + rf * g('x', 'rf') + hi * g('x', 'hi')
                                   + rf * hi * g('x', 'rf', 'hi'))
    return out


def curves_predictive(d, out_tsv, n_boot=300, seed=0):
    """Posterior-predictive-style band for the fitted psychometric curves.

    An s.e.m. on the observed points answers "how precisely did we measure this
    proportion". The question a model figure has to answer is different: "could
    this model have produced these data". That needs the model's PREDICTIVE
    uncertainty -- parameter uncertainty AND the sampling noise of a proportion
    at the trial counts actually run -- drawn as a band, with the observed
    points on top of it. Bands and points then live on the same footing and a
    point outside the band is a real misfit rather than a small error bar.

    Here the fit is maximum-likelihood, so parameter uncertainty comes from a
    nonparametric bootstrap over PARTICIPANTS (the unit of exchangeability),
    and the sampling noise from a binomial draw at each cell's n_trials.
    """
    rng = np.random.default_rng(seed)
    d = d.copy()
    d['rung'] = (d.groupby(['subject', 'n_safe'], group_keys=False)['frac']
                 .rank(method='dense').astype(int))
    d['order'] = np.where(d.rf == 1, 'Risky first', 'Risky second')
    d['stake'] = np.where(d.hi == 1, 'high', 'low')
    keys = ['order', 'stake', 'rung']
    cells = (d.groupby(keys).agg(frac=('frac', 'mean'),
                                 n_trials=('y', 'size')).reset_index())
    subs = d.subject.unique()
    draws = []
    for _ in range(n_boot):
        take = rng.choice(subs, size=len(subs), replace=True)
        b = pd.concat([d[d.subject == s].assign(subject=f'{s}_{i}')
                       for i, s in enumerate(take)], ignore_index=True)
        try:
            y, X = patsy.dmatrices(FORMULA, b, return_type='dataframe')
            r = sm.GLM(y, X, family=sm.families.Binomial(
                sm.families.links.Probit())).fit()
        except Exception:
            continue
        pp = r.params

        def gt(*parts):
            for perm in permutations(parts):
                n = ':'.join(perm)
                if n in pp.index:
                    return float(pp[n])
            return 0.0
        sub = [n for n in pp.index if n.startswith('C(subject)')]
        b0 = (float(pp['Intercept'])
              + float(np.sum([pp[n] for n in sub])) / (len(sub) + 1))
        from scipy.stats import norm
        rfv = (cells.order == 'Risky first').astype(float).values
        hiv = (cells.stake == 'high').astype(float).values
        inter = b0 + rfv * gt('rf') + hiv * gt('hi') + rfv * hiv * gt('rf', 'hi')
        slope = (gt('x') + rfv * gt('x', 'rf') + hiv * gt('x', 'hi')
                 + rfv * hiv * gt('x', 'rf', 'hi'))
        pr = norm.cdf(inter + slope * np.log(cells.frac.values))
        # add the sampling noise of a proportion at the real trial counts
        draws.append(rng.binomial(cells.n_trials.values, pr)
                     / cells.n_trials.values)
    D = np.array(draws)
    out = cells.copy()
    out['pred_lo'], out['pred_mid'], out['pred_hi'] = np.quantile(
        D, [.025, .5, .975], axis=0)
    out['n_boot'] = len(D)
    f = str(out_tsv).replace('.tsv', '.pred.tsv')
    out.to_csv(f, sep='\t', index=False)
    print(f'wrote {f} ({len(D)} bootstrap replicates)')


def curves(d, out_tsv):
    """Observed psychometric curves, and the fitted probit through them.

    Points are aligned on each participant's own calibrated ladder rung before
    averaging. Pooling raw ratios instead would mix participants with different
    indifference points and flatten every curve, which is exactly the effect
    under test.
    """
    d = d.copy()
    d['rung'] = (d.groupby(['subject', 'n_safe'], group_keys=False)['frac']
                 .rank(method='dense').astype(int))
    keys = ['order', 'stake', 'rung']
    d['order'] = np.where(d.rf == 1, 'Risky first', 'Risky second')
    d['stake'] = np.where(d.hi == 1, 'high', 'low')
    per = d.groupby(keys + ['subject'])['y'].mean()
    g = per.groupby(keys).agg(['mean', 'sem', 'size'])
    g.columns = ['observed', 'observed_sem', 'n_sub']
    g = g.join(d.groupby(keys)['frac'].mean().rename('frac'))
    g = g.join(d.groupby(keys).size().rename('n_trials')).reset_index()

    # the fitted probit, at the average participant's intercept
    y, X = patsy.dmatrices(FORMULA, d, return_type='dataframe')
    r = sm.GLM(y, X, family=sm.families.Binomial(
        sm.families.links.Probit())).fit()
    pp = r.params

    def gt(*parts):
        for perm in permutations(parts):
            n = ':'.join(perm)
            if n in pp.index:
                return float(pp[n])
        return 0.0

    sub = [n for n in pp.index if n.startswith('C(subject)')]
    b0 = float(pp['Intercept']) + float(np.sum([pp[n] for n in sub])) / (len(sub) + 1)
    from scipy.stats import norm
    rows, pse = [], {}
    xs = np.linspace(np.log(d.frac.min()), np.log(d.frac.max()), 60)
    for rf, order in [(0., 'Risky second'), (1., 'Risky first')]:
        for hiv, stake in [(0., 'low'), (1., 'high')]:
            inter = (b0 + rf * gt('rf') + hiv * gt('hi') + rf * hiv * gt('rf', 'hi'))
            slope = (gt('x') + rf * gt('x', 'rf') + hiv * gt('x', 'hi')
                     + rf * hiv * gt('x', 'rf', 'hi'))
            # the curves differ in TWO ways: where they sit (the indifference
            # point -- a bias effect) and how steep they are. Weber's law is a
            # claim about the slope alone, so centre each curve on its own
            # indifference point and the comparison isolates it.
            pse[(order, stake)] = -inter / slope
            for xv in xs:
                rows.append(dict(order=order, stake=stake, frac=float(np.exp(xv)),
                                 x_centred=float(xv - pse[(order, stake)]),
                                 fitted=float(norm.cdf(inter + slope * xv))))
    fit = pd.DataFrame(rows)
    g['x_centred'] = [np.log(f) - pse[(o, st)]
                      for f, o, st in zip(g.frac, g.order, g.stake)]
    ps = pd.DataFrame([dict(order=o, stake=st, pse_logfrac=v,
                            pse_frac=float(np.exp(v))) for (o, st), v in pse.items()])
    ps.to_csv(str(out_tsv).replace('.tsv', '.pse.tsv'), sep='\t', index=False)
    print(ps.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    g.to_csv(out_tsv, sep='\t', index=False)
    fit.to_csv(str(out_tsv).replace('.tsv', '.fitted.tsv'), sep='\t', index=False)
    print(f'wrote {out_tsv} and .fitted.tsv')


def slopes_boot(d, n_boot, seed, tag):
    """Point estimate and bootstrap interval for every (order, stake) cell."""
    subs = np.sort(d.subject.unique())
    point = slopes(d)
    by_sub = {s: g for s, g in d.groupby('subject')}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.choice(len(subs), len(subs), replace=True)
        r = pd.concat([by_sub[subs[i]].assign(subject=f'b{j}')
                       for j, i in enumerate(idx)], ignore_index=True)
        try:
            s_ = slopes(r)
        except Exception:                                   # noqa: BLE001
            continue
        for k, v in s_.items():
            boot[k].append(v)
    rows = []
    for (order, stake), v in point.items():
        b = np.array(boot[(order, stake)], float)
        b = b[np.isfinite(b)]
        rows.append(dict(group=tag, order=order, stake=stake, slope=v,
                         lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                         n_sub=len(subs)))
    for order in ['Risky second', 'Risky first']:
        b = (np.array(boot[(order, 'low')], float)
             - np.array(boot[(order, 'high')], float))
        b = b[np.isfinite(b)]
        rows.append(dict(group=tag, order=order, stake='low-minus-high',
                         slope=point[(order, 'low')] - point[(order, 'high')],
                         lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                         p_gt0=float((b > 0).mean()), n_sub=len(subs)))
    return pd.DataFrame(rows)


def main(bids_folder, session, out_tsv, n_boot, seed):
    d = prep(bids_folder, session)
    subs = np.sort(d.subject.unique())
    print(f'{len(d)} trials, {len(subs)} participants, session={session}')
    point = slopes(d)
    by_sub = {s: g for s, g in d.groupby('subject')}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.choice(len(subs), len(subs), replace=True)
        r = pd.concat([by_sub[subs[i]].assign(subject=f'b{j}')
                       for j, i in enumerate(idx)], ignore_index=True)
        try:
            s_ = slopes(r)
        except Exception:                                   # noqa: BLE001
            continue
        for k, v in s_.items():
            boot[k].append(v)

    rows = []
    for (order, stake), v in point.items():
        b = np.array(boot[(order, stake)], float)
        b = b[np.isfinite(b)]
        rows.append(dict(session=session, order=order, stake=stake, slope=v,
                         lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                         n_boot=len(b)))
    for order in ['Risky second', 'Risky first']:
        b = (np.array(boot[(order, 'low')], float)
             - np.array(boot[(order, 'high')], float))
        b = b[np.isfinite(b)]
        rows.append(dict(session=session, order=order, stake='low-minus-high',
                         slope=point[(order, 'low')] - point[(order, 'high')],
                         lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                         p_gt0=float((b > 0).mean()), n_boot=len(b)))
    out = pd.DataFrame(rows)
    out.to_csv(out_tsv, sep='\t', index=False)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'wrote {out_tsv}')
    curves(d, str(out_tsv).replace('slopes', 'curves'))
    curves_predictive(d, str(out_tsv).replace('slopes', 'curves'),
                      n_boot=n_boot, seed=seed)

    # Does the violation depend on who the participant is? The 38 never
    # stimulated are an independent replication of the 35 who were.
    from tms_risk.utils.data import get_tms_subjects
    tms = {str(x) for x in get_tms_subjects()}
    d['grp'] = np.where(d.subject.isin(tms), 'Stimulated later',
                        'Never stimulated')
    parts = [slopes_boot(g, n_boot, seed + 7, tag)
             for tag, g in d.groupby('grp')]
    parts.append(slopes_boot(d, n_boot, seed + 7, 'All'))
    gb = pd.concat(parts, ignore_index=True)
    f2 = str(out_tsv).replace('slopes', 'slopes_bygroup')
    gb.to_csv(f2, sep='\t', index=False)
    print(gb[gb.stake == 'low-minus-high'].to_string(
        index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'wrote {f2}')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--session', default='1')
    ap.add_argument('--out_tsv', default=None)
    ap.add_argument('--n_boot', default=300, type=int)
    ap.add_argument('--seed', default=0, type=int)
    a = ap.parse_args()
    out = a.out_tsv or str(REPO / f'notes/data/weber_baseline_slopes.ses{a.session}.tsv')
    main(a.bids_folder, a.session, out, a.n_boot, a.seed)
