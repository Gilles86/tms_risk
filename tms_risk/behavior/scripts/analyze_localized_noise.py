"""Is the cTBS effect a *local* increase in randomness rather than a shift in preference?

The paper's psychophysical (probit) model has exactly two knobs per condition:
a slope (choice consistency) and an indifference point (risk attitude). This
script asks whether the empirical cTBS effect can be honestly described by those
two knobs, or whether it is a magnitude-localised increase in decision noise that
the probit is *forced* to re-express as a spurious preference shift.

Outputs (all TSVs into notes/data/):
  localnoise_delta_by_ratio.tsv    observed + probit-predicted DeltaP per ratio bin
  localnoise_delta_by_nrisky.tsv   observed + probit-predicted DeltaP per risky-payoff bin
  localnoise_signatures.tsv        group probit curves: vertex / IPS / slope-only / shift-only
  localnoise_recovery.tsv          probit parameters recovered from data simulated
                                   *without any* preference change
  localnoise_stats.txt             the numbers quoted in the text
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.stats as ss
import arviz as az

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit_probit import build_model, get_data  # noqa: E402

RATIO_BINS = ['20%', '32%', '44%', '56%', '68%', '80%']
# a-priori magnitude bands. nPRF preferred numerosities had IQR [6, 10] (Fig 3B),
# so the effect should live in the smallest payoffs.
NRISKY_EDGES = [0, 17, 26, 36, 53, np.inf]
NRISKY_LABELS = ['7-17', '18-26', '27-36', '37-53', '54-112']


def tidy(df):
    d = df.reset_index().copy()
    d['bin'] = d['bin(risky/safe)'].astype(str)
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['stim'] = d['stimulation_condition']
    d['n_risky_bin'] = pd.cut(d['n_risky'], NRISKY_EDGES, labels=NRISKY_LABELS)
    return d


def paired_delta(d, keys, value='chose_risky'):
    """Within-subject IPS - vertex difference, averaged over subjects."""
    g = (d.groupby(['subject'] + keys + ['stim'], observed=True)[value]
           .mean().unstack('stim'))
    g['delta'] = g['ips'] - g['vertex']
    out = g.groupby(keys, observed=True).agg(
        vertex=('vertex', 'mean'), ips=('ips', 'mean'),
        delta=('delta', 'mean'), sem=('delta', 'sem'), n=('delta', 'count'))
    out['t'] = out['delta'] / out['sem']
    out['p'] = 2 * ss.t.sf(np.abs(out['t']), out['n'] - 1)
    out['ci_lo'] = out['delta'] - ss.t.ppf(.975, out['n'] - 1) * out['sem']
    out['ci_hi'] = out['delta'] + ss.t.ppf(.975, out['n'] - 1) * out['sem']
    return out


def probit_delta(p_trial, d, keys):
    """Posterior of the probit's own predictions, aggregated exactly like the data."""
    idx = pd.MultiIndex.from_frame(d[['subject'] + keys + ['stim']])
    p = p_trial.copy()
    p.index = idx
    pm = p.groupby(level=list(idx.names), observed=True).mean()
    dd = (pm.xs('ips', level='stim') - pm.xs('vertex', level='stim')
          ).groupby(keys, observed=True).mean()
    out = pd.DataFrame({'probit': dd.mean(1),
                        'probit_lo': dd.quantile(.025, axis=1),
                        'probit_hi': dd.quantile(.975, axis=1)})
    # subject-averaged predicted P(risky) per condition, for the psychometric panel
    for stim in ['vertex', 'ips']:
        g = pm.xs(stim, level='stim').groupby(keys, observed=True).mean()
        out[f'probit_{stim}'] = g.mean(1)
        out[f'probit_{stim}_lo'] = g.quantile(.025, axis=1)
        out[f'probit_{stim}_hi'] = g.quantile(.975, axis=1)
    return out


def group_probit_pars(idata, model, df):
    """Group-level intercept / slope of the probit per (order, stimulation)."""
    from utils import extract_intercept_gamma  # noqa
    intercept, gamma = extract_intercept_gamma(idata, model, df, group=True)
    b0 = intercept['intercept'].groupby(['risky_first', 'stimulation_condition']).mean().mean(1)
    b1 = gamma['gamma'].groupby(['risky_first', 'stimulation_condition']).mean().mean(1)
    return b0, b1


def main(bids_folder, out_dir, n_sim=5, seed=1, fit_recovery=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=''):
        print(s)
        lines.append(str(s))

    # ---------------------------------------------------------------- data
    df = get_data('probit_order', bids_folder)
    d = tidy(df)
    say(f'{d.subject.nunique()} subjects, {len(d)} trials')

    # ------------------------------------------------ fitted probit (paper's)
    model = build_model('probit_order', df)
    idata = az.from_netcdf(Path(bids_folder) / 'derivatives' / 'cogmodels' /
                           'model-probit_order_trace.netcdf')
    pred = model.predict(idata, kind='mean', inplace=False)['posterior']['chose_risky_mean']
    p_trial = pred.to_dataframe().unstack(['chain', 'draw'])['chose_risky_mean']
    p_trial.index = df.index

    # ------------------------------------------- 1. delta per ratio bin
    obs_ratio = paired_delta(d, ['order', 'bin'])
    mod_ratio = probit_delta(p_trial, d, ['order', 'bin'])
    ratio = obs_ratio.join(mod_ratio).reindex(
        pd.MultiIndex.from_product([['Risky first', 'Risky second'], RATIO_BINS],
                                   names=['order', 'bin']))
    ratio.to_csv(out_dir / 'localnoise_delta_by_ratio.tsv', sep='\t')
    say('\n=== Delta P(risky) per risky/safe-ratio bin ===')
    say(ratio.round(3).to_string())

    # ------------------------------------------- 2. delta per risky payoff
    obs_n = paired_delta(d, ['order', 'n_risky_bin'])
    mod_n = probit_delta(p_trial, d, ['order', 'n_risky_bin'])
    nr = obs_n.join(mod_n)
    nr.to_csv(out_dir / 'localnoise_delta_by_nrisky.tsv', sep='\t')
    say('\n=== Delta P(risky) per risky-payoff bin ===')
    say(nr.round(3).to_string())

    # smallest-payoff contrast, risky-second trials
    g = (d[d.order == 'Risky second']
         .groupby(['subject', 'n_risky_bin', 'stim'], observed=True)['chose_risky']
         .mean().unstack('stim'))
    g['delta'] = g['ips'] - g['vertex']
    wide = g['delta'].unstack('n_risky_bin')
    contrast = wide[NRISKY_LABELS[0]] - wide[NRISKY_LABELS[1:]].mean(1)
    contrast = contrast.dropna()
    t, p = ss.ttest_1samp(contrast, 0)
    say(f'\nSmallest risky payoffs (7-17 CHF) vs the four larger bands, risky-second:'
        f' delta-of-deltas = {contrast.mean():+.3f} (SEM {contrast.sem():.3f}),'
        f' t({len(contrast)-1}) = {t:.2f}, p = {p:.4f}')

    cell = g.xs(NRISKY_LABELS[0], level='n_risky_bin')
    for c in ['vertex', 'ips']:
        m, s = cell[c].mean(), cell[c].sem()
        say(f'  {c}: P(risky) = {m:.3f} [{m-1.96*s:.3f}, {m+1.96*s:.3f}]')
    t05, p05 = ss.ttest_1samp(cell['ips'].dropna(), 0.5)
    say(f'  IPS vs chance (0.5): t({cell["ips"].notna().sum()-1}) = {t05:.2f}, p = {p05:.3f}')

    # ------------------------------- 3. what the two probit knobs can look like
    b0, b1 = group_probit_pars(idata, model, df)
    xs = np.linspace(d['x'].quantile(.01), d['x'].quantile(.99), 200)
    rows = []
    for rf, order in [(False, 'Risky second'), (True, 'Risky first')]:
        i_v, s_v = b0.loc[(rf, 'vertex')], b1.loc[(rf, 'vertex')]
        i_i, s_i = b0.loc[(rf, 'ips')], b1.loc[(rf, 'ips')]
        x0_v, x0_i = -i_v / s_v, -i_i / s_i
        pv = ss.norm.cdf(i_v + s_v * xs)
        curves = {
            'Vertex': pv,
            'IPS (full probit)': ss.norm.cdf(i_i + s_i * xs),
            'Consistency only': ss.norm.cdf(s_i * (xs - x0_v)),   # slope of IPS, indiff of vertex
            'Preference only': ss.norm.cdf(s_v * (xs - x0_i)),    # slope of vertex, indiff of IPS
        }
        for k, v in curves.items():
            rows.append(pd.DataFrame({'order': order, 'curve': k, 'x': xs,
                                      'p': v, 'delta': v - pv}))
        say(f'\n{order}: vertex slope {s_v:.2f}, RNP {np.exp(i_v/s_v):.3f} | '
            f'IPS slope {s_i:.2f}, RNP {np.exp(i_i/s_i):.3f}')
    sig = pd.concat(rows)
    sig.to_csv(out_dir / 'localnoise_signatures.tsv', sep='\t', index=False)

    # group-level posteriors of slope / RNP, kept as draws (for the recovery panel)
    from utils import extract_intercept_gamma  # noqa
    ic, gm = extract_intercept_gamma(idata, model, df, group=True)
    ic = ic['intercept'].groupby(['risky_first', 'stimulation_condition']).mean()
    gm = gm['gamma'].groupby(['risky_first', 'stimulation_condition']).mean()
    post = pd.concat({'rnp': np.exp(ic / gm), 'slope': gm}, names=['parameter'])
    post = post.stack([0, 1]).rename('value').reset_index()
    post['order'] = post['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    post.to_csv(out_dir / 'localnoise_group_posterior.tsv', sep='\t', index=False)

    # --------------------------- 4. recovery: simulate WITHOUT preference change
    if fit_recovery:
        rec = run_recovery(d, idata, model, df, n_sim=n_sim, seed=seed, say=say, out_dir=out_dir)
        rec.to_csv(out_dir / 'localnoise_recovery.tsv', sep='\t', index=False)

    (out_dir / 'localnoise_stats.txt').write_text('\n'.join(lines))


def fit_hier_probit(sub, draws=1000, tune=1000, seed=1, cores=4):
    """Hierarchical probit: P(risky) = Phi(b0_s + d0_s*IPS + (b1_s + d1_s*IPS) * x).

    Same structure as the paper's bambi model, written out in PyMC so it does not
    depend on the bambi version installed. Returns group-level posterior draws of
    the slope and the risk-neutral probability per stimulation condition.
    """
    import pymc as pm

    sub = sub.reset_index(drop=True)
    s_idx, subjects = pd.factorize(sub['subject'])
    ips = (sub['stim'].values == 'ips').astype(float)
    x = sub['x'].values
    y = sub['chose_risky'].astype(float).values
    ns = len(subjects)

    with pm.Model():
        names = ['b0', 'b1', 'd0', 'd1']
        mu = {n: pm.Normal(f'mu_{n}', 0., 2.5 if n in ('b0', 'd0') else 2.5) for n in names}
        sd = {n: pm.HalfNormal(f'sd_{n}', 1.) for n in names}
        off = {n: pm.Normal(f'off_{n}', 0., 1., shape=ns) for n in names}
        par = {n: mu[n] + sd[n] * off[n] for n in names}

        eta = ((par['b0'][s_idx] + par['d0'][s_idx] * ips)
               + (par['b1'][s_idx] + par['d1'][s_idx] * ips) * x)
        pm.Bernoulli('obs', p=pm.math.invprobit(eta), observed=y)

        pm.Deterministic('slope_vertex', mu['b1'])
        pm.Deterministic('slope_ips', mu['b1'] + mu['d1'])
        pm.Deterministic('rnp_vertex', pm.math.exp(mu['b0'] / mu['b1']))
        pm.Deterministic('rnp_ips', pm.math.exp((mu['b0'] + mu['d0']) / (mu['b1'] + mu['d1'])))

        idata = pm.sample(draws=draws, tune=tune, chains=4, cores=cores,
                          target_accept=0.9, random_seed=seed, progressbar=False)

    post = idata.posterior
    return pd.DataFrame({v: post[v].values.ravel() for v in
                         ['slope_vertex', 'slope_ips', 'rnp_vertex', 'rnp_ips']})


def run_recovery(d, idata, model, df, n_sim=5, seed=1, say=print, out_dir=None):
    """Generate choices in which cTBS changes *only* the noise on small payoffs.

    Ground truth: the indifference point (risk attitude) is IDENTICAL in the two
    stimulation conditions. cTBS multiplies the SD of the decision variable by
    k(n_risky) = 1 + a*exp(-(n_risky - 7)/tau) -- extra randomness that fades out
    as the risky payoff grows, the magnitude dependence the nPRF tuning predicts.
    (a, tau) are calibrated so the simulated change in choice proportions matches
    the observed one; we then apply the paper's read-out (hierarchical probit) and
    ask what it says about "risk attitude".
    """
    import statsmodels.api as sm

    sim0 = d[d.order == 'Risky second'].copy()

    # per-subject generating psychometric function = that subject's vertex fit
    pars = {}
    for subject, g in sim0[sim0.stim == 'vertex'].groupby('subject'):
        X = sm.add_constant(g['x'].values)
        fit = sm.GLM(g['chose_risky'].astype(float).values, X,
                     family=sm.families.Binomial(sm.families.links.Probit())).fit()
        pars[subject] = (fit.params[0], np.clip(fit.params[1], 0.5, 6.0))
    b0 = np.array([pars[s][0] for s in sim0['subject']])
    b1 = np.array([pars[s][1] for s in sim0['subject']])
    z = b0 + b1 * sim0['x'].values
    is_ips = (sim0['stim'].values == 'ips')

    obs_profile = paired_delta(sim0, ['bin'])['delta'].reindex(RATIO_BINS)

    def profile(a, tau):
        k = np.where(is_ips, 1 + a * np.exp(-(sim0['n_risky'].values - 7.) / tau), 1.)
        tmp = sim0.assign(p=ss.norm.cdf(z / k))
        return paired_delta(tmp, ['bin'], value='p')['delta'].reindex(RATIO_BINS)

    best, grid = None, [(a, tau) for a in np.arange(0.25, 6.01, 0.25)
                        for tau in np.arange(4., 60.1, 2.)]
    for a, tau in grid:
        sse = float(((profile(a, tau) - obs_profile) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, a, tau)
    sse, a, tau = best
    say(f'\n=== Noise-only generator calibrated to the observed choice pattern ===')
    say(f'  cTBS multiplies the SD of the decision variable by '
        f'1 + {a:.2f}*exp(-(n_risky - 7)/{tau:.0f}); indifference point unchanged')
    prof = profile(a, tau)
    say('  Delta P(risky) per ratio bin:')
    say('    observed  ' + '  '.join(f'{v:+.3f}' for v in obs_profile))
    say('    simulated ' + '  '.join(f'{v:+.3f}' for v in prof))
    if out_dir is not None:
        prof.rename('delta_sim').to_frame().assign(
            delta_obs=obs_profile, a=a, tau=tau).to_csv(
                Path(out_dir) / 'localnoise_generator_profile.tsv', sep='\t')

    k = np.where(is_ips, 1 + a * np.exp(-(sim0['n_risky'].values - 7.) / tau), 1.)
    p_sim = ss.norm.cdf(z / k)

    real = fit_hier_probit(sim0, seed=seed)
    say('\n=== Hierarchical probit on the REAL risky-second data ===')
    _report(real, say)

    rng = np.random.default_rng(seed)
    sim = sim0.copy()
    out = [real.assign(source='Real data', sim=-1)]
    say('\n=== Same read-out applied to the noise-only simulations ===')
    for i in range(n_sim):
        sim['chose_risky'] = (rng.random(len(sim)) < p_sim).astype(float)
        r = fit_hier_probit(sim, seed=seed + 1 + i)
        say(f'  simulation {i}:')
        _report(r, say, indent='    ')
        out.append(r.assign(source='Simulated (no preference change)', sim=i))
    return pd.concat(out, ignore_index=True)


def _report(r, say, indent='  '):
    drnp = r['rnp_ips'] - r['rnp_vertex']
    dsl = r['slope_ips'] - r['slope_vertex']
    say(f'{indent}RNP {r.rnp_vertex.mean():.3f} -> {r.rnp_ips.mean():.3f}, '
        f'delta {drnp.mean():+.3f} [{drnp.quantile(.025):+.3f}, {drnp.quantile(.975):+.3f}], '
        f'p(delta>0) = {(drnp > 0).mean():.3f}')
    say(f'{indent}slope {r.slope_vertex.mean():.2f} -> {r.slope_ips.mean():.2f}, '
        f'delta {dsl.mean():+.2f} [{dsl.quantile(.025):+.2f}, {dsl.quantile(.975):+.2f}], '
        f'p(delta<0) = {(dsl < 0).mean():.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--out_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--no_recovery', action='store_true')
    args = parser.parse_args()
    main(args.bids_folder, args.out_dir, fit_recovery=not args.no_recovery)
