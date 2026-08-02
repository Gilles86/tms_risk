"""Which channel does the fitted Flexible PMC use to produce its cTBS effect?

The PMC turns exactly one knob for cTBS -- the noise function nu(n) -- but that
knob feeds two channels of the choice rule:

    P(choose 2) = Phi( m / s )
        m  = perceived advantage of option 2   <- the BIAS channel
             (moves because more noise shrinks the percept toward the prior)
        s  = SD of the perceived difference    <- the RANDOMNESS channel

This script takes the fitted `flexible2.6` posterior, evaluates m and s for every
trial under BOTH stimulation conditions, and forms the counterfactuals

    P_vertex     = Phi(m_v / s_v)     P_full     = Phi(m_i / s_i)
    P_bias_only  = Phi(m_i / s_v)     P_noise_only = Phi(m_v / s_i)

so the model's own effect can be split into the part carried by the bias and the
part carried by the randomness, in the same units as the data (Delta P(risky)).

Implementation note: rather than re-deriving the model's algebra by hand, we let
bauer build its own graph and simply tap `get_diff_dist`, which is where m and s
are produced. The paradigm is doubled (every trial once as vertex, once as IPS)
so both counterfactuals exist for the same trial, and so patsy still sees both
levels of `stimulation_condition` and builds the same 2-column design matrix the
model was fitted with.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss

# IMPORTANT: bauer's likelihood has drifted since these traces were fitted.
# `flexible2.6_trace.netcdf` was written 2024-11-05; commit b66c806 (2026-04-03)
# changed the 'payoff' branch of _get_choice_predictions from
#     diff_sd = sqrt(n1_evidence_sd**2 + n2_evidence_sd**2)
# to  diff_sd = sqrt((post_sd**2/evidence_sd * p)**2 summed over options)
# without renaming a single parameter, so the stored posterior loads cleanly into
# the new graph and silently predicts something else. Point --bauer_path at a
# worktree of the contemporary commit (a717fa3); the PPC check below is what
# actually verifies you got the right one.
REPO = Path(__file__).resolve().parents[3]
_bauer_path = None
for i, a in enumerate(sys.argv):
    if a == '--bauer_path':
        _bauer_path = sys.argv[i + 1]
sys.path.insert(0, _bauer_path or str(REPO / 'libs' / 'bauer'))
sys.path.insert(0, str(REPO / 'tms_risk' / 'behavior'))

import arviz as az           # noqa: E402
import pymc as pm            # noqa: E402
import bauer                 # noqa: E402
import bauer.models as bm    # noqa: E402
from fit_model import get_data   # noqa: E402

try:                                     # module layout differs across versions
    import bauer.models.risky_choice as rc
except ImportError:
    rc = bm

RATIO_BINS = ['20%', '32%', '44%', '56%', '68%', '80%']
NRISKY_EDGES = [0, 17, 26, 36, 53, np.inf]
NRISKY_LABELS = ['7-17', '18-26', '27-36', '37-53', '54-112']


def build_flexible(df, spline_order=6, family=2):
    """Build a flexible-noise PMC.

    family=2 (`flexible2*`): memory_model='shared_perceptual_noise' -- nu1 is built
        from memory + perceptual coefficients, nu2 from perceptual alone.
    family=1 (`flexible1*`): memory_model='independent' -- nu1 and nu2 are separate
        spline functions. Note this is an exact reparameterisation of family 2
        (c1 = mem + perc, c2 = perc), so the likelihood is identical; only the
        prior coordinates and the sampling geometry differ.

    Constructed directly rather than via fit_model.build_model because the
    keyword was renamed (polynomial_order -> spline_order) after these fits.
    """
    names = (['memory_noise_sd', 'perceptual_noise_sd'] if family == 2
             else ['n1_evidence_sd', 'n2_evidence_sd'])
    regressors = {n: 'stimulation_condition' for n in names}
    kw = dict(regressors=regressors,
              memory_model='shared_perceptual_noise' if family == 2 else 'independent',
              prior_estimate='full')
    cls = bm.FlexibleNoiseRiskRegressionModel
    import inspect
    key = ('spline_order' if 'spline_order' in inspect.signature(cls).parameters
           else 'polynomial_order')
    kw[key] = spline_order
    return cls(df, **kw)


def tap_get_diff_dist():
    """Expose the model's own m and s as Deterministics, without changing algebra."""
    original = rc.get_diff_dist

    def tapped(mu1, sd1, mu2, sd2):
        diff_mu, diff_sd = original(mu1, sd1, mu2, sd2)
        pm.Deterministic('diff_mu', diff_mu)
        pm.Deterministic('diff_sd', diff_sd)
        return diff_mu, diff_sd

    rc.get_diff_dist = tapped
    return original


def tap_get_posterior():
    """Expose each option's posterior-mean percept. `_get_choice_predictions` calls
    get_posterior for n1 first and n2 second, so the call order names them."""
    original = rc.get_posterior
    calls = {'n': 0}

    def tapped(*a):
        mu, sd = original(*a)
        calls['n'] += 1
        if calls['n'] <= 2:
            pm.Deterministic(f'post_mu_{calls["n"]}', mu)
        return mu, sd

    rc.get_posterior = tapped
    return original


def double_paradigm(df):
    """Every trial twice: once labelled vertex, once labelled IPS."""
    out = []
    for cond in ['vertex', 'ips']:
        d = df.copy()
        d['stimulation_condition'] = cond
        d['counterfactual'] = cond
        out.append(d)
    return pd.concat(out)


def main(bids_folder, out_dir, model_label='flexible2.6', n_draws=200, seed=1,
         trace_dir=None, tag=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=''):
        print(s, flush=True)
        lines.append(str(s))

    say(f'bauer from {Path(bauer.__file__).parent}')

    df = get_data(bids_folder)
    say(f'{df.index.get_level_values("subject").nunique()} subjects, {len(df)} trials')

    tdir = Path(trace_dir) if trace_dir else Path(bids_folder) / 'derivatives' / 'cogmodels'
    idata = az.from_netcdf(tdir / f'model-{model_label}_trace.netcdf')

    doubled = double_paradigm(df)
    tap_get_diff_dist()
    tap_get_posterior()
    # bare `flexible2` is 5 splines -- the paper's model; `.6` is a later variant.
    # A trailing `_noisefix[.variant]` marks a refit; it does not change the graph.
    m = re.fullmatch(r'flexible([12])(\.\d)?(_noisefix)?(\.\w+)?', model_label)
    if not m:
        raise SystemExit(f'unsupported label {model_label!r}')
    family = int(m.group(1))
    spline_order = 5 if m.group(2) is None else int(m.group(2)[1:])
    model_label = tag or model_label
    say(f'family {family} ({"independent" if family == 1 else "shared perceptual"}), '
        f'spline order {spline_order}')
    model = build_flexible(doubled.copy(), spline_order=spline_order, family=family)
    model.build_estimation_model(save_p_choice=True)
    pm_model = model.estimation_model

    # sanity: the rebuilt graph must line up with the stored posterior
    trace_subjects = np.asarray(idata.posterior.coords['subject'].values)
    model_subjects = np.asarray(pm_model.coords['subject'])
    assert np.array_equal(trace_subjects, model_subjects), 'subject coords differ'
    reg = next(c for c in idata.posterior.coords if c.endswith('_spline1_regressors'))
    assert list(idata.posterior.coords[reg].values) == list(pm_model.coords[reg]), \
        'regressor coding differs'
    say(f'graph matches trace: {len(model_subjects)} subjects, '
        f'regressors {list(pm_model.coords[reg])}')

    # thin on the draw axis only -- stacking and unstacking a random subset
    # reintroduces the full chain x draw grid and fills the gaps with NaN.
    n_chain, n_draw = idata.posterior.sizes['chain'], idata.posterior.sizes['draw']
    per_chain = max(1, n_draws // n_chain)
    keep_draws = np.linspace(0, n_draw - 1, per_chain).astype(int)
    thin = idata.posterior.isel(draw=keep_draws)
    say(f'evaluating {n_chain} x {per_chain} = {n_chain * per_chain} posterior draws '
        f'on {len(doubled)} rows')

    det = pm.compute_deterministics(thin, model=pm_model,
                                    var_names=['diff_mu', 'diff_sd', 'p',
                                               'post_mu_1', 'post_mu_2'],
                                    merge_dataset=False, progressbar=False)

    # P(choose option 2) = cumulative_normal(0, diff_mu, diff_sd) = Phi(-diff_mu/diff_sd),
    # so the perceived advantage of option 2 is m = -diff_mu.
    m = -det['diff_mu'].stack(sample=('chain', 'draw')).values     # (trial, draw)
    s = det['diff_sd'].stack(sample=('chain', 'draw')).values
    p_model = det['p'].stack(sample=('chain', 'draw')).values

    bad = ~np.isfinite(m) | ~np.isfinite(s) | ~np.isfinite(p_model)
    keep = ~bad.any(0)
    if bad.any():
        say(f'dropping {(~keep).sum()}/{bad.shape[1]} draws with non-finite values '
            f'({bad.any(1).sum()} rows affected)')
        m, s, p_model = m[:, keep], s[:, keep], p_model[:, keep]

    # validate the sign convention against the model's own p_choice
    err = np.abs(ss.norm.cdf(m / s) - p_model).max()
    say(f'reconstruction check: max |Phi(m/s) - p_model| = {err:.2e}')
    assert err < 1e-6, 'sign/scale convention does not reproduce the model p'

    n = len(df)
    m_v, m_i = m[:n], m[n:]
    s_v, s_i = s[:n], s[n:]

    # option 2 is the risky one when the safe option came first
    risky_second = (~df['risky_first'].values)[:, None]
    def p_risky(mu, sd):
        p2 = ss.norm.cdf(mu / sd)
        return np.where(risky_second, p2, 1 - p2)

    # --- split the bias channel by option -------------------------------------
    # m = EV2_perceived - EV1_perceived, with EVk = post_mu_k * pk. So the bias
    # channel can be attributed to the risky option's percept moving, the safe
    # option's percept moving, or both.
    def ev(which, draws):
        """Perceived expected value of the risky / safe option, per trial."""
        mu1 = det[f'post_mu_1'].stack(sample=('chain', 'draw')).values[draws]
        mu2 = det[f'post_mu_2'].stack(sample=('chain', 'draw')).values[draws]
        ev1 = mu1 * df['p1'].values[:, None]
        ev2 = mu2 * df['p2'].values[:, None]
        rf = df['risky_first'].values[:, None]
        if which == 'risky':
            return np.where(rf, ev1, ev2)
        return np.where(rf, ev2, ev1)

    sl_v, sl_i = slice(0, n), slice(n, 2 * n)
    ev_risky_v, ev_risky_i = ev('risky', sl_v), ev('risky', sl_i)
    ev_safe_v, ev_safe_i = ev('safe', sl_v), ev('safe', sl_i)
    if bad.any():
        ev_risky_v, ev_risky_i = ev_risky_v[:, keep], ev_risky_i[:, keep]
        ev_safe_v, ev_safe_i = ev_safe_v[:, keep], ev_safe_i[:, keep]

    sign = np.where(df['risky_first'].values[:, None], -1.0, 1.0)  # m is EV2 - EV1

    def p_from(evr, evs, sd):
        return ss.norm.cdf(sign * (evr - evs) / sd * sign)   # = Phi((evr-evs)/sd)

    preds = {
        'vertex': p_risky(m_v, s_v),
        'full': p_risky(m_i, s_i),
        'bias_only': p_risky(m_i, s_v),
        'noise_only': p_risky(m_v, s_i),
        'bias_risky_percept_only': ss.norm.cdf((ev_risky_i - ev_safe_v) / s_v),
        'bias_safe_percept_only': ss.norm.cdf((ev_risky_v - ev_safe_i) / s_v),
    }
    say('\n=== is the risky option pulled up, or the safe option pulled down? ===')
    say('    (risky-second trials; perceived EV in CHF, objective payoff for reference)')
    mask = (~df['risky_first'].values)
    for nm, obj, pv, pi in [('risky option', df.loc[~df['risky_first'], 'n_risky'].values * 0.55,
                             ev_risky_v[mask], ev_risky_i[mask]),
                            ('safe option', df.loc[~df['risky_first'], 'n_safe'].values,
                             ev_safe_v[mask], ev_safe_i[mask])]:
        shift = (pi - pv).mean(0)
        say(f'  {nm:13s} objective {obj.mean():6.2f} | vertex {pv.mean():6.2f} '
            f'({100*(pv.mean()/obj.mean()-1):+5.1f}%) -> IPS {pi.mean():6.2f}  '
            f'delta {shift.mean():+.3f} [{np.quantile(shift,.025):+.3f}, {np.quantile(shift,.975):+.3f}]')

    # validation against the model's own p_choice under the real design
    if 'p' in det:
        say('note: p_choice deterministic also available')

    # Per safe-payoff level: shrinkage toward the fitted safe prior pulls payoffs
    # ABOVE the prior mean down and payoffs BELOW it up, so the direction of the
    # cTBS effect on the safe percept should flip at the prior mean.
    sp = float(np.mean(idata.posterior['safe_prior_mu'].values))
    subj_sp = idata.posterior['safe_prior_mu'].mean(('chain', 'draw')).values.ravel()
    say(f'\n=== fitted safe prior mean: group {sp:.2f} CHF; per subject '
        f'median {np.median(subj_sp):.2f}, '
        f'{(subj_sp < 7).sum()}/{len(subj_sp)} below 7, '
        f'{(subj_sp < 10).sum()}/{len(subj_sp)} below 10 ===')
    say('\n=== perceived EV per objective safe payoff (risky-second trials, CHF) ===')
    say(f'  {"n_safe":>6} {"safe: obj":>10} {"percept":>8} {"bias":>6} {"cTBS d":>18} '
        f'| {"risky: obj":>10} {"percept":>8} {"bias":>6} {"cTBS d":>18}')
    ns = df.loc[~df['risky_first'], 'n_safe'].values
    nr = df.loc[~df['risky_first'], 'n_risky'].values
    for lvl in sorted(np.unique(ns)):
        sel = ns == lvl
        row = f'  {lvl:6.0f}'
        for obj, pv, pi in [(lvl, ev_safe_v[mask][sel], ev_safe_i[mask][sel]),
                            (0.55 * nr[sel].mean(), ev_risky_v[mask][sel], ev_risky_i[mask][sel])]:
            dl = (pi - pv).mean(0)
            arrow = 'up' if pv.mean() > obj else 'down'
            row += (f' {obj:10.2f} {pv.mean():8.2f} {arrow:>6} '
                    f'{dl.mean():+6.3f} [{np.quantile(dl,.025):+.3f},{np.quantile(dl,.975):+.3f}]')
            row += ' |' if obj == lvl else ''
        say(row)
    # source data for the mechanism illustration
    rows = []
    for lvl in sorted(np.unique(ns)):
        sel = ns == lvl
        for opt, obj, pv, pi in [
                ('safe', float(lvl), ev_safe_v[mask][sel], ev_safe_i[mask][sel]),
                ('risky', 0.55 * nr[sel].mean(), ev_risky_v[mask][sel], ev_risky_i[mask][sel])]:
            dl = (pi - pv).mean(0)
            rows.append({'n_safe': lvl, 'option': opt, 'objective_ev': obj,
                         'vertex': pv.mean(), 'ips': pi.mean(),
                         'delta': dl.mean(), 'lo': np.quantile(dl, .025),
                         'hi': np.quantile(dl, .975)})
    pd.DataFrame(rows).to_csv(
        out_dir / f'pmc_percepts.{model_label}.tsv', sep='\t', index=False)

    # The same percepts, but split by PRESENTATION ORDER -- this is the quantity the
    # order-specificity of the behavioural effect rests on. The first-presented option
    # carries memory noise on top of the shared perceptual noise, so it sits further
    # along the shrinkage curve; adding perceptual noise there costs it more. Whether
    # the safe option is devalued more when it comes first is therefore a direct,
    # falsifiable prediction, and this table is what tests it.
    rows = []
    for oname, omask in [('Risky first', df['risky_first'].values),
                         ('Risky second', ~df['risky_first'].values)]:
        ns_o = df.loc[omask, 'n_safe'].values
        nr_o = df.loc[omask, 'n_risky'].values
        for lvl in sorted(np.unique(ns_o)):
            sel = ns_o == lvl
            for opt, obj, pv, pi in [
                    ('safe', float(lvl),
                     ev_safe_v[omask][sel], ev_safe_i[omask][sel]),
                    ('risky', 0.55 * nr_o[sel].mean(),
                     ev_risky_v[omask][sel], ev_risky_i[omask][sel])]:
                dl = (pi - pv).mean(0)
                # `position` is what actually drives the prediction: the safe option is
                # presented first exactly when the risky option is presented second.
                position = ('first' if ((opt == 'safe') == (oname == 'Risky second'))
                            else 'second')
                rows.append({'order': oname, 'n_safe': lvl, 'option': opt,
                             'position': position, 'objective_ev': obj,
                             'vertex': pv.mean(), 'ips': pi.mean(),
                             'delta': dl.mean(), 'lo': np.quantile(dl, .025),
                             'hi': np.quantile(dl, .975),
                             'p_decrease': float((dl < 0).mean())})
    by_order = pd.DataFrame(rows)
    by_order.to_csv(out_dir / f'pmc_percepts_by_order.{model_label}.tsv',
                    sep='\t', index=False)
    say('\n=== cTBS effect on the perceived value of each option, by position (CHF) ===')
    say(by_order.pivot_table(index=['option', 'position'], values='delta',
                             aggfunc='mean').round(3).to_string())

    # tidy trial table used for every aggregation below
    d = df.reset_index().copy()
    d['bin'] = d['bin(risky/safe)'].astype(str)
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['n_risky_bin'] = pd.cut(d['n_risky'], NRISKY_EDGES, labels=NRISKY_LABELS)
    d['stim'] = d['stimulation_condition'].values

    def profile(keys, labels=None):
        """Delta P(risky) attributable to each channel, aggregated like the data."""
        rows = []
        for name in ['full', 'bias_only', 'noise_only',
                     'bias_risky_percept_only', 'bias_safe_percept_only']:
            delta = np.asarray(preds[name]) - np.asarray(preds['vertex'])
            assert delta.ndim == 2, (name, delta.shape)
            tmp = d.copy()
            per_draw = []
            for j in range(delta.shape[1]):
                tmp['v'] = delta[:, j]
                per_draw.append(tmp.groupby(keys, observed=True)['v'].mean())
            per_draw = pd.concat(per_draw, axis=1)
            rows.append(pd.DataFrame({
                'channel': name,
                'delta': per_draw.mean(1),
                'lo': per_draw.quantile(.025, axis=1),
                'hi': per_draw.quantile(.975, axis=1)}))
        out = pd.concat(rows).reset_index()
        if labels is not None:
            out[keys[-1]] = pd.Categorical(out[keys[-1]], labels, ordered=True)
        return out.sort_values(['channel'] + keys)

    by_ratio = profile(['order', 'bin'], RATIO_BINS)
    by_ratio.to_csv(out_dir / f'pmc_channels_by_ratio.{model_label}.tsv', sep='\t', index=False)
    say('\n=== Delta P(risky) predicted by the fitted Flexible PMC, per channel ===')
    say(by_ratio.pivot_table(index=['order', 'bin'], columns='channel',
                             values='delta', observed=True).round(3).to_string())

    # Same breakdown on the SAFE payoff axis, next to the observed effect. A shift in
    # perceived CHF is not the same thing as a shift in choice probability: what matters
    # is the shift relative to the decision variable's spread, which itself grows with
    # magnitude. So the channel that moves most in CHF need not move choices most.
    from tms_risk.behavior.scripts.analyze_localized_noise import paired_delta  # noqa
    dd = d.copy()
    dd['stim'] = dd['stimulation_condition'].values
    obs_safe = paired_delta(dd[dd.order == 'Risky second'], ['n_safe'])
    by_safe = profile(['order', 'n_safe'])
    say('\n=== Delta P(risky) by SAFE payoff: observed vs model channels (risky second) ===')
    tab = (by_safe[by_safe.order == 'Risky second']
           .pivot_table(index='n_safe', columns='channel', values='delta', observed=True))
    tab.insert(0, 'observed', obs_safe['delta'])
    tab.insert(1, 'obs_ci', obs_safe.apply(
        lambda r: f"[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]", axis=1))
    say(tab[['observed', 'obs_ci', 'full', 'bias_only',
             'bias_safe_percept_only', 'bias_risky_percept_only',
             'noise_only']].round(3).to_string())
    by_safe.to_csv(out_dir / f'pmc_channels_by_nsafe.{model_label}.tsv', sep='\t', index=False)

    by_payoff = profile(['order', 'n_risky_bin'], NRISKY_LABELS)
    by_payoff.to_csv(out_dir / f'pmc_channels_by_nrisky.{model_label}.tsv', sep='\t', index=False)
    say('\n=== the same, by risky payoff ===')
    say(by_payoff.pivot_table(index=['order', 'n_risky_bin'], columns='channel',
                              values='delta', observed=True).round(3).to_string())

    # overall size of each channel, risky-second trials
    say('\n=== mean |Delta P| over risky-second trials ===')
    mask = (d['order'] == 'Risky second').values
    for name in ['full', 'bias_only', 'noise_only',
                 'bias_risky_percept_only', 'bias_safe_percept_only']:
        delta = (np.asarray(preds[name]) - np.asarray(preds['vertex']))[mask]
        per_draw_mean = delta.mean(0)
        per_draw_abs = np.abs(delta).mean(0)
        say(f'  {name:11s} mean Delta = {per_draw_mean.mean():+.4f} '
            f'[{np.quantile(per_draw_mean, .025):+.4f}, {np.quantile(per_draw_mean, .975):+.4f}]'
            f'   mean |Delta| = {per_draw_abs.mean():.4f}')

    (out_dir / f'pmc_channels_stats.{model_label}.txt').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--out_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--model_label', default='flexible2.6')
    parser.add_argument('--n_draws', default=200, type=int)
    parser.add_argument('--bauer_path', default=None,
                        help='path to a bauer checkout contemporary with the trace')
    parser.add_argument('--trace_dir', default=None)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()
    main(args.bids_folder, args.out_dir, args.model_label, args.n_draws,
         trace_dir=args.trace_dir, tag=args.tag)
