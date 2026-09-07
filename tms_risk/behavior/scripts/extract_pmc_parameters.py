"""Extract every group-level parameter of the published Flexible PMC (`flexible2`).

Writes three source-data files into notes/data/:
  pmcpars_priors.<label>.tsv     prior mu / sd for risky and safe options, group + per subject
  pmcpars_splines.<label>.tsv    the 5 x 2 x 2 spline coefficients (forest-plot input)
  pmcpars_curves.<label>.tsv     the noise functions nu(n) per condition, in CHF

The noise curves are rebuilt with patsy directly rather than via bauer's
`get_sd_curve`, because the knot anchoring changed after these fits: `ecc6454` built
the basis per call over min/max of *both* n1 and n2, while later versions fix
`design_info` at construction anchored to one paradigm column. Using the ecc6454
formula here is what reproduces the published numbers (see CLAUDE.md).
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from patsy import dmatrix

from tms_risk.behavior.scripts.noise_curve_inference import (basis as anchored_basis,
                                                             paradigm_columns)

import sys
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'libs' / 'bauer'))
sys.path.insert(0, str(REPO / 'tms_risk' / 'behavior'))
from fit_model import get_data   # noqa: E402

PRIORS = ['risky_prior_mu', 'risky_prior_sd', 'safe_prior_mu', 'safe_prior_sd']
NOISE_TERMS = {2: ['memory_noise_sd', 'perceptual_noise_sd'],
               1: ['n1_evidence_sd', 'n2_evidence_sd']}
TERM_LABEL = {'memory_noise_sd': 'Memory', 'perceptual_noise_sd': 'Perceptual',
              'n1_evidence_sd': 'First-presented option',
              'n2_evidence_sd': 'Second-presented option'}


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def coef_by_condition(coefs):
    """(sample, regressor, spline) -> {'ips': ..., 'vertex': ...}.

    Regressors are ['Intercept', 'stimulation_condition[T.vertex]'], so IPS is the
    reference level and vertex = Intercept + contrast. A model that does not put the
    stimulation regressor on this term has a single column, and the two conditions
    are then the same curve by construction.
    """
    ips = coefs[:, 0, :]
    vertex = ips + coefs[:, 1, :] if coefs.shape[1] > 1 else ips
    return {'ips': ips, 'vertex': vertex}


def basis(x, df_, lower, upper):
    """The ecc6454 spline basis: cubic B-spline over the full payoff range."""
    return np.asarray(dmatrix(
        f'bs(x, degree=3, df={df_}, include_intercept=True, '
        f'lower_bound={lower}, upper_bound={upper}) - 1', {'x': x}))


def main(bids_folder, out_dir, label, spline_order, trace_dir=None, tag=None):
    import re
    family = int(re.match(r'flexible([12])', label).group(1))
    terms = NOISE_TERMS[family]
    out_dir = Path(out_dir)
    tdir = Path(trace_dir) if trace_dir else Path(bids_folder) / 'derivatives' / 'cogmodels'
    idata = az.from_netcdf(tdir / f'model-{label}_trace.netcdf')
    label = tag or label
    post = idata.posterior
    family = int(post.attrs.get('tms_risk_family', family))
    terms = NOISE_TERMS[family]
    spline_order = int(post.attrs.get('tms_risk_spline_order', spline_order)) or spline_order
    # How nu_1 is built from the two noise terms. The default wraps the softplus around
    # the SUM, so the memory contribution nu_1 - nu_2 is unconstrained in sign; the
    # `additive` variant adds two softplus outputs, pinning it at >= 0. Reconstructing a
    # trace with the wrong one is silent -- every parameter is present either way -- so
    # it is read from the trace, never assumed.
    composition = post.attrs.get('tms_risk_memory_composition', 'sum_then_softplus')
    df = get_data(bids_folder)
    lower = float(df[['n1', 'n2']].min().min())
    upper = float(df[['n1', 'n2']].max().max())
    print(f'{label}: payoff range [{lower:.0f}, {upper:.0f}], {spline_order} splines')

    # ---------------------------------------------------------------- priors
    rows = []
    for v in PRIORS:
        grp = post[f'{v}_mu'].values.ravel()                       # group mean
        subj = post[v].mean(('chain', 'draw')).values.ravel()       # per-subject means
        rows.append({'parameter': v, 'level': 'group',
                     'mean': grp.mean(), 'lo': np.quantile(grp, .025),
                     'hi': np.quantile(grp, .975),
                     'p_gt0': float((grp > 0).mean())})
        for s, val in zip(post.coords['subject'].values, subj):
            rows.append({'parameter': v, 'level': f'sub-{s:02d}', 'mean': val})
    priors = pd.DataFrame(rows)
    priors.to_csv(out_dir / f'pmcpars_priors.{label}.tsv', sep='\t', index=False)
    print('\n=== prior parameters (group posterior, CHF) ===')
    g = priors[priors.level == 'group']
    for _, r in g.iterrows():
        print(f'  {r.parameter:16s} {r["mean"]:7.2f} [{r.lo:6.2f}, {r.hi:6.2f}]')

    # -------------------------------------------------------------- splines
    rows = []
    for term in terms:
        for i in range(1, spline_order + 1):
            v = f'{term}_spline{i}_mu'
            regs = list(post.coords[f'{term}_spline{i}_regressors'].values)
            for j, reg in enumerate(regs):
                x = post[v].values[..., j].ravel()
                rows.append({'term': term, 'spline': i, 'regressor': reg,
                             'mean': x.mean(), 'lo': np.quantile(x, .025),
                             'hi': np.quantile(x, .975),
                             'p_gt0': float((x > 0).mean())})
    splines = pd.DataFrame(rows)
    splines.to_csv(out_dir / f'pmcpars_splines.{label}.tsv', sep='\t', index=False)
    n_cred = ((splines.p_gt0 > .975) | (splines.p_gt0 < .025)).sum()
    print(f'\n=== spline coefficients: {n_cred}/{len(splines)} with 95% CrI excluding 0 ===')
    stim = splines[splines.regressor != 'Intercept']
    print('  stimulation contrasts (vertex - IPS):')
    for _, r in stim.iterrows():
        flag = ' *' if (r.p_gt0 > .975 or r.p_gt0 < .025) else ''
        print(f'    {r.term:20s} spline{r.spline}  {r["mean"]:+7.3f} '
              f'[{r.lo:+.3f}, {r.hi:+.3f}]{flag}')

    # --------------------------------------------------------- noise curves
    # regressors are ['Intercept', 'stimulation_condition[T.vertex]'], so IPS is the
    # reference level and vertex = Intercept + contrast.
    # patsy puts the single interior knot at the median of whatever x it is handed,
    # so the basis has to be anchored to the paradigm column the model was fitted
    # on -- not to the plotting grid. n1/memory <- n1, n2/perceptual <- n2.
    n1_col, n2_col = paradigm_columns(bids_folder)
    anchor = {'n1_evidence_sd': n1_col, 'memory_noise_sd': n1_col,
              'n2_evidence_sd': n2_col, 'perceptual_noise_sd': n2_col}
    xs = np.linspace(lower, upper, 120)
    B = {t: anchored_basis(xs, spline_order, lower, upper, anchor=anchor[t])
         for t in terms}
    rows = []
    for term in terms:
        coefs = np.stack([post[f'{term}_spline{i}_mu'].values for i in
                          range(1, spline_order + 1)], -1)         # chain, draw, reg, spline
        coefs = coefs.reshape(-1, coefs.shape[-2], coefs.shape[-1])   # sample, reg, spline
        for cond, c in coef_by_condition(coefs).items():
            nu = softplus(c @ B[term].T)                            # sample x x
            rows.append(pd.DataFrame({
                'term': term, 'stimulation': cond, 'payoff': xs,
                'nu': nu.mean(0), 'lo': np.quantile(nu, .025, axis=0),
                'hi': np.quantile(nu, .975, axis=0)}))
        # IPS - vertex difference, propagated through the same draws
        cc = coef_by_condition(coefs)
        nu_i = softplus(cc['ips'] @ B[term].T)
        nu_v = softplus(cc['vertex'] @ B[term].T)
        dnu = nu_i - nu_v
        rows.append(pd.DataFrame({
            'term': term, 'stimulation': 'ips - vertex', 'payoff': xs,
            'nu': dnu.mean(0), 'lo': np.quantile(dnu, .025, axis=0),
            'hi': np.quantile(dnu, .975, axis=0)}))
    if family == 2:
        # Emit the per-position curves too, so downstream figures never have to
        # recompose them. Under the default composition bauer builds
        # nu_1 = softplus(eta_mem + eta_perc) -- the softplus wraps the SUM -- and
        # nu_2 = softplus(eta_perc); under `additive` it is nu_1 = nu_2 + softplus(eta_mem).
        def compose_n1(cd):
            m, p = eta[('memory_noise_sd', cd)], eta[('perceptual_noise_sd', cd)]
            return softplus(p) + softplus(m) if composition == 'additive' else softplus(m + p)

        def compose_n2(cd):
            return softplus(eta[('perceptual_noise_sd', cd)])

        eta = {}
        for term in terms:
            c = np.stack([post[f'{term}_spline{i}_mu'].values for i in
                          range(1, spline_order + 1)], -1)
            c = c.reshape(-1, c.shape[-2], c.shape[-1])
            cc = coef_by_condition(c)
            eta[(term, 'ips')] = cc['ips'] @ B[term].T
            eta[(term, 'vertex')] = cc['vertex'] @ B[term].T
        for key, nu_of in [('n1_evidence_sd', compose_n1),
                           ('n2_evidence_sd', compose_n2)]:
            nu = {cd: nu_of(cd) for cd in ['ips', 'vertex']}
            for cd in ['ips', 'vertex']:
                rows.append(pd.DataFrame({
                    'term': key, 'stimulation': cd, 'payoff': xs,
                    'nu': nu[cd].mean(0), 'lo': np.quantile(nu[cd], .025, axis=0),
                    'hi': np.quantile(nu[cd], .975, axis=0)}))
            d_ = nu['ips'] - nu['vertex']
            rows.append(pd.DataFrame({
                'term': key, 'stimulation': 'ips - vertex', 'payoff': xs,
                'nu': d_.mean(0), 'lo': np.quantile(d_, .025, axis=0),
                'hi': np.quantile(d_, .975, axis=0)}))
        # The memory CONTRIBUTION, nu_1 - nu_2, propagated through the same draws so it
        # carries a real credible interval. It cannot be recovered from the marginal
        # intervals of nu_1 and nu_2, and its sign is the substantive question: the
        # default composition does not constrain the contribution to be positive, and
        # its sign is the substantive question. Under `additive` it is >= 0 by
        # construction, so a non-negative curve there is a property of the model, not a
        # finding -- which is exactly why the two are compared by ELPD.
        for cd in ['ips', 'vertex']:
            g = compose_n1(cd) - compose_n2(cd)
            rows.append(pd.DataFrame({
                'term': 'memory_contribution', 'stimulation': cd, 'payoff': xs,
                'nu': g.mean(0), 'lo': np.quantile(g, .025, axis=0),
                'hi': np.quantile(g, .975, axis=0), 'p_pos': (g > 0).mean(0)}))

    if family == 1:
        # Family 1 fits the two positions as INDEPENDENT spline functions, so the
        # memory contribution is just nu_1 - nu_2 and nothing in the parameterisation
        # constrains its sign. Emitting it here makes family 1 and family 2 directly
        # comparable on the substantive question -- is the first-presented option
        # encoded MORE precisely than the second at small payoffs? -- and family 1 is
        # the stronger test, because it has no memory/perceptual composition that
        # could be manufacturing the answer.
        # It cannot be read off the marginal intervals of nu_1 and nu_2: the two share
        # subject-level and spline structure, so their draws are correlated and the
        # difference is far better determined than either curve alone.
        nu_by = {}
        for term in ['n1_evidence_sd', 'n2_evidence_sd']:
            c = np.stack([post[f'{term}_spline{i}_mu'].values for i in
                          range(1, spline_order + 1)], -1)
            c = c.reshape(-1, c.shape[-2], c.shape[-1])
            cc = coef_by_condition(c)
            for cd in ['ips', 'vertex']:
                nu_by[(term, cd)] = softplus(cc[cd] @ B[term].T)
        for cd in ['ips', 'vertex']:
            g = nu_by[('n1_evidence_sd', cd)] - nu_by[('n2_evidence_sd', cd)]
            rows.append(pd.DataFrame({
                'term': 'memory_contribution', 'stimulation': cd, 'payoff': xs,
                'nu': g.mean(0), 'lo': np.quantile(g, .025, axis=0),
                'hi': np.quantile(g, .975, axis=0), 'p_pos': (g > 0).mean(0)}))

    curves = pd.concat(rows)
    curves.to_csv(out_dir / f'pmcpars_curves.{label}.tsv', sep='\t', index=False)

    # Relative (proportional) cTBS effect, nu_ips/nu_vertex - 1, propagated through the
    # same draws so it carries a real credible interval. This is the scale the paper's
    # magnitude-specificity claim lives on: a roughly constant *absolute* noise
    # injection is a much larger *proportional* degradation where baseline noise is
    # small, and baseline noise grows with magnitude.
    rel_rows = []
    for term in terms:
        coefs = np.stack([post[f'{term}_spline{i}_mu'].values for i in
                          range(1, spline_order + 1)], -1)
        coefs = coefs.reshape(-1, coefs.shape[-2], coefs.shape[-1])
        cc = coef_by_condition(coefs)
        nu_i = softplus(cc['ips'] @ B[term].T)
        nu_v = softplus(cc['vertex'] @ B[term].T)
        rel = 100. * (nu_i / nu_v - 1.)
        rel_rows.append(pd.DataFrame({
            'term': term, 'payoff': xs, 'pct': rel.mean(0),
            'lo': np.quantile(rel, .025, axis=0), 'hi': np.quantile(rel, .975, axis=0),
            'p_increase': (rel > 0).mean(0)}))
    pd.DataFrame(pd.concat(rel_rows)).to_csv(
        out_dir / f'pmcpars_relative.{label}.tsv', sep='\t', index=False)

    print('\n=== noise increase after cTBS (IPS - vertex), at selected payoffs ===')
    d = curves[curves.stimulation == 'ips - vertex']
    for term in terms:
        sub = d[d.term == term]
        picks = [7, 10, 14, 20, 28, 56, 112]
        vals = [sub.iloc[(sub.payoff - v).abs().argmin()] for v in picks]
        txt = '  '.join(f'{v:3.0f}:{r.nu:+.2f}' + ('*' if r.lo > 0 or r.hi < 0 else ' ')
                        for v, r in zip(picks, vals))
        print(f'  {term:20s} {txt}')
    print('\n(* = 95% CrI excludes 0)')

    # posterior probabilities, per payoff, for the cTBS contrast on each noise term
    rows = []
    picks = [7, 10, 14, 20, 28, 56, 112]
    print('\n=== posterior probability that cTBS INCREASED noise, p(IPS > vertex) ===')
    print('  payoff              ' + '  '.join(f'{v:6.0f}' for v in picks))
    Bp = {t: anchored_basis(np.array(picks, float), spline_order, lower, upper,
                            anchor=anchor[t]) for t in terms}
    for term in terms:
        coefs = np.stack([post[f'{term}_spline{i}_mu'].values for i in
                          range(1, spline_order + 1)], -1)
        coefs = coefs.reshape(-1, coefs.shape[-2], coefs.shape[-1])
        cc = coef_by_condition(coefs)
        dnu = (softplus(cc['ips'] @ Bp[term].T)
               - softplus(cc['vertex'] @ Bp[term].T))
        pg = (dnu > 0).mean(0)
        print(f'  {TERM_LABEL[term]:20s}' + '  '.join(f'{v:6.3f}' for v in pg))
        for v, x, g in zip(picks, dnu.T, pg):
            rows.append({'term': term, 'payoff': v, 'delta': x.mean(),
                         'lo': np.quantile(x, .025), 'hi': np.quantile(x, .975),
                         'p_increase': g})
    pd.DataFrame(rows).to_csv(out_dir / f'pmcpars_contrast.{label}.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--out_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible2')
    parser.add_argument('--spline_order', default=5, type=int)
    parser.add_argument('--trace_dir', default=None,
                        help='read the trace from here instead of derivatives/cogmodels')
    parser.add_argument('--tag', default=None,
                        help='name the output TSVs after this instead of --label')
    args = parser.parse_args()
    main(args.bids_folder, args.out_dir, args.label, args.spline_order,
         args.trace_dir, args.tag)
