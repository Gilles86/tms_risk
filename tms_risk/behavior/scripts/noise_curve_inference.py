"""Noise functions in memory/perceptual coordinates, with honest inference on a curve.

Reparameterisation
------------------
`flexible1` estimates the noise on the first- and second-presented option;
`flexible2` estimates memory and perceptual noise. bauer composes the latter as

    nu_1 = perceptual + memory      nu_2 = perceptual

with the softplus applied to each *component* (`risky_choice.py:621`). So the
family-2 coordinates read off a family-1 posterior are

    perceptual = nu_2               memory = nu_1 - nu_2

and the subtraction has to happen in nu space, after the softplus -- not in
spline-coefficient space, which would both misstate the curve and (because
softplus >= 0) hide the sign. The distinction is not academic here: in the
refit nu_1 < nu_2 at low payoffs, i.e. *negative* memory noise, which family 2
cannot represent at all since it applies softplus to memory directly.

For a family-2 trace the two components are read straight off the posterior.

Inference on a smooth curve
---------------------------
Pointwise 95% CrIs at 120 grid points answer "is the effect credible *here*", which
over-states evidence when you scan the whole curve. The frequentist reflex is a
cluster-based permutation test, but that needs a null distribution from refitting
under label permutation -- hours per refit here, and it answers a different question
than a posterior does. Two Bayesian analogues are cleaner:

1. **Simultaneous (joint) credible band** (Krivobokova, Kneib & Claeskens 2010).
   Find k such that 95% of posterior draws satisfy
   |f_s(x) - mean(x)| <= k * sd(x) for *all* x at once, then band = mean +- k*sd.
   Where that band excludes zero, the effect is credible accounting for the fact
   that you looked at the entire curve. This is the direct analogue of a
   multiple-comparison-corrected band, and it costs nothing beyond the draws.

2. **Posterior probability of a regional hypothesis**, e.g.
   P(delta nu(x) > 0 for every x in [7, 20]). That is one probability of one
   compound statement, so no correction is needed at all -- it is the cleanest
   thing to put in a Results sentence.

We report both, plus the pointwise band for comparison.
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def basis(x, df_, lower, upper, anchor=None):
    """Spline basis at `x`, with knots anchored to `anchor` (the paradigm column).

    patsy's `bs` places interior knots at *quantiles of the x it is given*, so a
    basis rebuilt on a plotting grid does NOT match the one the model was fitted
    with. bauer anchors each variable's design_info to a paradigm column at
    construction time (`magnitude.py::_spline_x_for`): n1 for
    `n1_evidence_sd` / `memory_noise_sd`, n2 for `n2_evidence_sd` /
    `perceptual_noise_sd`. Reproduce that, then evaluate on the grid.
    """
    formula = (f'bs(x, degree=3, df={df_}, include_intercept=True, '
               f'lower_bound={lower}, upper_bound={upper}) - 1')
    if anchor is None:
        return np.asarray(dmatrix(formula, {'x': x}))
    di = dmatrix(formula, {'x': np.asarray(anchor)}).design_info
    return np.asarray(build_design_matrices([di], {'x': x})[0])


def paradigm_columns(bids_folder):
    """The n1 / n2 the models were fitted on, for knot anchoring."""
    from tms_risk.utils.data import get_all_behavior
    df = get_all_behavior(bids_folder=bids_folder, exclude_outliers=True)
    df = df[df.index.get_level_values('session').astype(str).str.startswith(('2', '3'))]
    n_safe, n_risky = df['n_safe'].values, df['n_risky'].values
    risky_first = (df['p1'] == 0.55).values
    return (np.where(risky_first, n_risky, n_safe),      # n1
            np.where(risky_first, n_safe, n_risky))      # n2


def simultaneous_band(draws, level=.95):
    """Krivobokova et al. simultaneous credible band. draws: (n_draw, n_grid)."""
    m, sd = draws.mean(0), draws.std(0)
    sd = np.where(sd > 0, sd, np.inf)
    z = np.max(np.abs(draws - m) / sd, axis=1)      # per draw, worst point on the curve
    k = np.quantile(z, level)
    return m, m - k * sd, m + k * sd, k


def main(bids_folder, label, spline_order, out_dir, lower, upper,
         grid_lower, grid_upper, trace_dir=None, tag=None):
    out_dir = Path(out_dir)
    tdir = Path(trace_dir) if trace_dir else Path(bids_folder) / 'derivatives' / 'cogmodels'
    post = az.from_netcdf(tdir / f'model-{label}_trace.netcdf').posterior
    label = tag or label

    # Read the family off the trace itself rather than off the label, so that
    # --tag cannot silently pick the wrong parameterisation.
    family = 2 if 'perceptual_noise_sd_spline1_mu' in post else 1

    def coefs(term):
        c = np.stack([post[f'{term}_spline{i}_mu'].values
                      for i in range(1, spline_order + 1)], -1)
        c = c.reshape(-1, c.shape[-2], c.shape[-1])          # sample, regressor, spline
        return c[:, 0, :], c[:, 0, :] + c[:, 1, :]           # ips (reference), vertex

    # The spline basis is anchored to the range the model was FITTED on; the
    # evaluation grid may be narrower. That matters for the simultaneous band:
    # the extrapolated tail above the presented range has huge variance and
    # inflates the critical k for the whole curve.
    xs = np.linspace(grid_lower, grid_upper, 120)
    n1, n2 = paradigm_columns(bids_folder)
    B = {'memory': basis(xs, spline_order, lower, upper, anchor=n1),
         'perceptual': basis(xs, spline_order, lower, upper, anchor=n2)}
    print(f'family {family}; knots from the paradigm (n1 median {np.median(n1):.0f}, '
          f'n2 median {np.median(n2):.0f}), bounds [{lower:.0f}, {upper:.0f}]; '
          f'band evaluated on [{grid_lower:.0f}, {grid_upper:.0f}]')

    # bauer composes family 2 as
    #     nu_1 = softplus(eta_memory + eta_perceptual),  nu_2 = softplus(eta_perceptual)
    # -- the softplus wraps the SUM (magnitude.py::_get_trialwise_evidence_sd), so the
    # components add on the *linear-predictor* scale, not in CHF. The family-1 rotation
    # is therefore eta_perc = eta_2, eta_mem = eta_1 - eta_2, and the plotted curves are
    # softplus of each. Memory noise is consequently >= 0 by construction, in both
    # families; a negative "memory noise" would mean the subtraction was done in nu
    # space, which is not the model's parameterisation.
    src = ({'perceptual': 'n2_evidence_sd', 'memory': 'n1_evidence_sd'} if family == 1
           else {'perceptual': 'perceptual_noise_sd', 'memory': 'memory_noise_sd'})
    eta = {(t, c): v @ B[t].T
           for t, term in src.items()
           for c, v in zip(['ips', 'vertex'], coefs(term))}
    if family == 1:
        for c in ['ips', 'vertex']:
            eta[('memory', c)] = eta[('memory', c)] - eta[('perceptual', c)]
    nus = {k: softplus(v) for k, v in eta.items()}

    rows, contrasts = [], {}
    for term in ['perceptual', 'memory']:
        nu = {c: nus[(term, c)] for c in ['ips', 'vertex']}
        for c in ['ips', 'vertex']:
            rows.append(pd.DataFrame({
                'term': term, 'stimulation': c, 'payoff': xs, 'nu': nu[c].mean(0),
                'lo': np.quantile(nu[c], .025, 0), 'hi': np.quantile(nu[c], .975, 0)}))
        d = nu['ips'] - nu['vertex']
        contrasts[term] = d
        m, slo, shi, k = simultaneous_band(d)
        rows.append(pd.DataFrame({
            'term': term, 'stimulation': 'ips - vertex', 'payoff': xs, 'nu': m,
            'lo': np.quantile(d, .025, 0), 'hi': np.quantile(d, .975, 0),
            'sim_lo': slo, 'sim_hi': shi}))
        print(f'\n=== {term} noise: cTBS contrast ===')
        print(f'  simultaneous-band critical k = {k:.2f} '
              f'(pointwise would be {1.96:.2f}); band is {k/1.96:.2f}x wider')
        pw = (np.quantile(d, .025, 0) > 0)
        sm = (slo > 0)
        rng = lambda mask: (f'{xs[mask].min():.0f}-{xs[mask].max():.0f} CHF'
                            if mask.any() else 'none')
        print(f'  credible increase, pointwise    : {rng(pw)}')
        print(f'  credible increase, simultaneous : {rng(sm)}')

    curves = pd.concat(rows)
    tag = '' if grid_upper >= upper else f'.to{grid_upper:.0f}'
    curves.to_csv(out_dir / f'noisecurve_reparam.{label}{tag}.tsv', sep='\t', index=False)

    # ------------------------------------------------------------------ localisation
    # "The effect is specific to low magnitudes" is a claim about the SLOPE of the
    # contrast, and it is scale-dependent: nu itself triples over the range, so a
    # constant absolute delta is a shrinking relative one. Test both, on draws.
    print('\n=== is the effect localised at low magnitudes? ===')
    loc = []
    for term in ['perceptual', 'memory']:
        d = contrasts[term]
        rel = d / nus[(term, 'vertex')]                      # fraction of vertex noise
        at = lambda a, x: a[:, np.argmin(np.abs(xs - x))]
        print(f'  {term}:')
        for scale, a in [('absolute (CHF)', d), ('relative (frac. of nu)', rel)]:
            print(f'    {scale:24s}' + ''.join(f'{v:>9.3f}' for v in
                                               [at(a, x).mean() for x in (7, 28, 56)]))
            for x in (28, 56, 112):
                if x > grid_upper:
                    continue
                p = float((at(a, 7) > at(a, x)).mean())
                print(f'      P[effect(7) > effect({x})] = {p:.3f}')
                loc.append({'term': term, 'scale': scale.split()[0], 'hi_x': x,
                            'p_larger_at_7': p,
                            'e7': float(at(a, 7).mean()), 'ehi': float(at(a, x).mean())})
    pd.DataFrame(loc).to_csv(out_dir / f'noisecurve_localisation.{label}{tag}.tsv',
                             sep='\t', index=False)

    print('\n=== regional hypotheses: P(delta nu > 0 everywhere in the region) ===')
    print(f'  {"region (CHF)":>16}  {"perceptual":>12}  {"memory":>12}')
    reg = []
    for lo_, hi_ in [(7, 14), (7, 20), (7, 28), (10, 30), (28, 112), (7, 112)]:
        if hi_ > grid_upper or lo_ < grid_lower:
            continue
        sel = (xs >= lo_) & (xs <= hi_)
        vals = {t: float((contrasts[t][:, sel] > 0).all(1).mean()) for t in contrasts}
        print(f'  {f"{lo_}-{hi_}":>16}  {vals["perceptual"]:12.3f}  {vals["memory"]:12.3f}')
        reg.append({'lo': lo_, 'hi': hi_, **{f'p_{t}': v for t, v in vals.items()}})
    pd.DataFrame(reg).to_csv(out_dir / f'noisecurve_regional.{label}{tag}.tsv',
                             sep='\t', index=False)
    print('\n  (one probability of one compound statement -- no correction needed)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--label', default='flexible1')
    parser.add_argument('--spline_order', default=5, type=int)
    parser.add_argument('--out_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--lower', default=7.0, type=float)
    parser.add_argument('--upper', default=112.0, type=float)
    parser.add_argument('--grid_lower', default=None, type=float)
    parser.add_argument('--grid_upper', default=None, type=float,
                        help='restrict the band to e.g. the presented range (28)')
    parser.add_argument('--trace_dir', default=None)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()
    main(args.bids_folder, args.label, args.spline_order, args.out_dir,
         args.lower, args.upper,
         args.grid_lower if args.grid_lower is not None else args.lower,
         args.grid_upper if args.grid_upper is not None else args.upper,
         args.trace_dir, args.tag)
