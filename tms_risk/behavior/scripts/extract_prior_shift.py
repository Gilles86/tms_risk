"""How big is the cTBS shift in the PRIOR, in the `_prior` model variant?

The `_prior` variant puts the cTBS regressor on `risky_prior_mu` / `safe_prior_mu`
instead of on the noise functions. It is the main alternative to the paper's account:
cTBS moved the observer's beliefs about the payoff distribution rather than adding
representational noise. It fits 21.5 nats worse than the perceptual-noise model
(`notes/data/table1_with_prior_variants.tsv`), but it is a serious competitor, so the
size of the shift it implies is worth reporting.

Everything the magnitude model does is in LOG space: `*_prior_mu` is the mean of a prior
over log(n), so exp() puts it back in CHF. `*_prior_sd` is softplus-transformed.
Regressors are ['Intercept', 'stimulation_condition[T.vertex]'] with IPS as the
reference, so the IPS - vertex contrast is MINUS the second coefficient.

To make the prior shift comparable to the noise shift, it is also propagated to the
quantity both accounts ultimately move -- the percept. The model's percept of a payoff
n is the posterior mean

    percept = w * log(n) + (1 - w) * mu_prior,     w = sigma^2 / (sigma^2 + nu^2)

so a shift in the prior mean moves the percept by (1 - w) * d(mu_prior), independent of
where the noise sits. nu comes from the same trace's (unmodulated) noise splines.

    python -m tms_risk.behavior.scripts.extract_prior_shift \
        --trace_dir /data/ds-tmsrisk/derivatives/cogmodels.overnight \
        --label flexible2_noisefix_prior.head --out_dir /data/prior_out
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from tms_risk.behavior.scripts.noise_curve_inference import basis as anchored_basis
from tms_risk.behavior.scripts.extract_pmc_parameters import (softplus,
                                                              paradigm_columns)


def flat(post, name, reg=None):
    v = post[name].values
    v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
    return v if reg is None else v[..., reg]


def main(bids_folder, trace_dir, label, out_dir, spline_order, tag):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idata = az.from_netcdf(Path(trace_dir) / f'model-{label}_trace.netcdf')
    post = idata.posterior
    tag = tag or label

    rows = []
    # ---------------------------------------------------------------- prior means
    for opt in ['risky', 'safe']:
        icpt = flat(post, f'{opt}_prior_mu_mu', 0)          # IPS (reference level)
        contrast = flat(post, f'{opt}_prior_mu_mu', 1)      # vertex - IPS
        ips, vertex = icpt, icpt + contrast
        shift = ips - vertex                                # = -contrast
        for name, v in [('ips', ips), ('vertex', vertex)]:
            rows.append(dict(quantity=f'{opt}_prior_mu', condition=name,
                             log_mean=v.mean(),
                             log_lo=np.quantile(v, .025), log_hi=np.quantile(v, .975),
                             chf_mean=np.exp(v).mean(),
                             chf_lo=np.quantile(np.exp(v), .025),
                             chf_hi=np.quantile(np.exp(v), .975)))
        rows.append(dict(quantity=f'{opt}_prior_mu', condition='ips - vertex',
                         log_mean=shift.mean(),
                         log_lo=np.quantile(shift, .025), log_hi=np.quantile(shift, .975),
                         p_lt0=(shift < 0).mean(),
                         chf_mean=(np.exp(ips) - np.exp(vertex)).mean(),
                         chf_lo=np.quantile(np.exp(ips) - np.exp(vertex), .025),
                         chf_hi=np.quantile(np.exp(ips) - np.exp(vertex), .975)))
        sd = softplus(flat(post, f'{opt}_prior_sd_mu', 0))
        rows.append(dict(quantity=f'{opt}_prior_sd', condition='both (not modulated)',
                         log_mean=sd.mean(), log_lo=np.quantile(sd, .025),
                         log_hi=np.quantile(sd, .975)))
    pd.DataFrame(rows).to_csv(out_dir / f'prior_shift.{tag}.tsv', sep='\t', index=False)

    # ------------------------------------------- propagate to the percept, via w
    n1_col, n2_col = paradigm_columns(bids_folder)
    lower, upper = 7., 112.
    xs = np.linspace(lower, upper, 60)
    eta = {}
    for term, anchor in [('memory_noise_sd', n1_col), ('perceptual_noise_sd', n2_col)]:
        B = anchored_basis(xs, spline_order, lower, upper, anchor=anchor)
        c = np.stack([flat(post, f'{term}_spline{i}_mu', 0)
                      for i in range(1, spline_order + 1)], -1)
        eta[term] = c @ B.T
    nu = {'n1 (first-presented)': softplus(eta['memory_noise_sd'] + eta['perceptual_noise_sd']),
          'n2 (second-presented)': softplus(eta['perceptual_noise_sd'])}

    prows = []
    for opt in ['risky', 'safe']:
        sd = softplus(flat(post, f'{opt}_prior_sd_mu', 0))[:, None]
        dmu = -flat(post, f'{opt}_prior_mu_mu', 1)[:, None]     # IPS - vertex, log units
        for pos, nu_ in nu.items():
            w = sd ** 2 / (sd ** 2 + nu_ ** 2)
            d_log = (1 - w) * dmu                              # percept shift, log units
            d_chf = xs[None, :] * d_log                        # first-order, in CHF
            for i, x in enumerate(xs):
                prows.append(dict(option=opt, position=pos, payoff=x,
                                  w=w[:, i].mean(),
                                  d_percept_log=d_log[:, i].mean(),
                                  d_percept_log_lo=np.quantile(d_log[:, i], .025),
                                  d_percept_log_hi=np.quantile(d_log[:, i], .975),
                                  d_percept_chf=d_chf[:, i].mean(),
                                  d_percept_chf_lo=np.quantile(d_chf[:, i], .025),
                                  d_percept_chf_hi=np.quantile(d_chf[:, i], .975),
                                  nu=nu_[:, i].mean()))
    pd.DataFrame(prows).to_csv(out_dir / f'prior_percept_shift.{tag}.tsv', sep='\t',
                               index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'\nwrote prior_shift.{tag}.tsv and prior_percept_shift.{tag}.tsv to {out_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--trace_dir', default='/data/ds-tmsrisk/derivatives/cogmodels.overnight')
    p.add_argument('--label', default='flexible2_noisefix_prior.head')
    p.add_argument('--out_dir', default='/data/prior_out')
    p.add_argument('--spline_order', default=5, type=int)
    p.add_argument('--tag', default='priorshift')
    a = p.parse_args()
    main(a.bids_folder, a.trace_dir, a.label, a.out_dir, a.spline_order, a.tag)
