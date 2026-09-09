"""Do the participants whose noise rose most under cTBS also show the largest
shift in where their magnitude prior sits?

The motivating idea: a prior has to be LEARNED from the same noisy
representations it then corrects. If cTBS raises the representational noise on
small payoffs, the prior over small payoffs -- which in this design is the prior
over SAFE options (mean 15.8 CHF against 36.2 for risky) -- is estimated from
worse evidence and should end up less well calibrated. That predicts a coupling
ACROSS PARTICIPANTS between the size of the noise increase and the size of the
prior shift, which is a stronger and much more falsifiable claim than either
group-level effect on its own.

Two ways this can be spuriously positive, both handled here.

1. **The posterior trade-off.** Within one participant, a noise increase and a
   prior shift move the choice probabilities in overlapping ways, so the two
   contrasts can be anticorrelated in the posterior FOR PURELY ALGEBRAIC
   REASONS. If that within-participant coupling is strong, an across-participant
   correlation inherits it and says nothing about biology. `r_within` is the
   median over participants of the posterior correlation between the two
   contrasts. **Read it before reading anything else**: if it is large and of
   the same sign as `r`, the result is an artefact of the fit.

2. **Correlating shrunken point estimates.** Taking the two posterior MEANS per
   participant and correlating those throws away the per-participant
   uncertainty and then rebuilds an interval by resampling 35 shrunken numbers.
   The correlation is instead computed once per posterior draw,

       for each draw d:  r_d = corr_s( Dnoise_{s,d} , Dprior_{s,d} )

   so the spread across draws already contains the measurement error. No
   attenuation correction is applied or needed -- a badly identified parameter
   simply yields a wide posterior on r. `r_posterior_mean_only` is reported
   alongside so the difference is visible.

Every noise-carrying parameter is paired with every prior parameter that has a
cTBS regressor, so the script needs no knowledge of the model's parameter names.

    python -m tms_risk.behavior.scripts.anchor_noise_prior_coupling \\
        --label log-power-n1n2pmu.mapjitter.klw \\
        --trace_dir /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
#: substrings that mark a parameter as a magnitude-prior parameter rather than
#: a noise channel. Read off the name so no model-specific list has to be kept
#: in sync (the stale-`SHARED`-list bug, twice over, came from doing otherwise).
PRIOR_MARKS = ('_prior_mu', '_prior_sd')


def ctbs_contrast(coef, colnames):
    """IPS - vertex from a regressor axis, whatever the contrast coding.

    Treatment coding (`stimulation_condition`) puts the IPS cell in the
    intercept and vertex - IPS in column 1, so the contrast is -coef[1]. Sum
    coding (`C(stimulation_condition, Sum)`) puts the grand mean in the
    intercept and (IPS - grand mean) in column 1, with vertex = -that, so the
    contrast is 2*coef[1]. Getting this wrong is silent -- same shape, same
    sign in some models -- so it is read off the column NAME, never assumed.
    """
    name = str(colnames[1])
    if 'Sum)' in name or '[S.' in name:
        return 2.0 * coef[..., 1]
    if 'T.vertex' in name:
        return -coef[..., 1]
    raise ValueError(f'unrecognised cTBS regressor column {name!r}')


def _rank(a):
    """Average-rank transform along the last axis (avoids a scipy import)."""
    order = np.argsort(a, axis=-1, kind='stable')
    r = np.empty_like(order, dtype=float)
    np.put_along_axis(r, order,
                      np.broadcast_to(np.arange(a.shape[-1], dtype=float),
                                      a.shape).copy(), axis=-1)
    return r


def _rowcorr(x, y):
    """Row-wise Pearson correlation of two (n_row, n_col) arrays."""
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x ** 2).sum(-1) * (y ** 2).sum(-1))
    return np.where(den > 0, (x * y).sum(-1) / np.where(den > 0, den, 1), np.nan)


def contrasts(ds):
    """{parameter: (n_subject, n_sample) cTBS contrast} for every parameter
    that carries a stimulation regressor."""
    out = {}
    for p in ds.attrs['tms_risk_parameters'].split(','):
        rdim = f'{p}_regressors'
        if p not in ds or ds.sizes.get(rdim, 1) < 2:
            continue
        coef = (ds[p].stack(sample=('chain', 'draw'))
                .transpose('subject', 'sample', rdim).values)
        out[p] = ctbs_contrast(coef, ds[rdim].values)
    return out


def main(label, trace_dir, out_tsv):
    import xarray as xr
    ds = xr.open_dataset(Path(trace_dir) / f'model-{label}_trace.netcdf',
                         group='posterior')
    d = contrasts(ds)
    ds.close()

    priors = [p for p in d if any(m in p for m in PRIOR_MARKS)]
    noises = [p for p in d if p not in priors]
    if not priors:
        raise SystemExit(
            f'{label} has no prior parameter with a cTBS regressor -- there is '
            f'no prior shift to couple to. Parameters with a cTBS term: '
            f'{sorted(d)}')

    rows = []
    for pn in noises:
        for pp in priors:
            a, b = d[pn], d[pp]                      # (n_sub, n_sample) each
            n_sub = a.shape[0]

            # --- the confound: within-participant posterior coupling --------
            within = _rowcorr(a, b)                  # one r per participant
            r_within = float(np.median(within))

            # --- the estimate: across participants, once per draw -----------
            x, y = a.T, b.T                          # (n_sample, n_sub)
            r = _rowcorr(x, y)
            rs = _rowcorr(_rank(x), _rank(y))

            rows.append(dict(
                label=label, noise=pn, prior=pp, n=n_sub,
                r=float(np.median(r)),
                lo=float(np.quantile(r, .025)),
                hi=float(np.quantile(r, .975)),
                p_gt0=float(np.nanmean(r > 0)),
                rho=float(np.median(rs)),
                rho_lo=float(np.quantile(rs, .025)),
                rho_hi=float(np.quantile(rs, .975)),
                r_within=r_within,
                r_within_lo=float(np.quantile(within, .025)),
                r_within_hi=float(np.quantile(within, .975)),
                r_posterior_mean_only=float(
                    _rowcorr(a.mean(1)[None], b.mean(1)[None])[0]),
            ))

    out = pd.DataFrame(rows).sort_values('r', key=abs, ascending=False)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep='\t', index=False)
    pd.set_option('display.width', 250)
    print(f'{label}: {len(noises)} noise x {len(priors)} prior parameters, '
          f'n = {rows[0]["n"]} participants\n')
    print(out[['noise', 'prior', 'r', 'lo', 'hi', 'p_gt0', 'rho', 'r_within',
               'r_posterior_mean_only']]
          .to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print('\n  r         across participants, per posterior draw '
          '(the estimate)\n'
          '  r_within  median WITHIN-participant posterior correlation of the '
          'two\n            contrasts. A large value of the SAME SIGN as r '
          'means the fit,\n            not the biology, produced r.')
    print(f'\nwrote {out_tsv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='log-power-n1n2pmu.mapjitter.klw')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_tsv', default=None)
    a = ap.parse_args()
    main(a.label, a.trace_dir,
         a.out_tsv or str(REPO / f'notes/data/noise_prior_coupling.{a.label}.tsv'))
