"""Random-slopes check of the stake x stimulation interaction. One model per process,
so the three can run concurrently on the CPU VM.

    python probit_rs_remote.py --model A_stake_rs --out_dir /data/probit_out

Factors are recoded as plain 0/1 numerics so bambi does not expand categoricals inside
the random-effects term (that raised a pytensor shape error locally). Coding matches
bambi's treatment coding on the fixed side, so the interaction coefficient means what it
did when read off the published trace:

    stim_v = 1 for vertex (0 = ips)      bin_hi = 1 for the high bin (0 = low)
    rf     = 1 when the risky option came FIRST (0 = risky second, the reference)

    gamma(ips,low)=x   gamma(vertex,low)=x+x:stim_v   gamma(ips,high)=x+x:bin_hi
    gamma(vertex,high)=x+x:stim_v+x:bin_hi+c
    => with Delta = IPS - vertex,  Delta_low - Delta_high = c = `x:stim_v:<bin>`
    NEGATIVE c = cTBS reduces the psychometric slope MORE at the low end (paper's claim).
"""
import argparse, os.path as op
import numpy as np, pandas as pd, arviz as az, bambi

FITS = {
    'A_stake_rs': ('chose_risky ~ x*rf*stim_v*stake_hi + (x*stim_v*stake_hi|subject)',
                   'x:stim_v:stake_hi', 'x:rf:stim_v:stake_hi'),
    'B_stake_ratio_rs': ('chose_risky ~ x*rf*stim_v*stake_hi + x*rf*stim_v*ratio_hi'
                         ' + (x*stim_v*stake_hi|subject)',
                         'x:stim_v:stake_hi', 'x:rf:stim_v:stake_hi'),
    'C_nsafe_rs': ('chose_risky ~ x*rf*stim_v*nsafe_hi + (x*stim_v*nsafe_hi|subject)',
                   'x:stim_v:nsafe_hi', 'x:rf:stim_v:nsafe_hi'),
    # intercept-only controls, matching the published random-effects structure
    'A_stake_ri': ('chose_risky ~ x*rf*stim_v*stake_hi + (1|subject)',
                   'x:stim_v:stake_hi', 'x:rf:stim_v:stake_hi'),
    'C_nsafe_ri': ('chose_risky ~ x*rf*stim_v*nsafe_hi + (1|subject)',
                   'x:stim_v:nsafe_hi', 'x:rf:stim_v:nsafe_hi'),
}


def flat(post, name):
    v = post[name].values
    return v.reshape(v.shape[0] * v.shape[1], -1)[:, 0] if v.ndim == 3 else v.ravel()


def main(model, bids_folder, out_dir, draws, tune, chains, cores):
    from tms_risk.behavior.fit_probit import get_data
    d = get_data('probit_average_n_full', bids_folder).reset_index()
    d['average_n'] = (d.n_safe + d.n_risky) / 2.
    q = lambda v: pd.qcut(v, 2, labels=[0, 1], duplicates='drop')
    d['stake_hi'] = d.groupby('subject')['average_n'].transform(q).astype(float)
    d['nsafe_hi'] = d.groupby('subject')['n_safe'].transform(q).astype(float)
    d['ratio_hi'] = d.groupby('subject')['x'].transform(q).astype(float)
    d['stim_v'] = (d.stimulation_condition == 'vertex').astype(float)
    d['rf'] = d.risky_first.astype(float)
    d['chose_risky'] = d.chose_risky.astype(float)
    d = d.dropna(subset=['stake_hi', 'nsafe_hi', 'ratio_hi', 'chose_risky', 'x'])
    print(f'{len(d)} trials, {d.subject.nunique()} subjects', flush=True)

    formula, c3n, c4n = FITS[model]
    print(f'=== {model} ===\n{formula}', flush=True)
    m = bambi.Model(formula, d, link='probit', family='bernoulli')
    idata = m.fit(draws=draws, tune=tune, chains=chains, cores=cores,
                  target_accept=0.95, random_seed=0, progressbar=False)
    idata.to_netcdf(op.join(out_dir, f'probit_{model}.netcdf'))

    fixed = [v for v in idata.posterior.data_vars
             if not v.endswith('_sigma') and '|' not in v]
    s = az.summary(idata, var_names=fixed)
    nd = int(idata.sample_stats.diverging.values.sum()) if 'diverging' in idata.sample_stats else -1
    print(f'  max r-hat {s.r_hat.max():.4f}  min ESS {s.ess_bulk.min():.0f}  '
          f'divergences {nd}', flush=True)

    post = idata.posterior
    c3 = flat(post, c3n)
    c4 = flat(post, c4n) if c4n in post else np.zeros_like(c3)
    rows = []
    for lab, v in [('risky SECOND (reference)', c3), ('risky FIRST', c3 + c4),
                   ('averaged over order', c3 + c4 / 2)]:
        lo, hi = np.quantile(v, [.025, .975])
        print(f'  interaction, {lab:26s} mean {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  '
              f'P(>0) = {(v > 0).mean():.4f}  P(<0) = {(v < 0).mean():.4f}', flush=True)
        rows.append(dict(model=model, contrast=lab, mean=v.mean(), lo=lo, hi=hi,
                         p_gt0=(v > 0).mean(), p_lt0=(v < 0).mean(),
                         max_rhat=float(s.r_hat.max()), min_ess=float(s.ess_bulk.min()),
                         divergences=nd, formula=formula))
    pd.DataFrame(rows).to_csv(op.join(out_dir, f'interaction_{model}.tsv'),
                              sep='\t', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=list(FITS))
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--out_dir', default='/data/probit_out')
    p.add_argument('--draws', default=1500, type=int)
    p.add_argument('--tune', default=1500, type=int)
    p.add_argument('--chains', default=4, type=int)
    p.add_argument('--cores', default=4, type=int)
    a = p.parse_args()
    main(a.model, a.bids_folder, a.out_dir, a.draws, a.tune, a.chains, a.cores)
