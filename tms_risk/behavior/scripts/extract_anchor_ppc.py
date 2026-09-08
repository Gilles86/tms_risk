"""Posterior predictive check for anchor-parameterised fits.

Runs where the traces are; writes a small TSV per model that
`plot_anchor_ppc.py` turns into a figure locally.

Two things make this safe against the version trap that CLAUDE.md documents:

1. The model is rebuilt by `fit_anchor.build_model` from the trace's own stamped
   label, and its `parameter_signature()` is asserted equal to what the trace
   carries (`tms_risk_parameters` / `tms_risk_anchors`). The anchor
   parameterisation names every structural choice, so a mismatched graph cannot
   load silently -- but assert it anyway rather than trusting the name.
2. The predicted grand-mean P(risky) is compared with the observed one and the
   run aborts past `--max_gap`, the same tripwire `decompose_pmc_channels` uses.

The band is built from SIMULATED CHOICES, not from the predicted probabilities:
aggregating p gives parameter uncertainty only, which is far narrower than the
sampling distribution of an observed proportion and makes ordinary binomial
scatter look like misfit.

    python -m tms_risk.behavior.scripts.extract_anchor_ppc log-affine-percmem \\
        --trace_dir .../cogmodels.anchor --out_dir /home/gdehol/ppc
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from tms_risk.behavior.fit_anchor import build_model, parse_label

#: `log-affine-perc.pathfinder` names a variant refit; the grammar and the
#: model it builds are those of the part before the dot.
BASE = lambda lbl: lbl.split('.')[0]
from tms_risk.behavior.fit_model import get_data


def main(labels, bids_folder, trace_dir, out_dir, n_draws, max_gap):
    df = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for label in labels:
        path = Path(trace_dir) / f'model-{label}_trace.netcdf'
        if not path.exists():
            print(f'!! {label}: no trace yet, skipping')
            continue
        idata = az.from_netcdf(path)
        a = idata.posterior.attrs
        model = build_model(BASE(label), df.copy())

        sig = model.parameter_signature()
        assert sorted(sig['parameters']) == sorted(a['tms_risk_parameters'].split(',')), (
            f'{label}: rebuilt parameter set differs from the stamped one')
        assert ([f'{v:.0f}' for v in sig['anchors']]
                == a['tms_risk_anchors'].split(',')), f'{label}: anchors differ'

        model.build_estimation_model(save_p_choice=True)
        n_chain = idata.posterior.sizes['chain']
        keep = np.linspace(0, idata.posterior.sizes['draw'] - 1,
                           max(1, n_draws // n_chain)).astype(int)
        det = pm.compute_deterministics(idata.posterior.isel(draw=keep),
                                        model=model.estimation_model,
                                        var_names=['p'], merge_dataset=False,
                                        progressbar=False)
        p2 = det['p'].stack(sample=('chain', 'draw')).values     # P(choose option 2)
        p2 = p2[:, np.isfinite(p2).all(0)]

        d = df.reset_index().copy()
        d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
        d['stim'] = d['stimulation_condition']
        # option 2 is the risky one exactly when the risky option came second
        p_risky = np.where((~d['risky_first']).values[:, None], p2, 1 - p2)

        gap = float(p_risky.mean() - d['chose_risky'].astype(float).mean())
        print(f'{label}: {p_risky.shape[1]} draws, grand-mean gap {gap:+.4f}')
        if abs(gap) > max_gap:
            raise SystemExit(f'{label}: predicted grand mean is off by {gap:+.3f} '
                             f'(> {max_gap}); the graph does not match the trace')

        rng = np.random.default_rng(0)
        sim = (rng.random(p_risky.shape) < p_risky).astype(float)

        # The paradigm is a per-subject calibrated ladder of six risky payoffs per
        # safe payoff, at the same ratios regardless of stake, so the ladder rung IS
        # the bin: deterministic, tie-free, every subject in every cell.
        d['rung'] = (d.groupby(['subject', 'n_safe'], group_keys=False)['frac']
                     .rank(method='dense').astype(int))
        d['stake'] = (d['n_safe'] + d['n_risky']) / 2
        d['stake_bin'] = (d.groupby('subject', group_keys=False)['stake']
                          .apply(lambda v: pd.qcut(v.rank(method='first'), 3,
                                                   labels=False)))

        # -- the cTBS EFFECT itself, differenced within draw ---------------
        # The band on a difference is not the difference of two bands. Both the
        # model and the observed effect are formed within subject first, so the
        # between-subject variance that is common to the two conditions drops
        # out -- the same paired logic the behavioural analysis uses.
        # -- per-subject coverage ------------------------------------------
        # The group-level band can cover the group mean while individual
        # participants sit outside their own intervals; 95% of subject-cells
        # should fall inside a well-calibrated 95% interval, so the shortfall is
        # readable as a number rather than eyeballed off a band.
        skeys = ['subject', 'order', 'stake_bin', 'stim']
        sidx = pd.MultiIndex.from_frame(d[skeys])
        per_subj = pd.DataFrame(sim, index=sidx).groupby(level=skeys).mean()
        sub_obs = (d.assign(y=d['chose_risky'].astype(float))
                     .groupby(skeys)['y'].mean().reindex(per_subj.index))
        sub = pd.DataFrame({
            'observed': sub_obs.values,
            'model': per_subj.mean(axis=1).values,
            'lo': per_subj.quantile(.025, axis=1).values,
            'hi': per_subj.quantile(.975, axis=1).values,
        }, index=per_subj.index).reset_index()
        sub['covered'] = (sub.lo <= sub.observed) & (sub.observed <= sub.hi)
        sub.insert(0, 'label', label)
        sub.to_csv(out_dir / f'ppc_subject.{label}.tsv', sep='\t', index=False)
        print(f'  per-subject coverage {sub.covered.mean():.1%} '
              f'({int((~sub.covered).sum())}/{len(sub)} outside 95%)')

        # -- targeted posterior-predictive checks --------------------------
        # ELPD is dominated by the bulk choice curve -- 8335 trials of "steeper
        # ratio, more risky choices" that every model in the grid gets right --
        # so the cTBS x order x stake interaction the paper is about contributes
        # almost nothing to it. These statistics ask the model directly whether
        # it can PRODUCE that interaction: simulate a dataset per draw, compute
        # the same statistic on it, and see where the observed value falls.
        # A posterior-predictive p near 0 or 1 means the model cannot generate
        # what was measured, however good its ELPD.
        cellkeys = ['subject', 'order', 'stake_bin', 'stim']
        cellgrp = ['order', 'stake_bin', 'stim']
        cidx = pd.MultiIndex.from_frame(d[cellkeys])
        sim_cells = (pd.DataFrame(sim, index=cidx).groupby(level=cellkeys).mean()
                     .groupby(cellgrp).mean())
        obs_cells = (d.assign(y=d['chose_risky'].astype(float))
                       .groupby(cellkeys)['y'].mean().groupby(cellgrp).mean())
        obs_cells = obs_cells.reindex(sim_cells.index)
        pos = {k: i for i, k in enumerate(sim_cells.index)}

        def statistics(v):
            """v is (n_cells,) or (n_cells, n_draws); returns dict of stats."""
            def dp(order, b):
                return (v[pos[(order, b, 'ips')]] - v[pos[(order, b, 'vertex')]])
            lo_s, hi_s = dp('Risky second', 0), dp('Risky second', 2)
            lo_f, hi_f = dp('Risky first', 0), dp('Risky first', 2)
            mean_s = sum(dp('Risky second', b) for b in (0, 1, 2)) / 3
            mean_f = sum(dp('Risky first', b) for b in (0, 1, 2)) / 3
            return {
                'dp_second_high': hi_s,
                'dp_second_mean': mean_s,
                'order_contrast': mean_s - mean_f,
                'stake_slope_second': hi_s - lo_s,
                'three_way': (hi_s - lo_s) - (hi_f - lo_f),
            }

        # A prior-MEAN shift can only move the psychometric function sideways;
        # it cannot change its slope, because the decision SD in this model is
        # sqrt(sd_n1^2 + sd_n2^2) over the RAW evidence SDs and the prior does
        # not enter it. A prior-WIDTH change alters the shrinkage weight, which
        # compresses the perceived difference and so flattens the curve without
        # touching the decision SD. Only the noise function moves the decision
        # SD itself. So a slope statistic separates the three mechanisms, and
        # the bias statistics above cannot.
        rkeys = ['subject', 'order', 'rung', 'stim']
        rgrp = ['order', 'rung', 'stim']
        ridx = pd.MultiIndex.from_frame(d[rkeys])
        sim_r = (pd.DataFrame(sim, index=ridx).groupby(level=rkeys).mean()
                 .groupby(rgrp).mean())
        obs_r = (d.assign(y=d['chose_risky'].astype(float))
                   .groupby(rkeys)['y'].mean().groupby(rgrp).mean()
                   .reindex(sim_r.index))
        xr_ = np.log(d.groupby(rgrp)['frac'].mean().reindex(sim_r.index).values)
        rpos = {k: i for i, k in enumerate(sim_r.index)}

        def slopes(v):
            out = {}
            for order in ('Risky first', 'Risky second'):
                for stim in ('ips', 'vertex'):
                    ix = [rpos[k] for k in sim_r.index
                          if k[0] == order and k[2] == stim]
                    xx = xr_[ix] - xr_[ix].mean()
                    yy = v[ix]
                    out[(order, stim)] = (xx @ yy) / (xx @ xx)
            d_second = out[('Risky second', 'ips')] - out[('Risky second', 'vertex')]
            d_first = out[('Risky first', 'ips')] - out[('Risky first', 'vertex')]
            return {'slope_second_ctbs': d_second,
                    'slope_contrast': d_second - d_first}

        T_sim = statistics(sim_cells.values)
        T_sim.update(slopes(sim_r.values))
        T_obs = statistics(obs_cells.values)
        T_obs.update(slopes(obs_r.values))
        srows = []
        for k, sim_v in T_sim.items():
            obs_v = float(T_obs[k])
            srows.append(dict(
                label=label, statistic=k, observed=obs_v,
                model_median=float(np.median(sim_v)),
                lo=float(np.quantile(sim_v, .025)),
                hi=float(np.quantile(sim_v, .975)),
                # two-sided: 0 = the model never reaches the data, 1 = always
                # exceeds it; 0.5 = perfectly centred
                ppp=float((sim_v >= obs_v).mean()),
                covered=bool(np.quantile(sim_v, .025) <= obs_v
                             <= np.quantile(sim_v, .975))))
        pd.DataFrame(srows).to_csv(out_dir / f'ppc_stats.{label}.tsv',
                                   sep='\t', index=False)
        print('  ' + '  '.join(f"{r['statistic']}={r['observed']:+.3f}"
                               f"(ppp {r['ppp']:.2f})" for r in srows))

        # -- the psychometric curve, split by stake AND order ---------------
        # The bias statistics can be matched by a model that gets the SLOPE
        # wrong, so keep the full curve within each stake x order cell rather
        # than collapsing it. Two stake bins (within-subject median split), so
        # each cell still holds ~3 ladder rungs' worth of trials per subject.
        d['stake2'] = (d.groupby('subject', group_keys=False)['stake']
                       .apply(lambda v: (v > v.median()).astype(int)))
        # Two versions of the same curve. The median split keeps ~3 rungs per
        # cell per subject and is the safer summary; the TERCILE split is the
        # one the published Figure 4A used, and three panels show the
        # magnitude-dependence of the cTBS effect that two cannot.
        for skey, fname in [('stake2', 'stakerung'),
                            ('stake_bin', 'stake3rung')]:
            sk = ['subject', 'order', skey, 'rung', 'stim']
            sg = ['order', skey, 'rung', 'stim']
            obs_sr = (d.assign(y=d['chose_risky'].astype(float))
                        .groupby(sk)['y'].mean().groupby(sg).agg(['mean', 'sem'])
                        .rename(columns={'mean': 'observed',
                                         'sem': 'observed_sem'}))
            pdr = (pd.DataFrame(sim, index=pd.MultiIndex.from_frame(d[sk]))
                     .groupby(level=sk).mean().groupby(sg).mean())
            out_sr = pd.DataFrame({'model': pdr.mean(axis=1),
                                   'lo': pdr.quantile(.025, axis=1),
                                   'hi': pdr.quantile(.975, axis=1)}).join(obs_sr)
            out_sr = out_sr.join(d.groupby(sg)['frac'].mean().rename('frac'))
            out_sr = out_sr.join(d.groupby(sg)['stake'].mean().rename('stake_chf'))
            out_sr = out_sr.join(d.groupby(sg)['n_safe'].mean().rename('n_safe_chf'))
            out_sr = out_sr.reset_index().rename(columns={skey: 'stake_grp'})
            out_sr.insert(0, 'label', label)
            out_sr.to_csv(out_dir / f'ppc_anchor.{fname}.{label}.tsv',
                          sep='\t', index=False)

        for name, grp in [('rung', ['order', 'rung']),
                          ('stake', ['order', 'stake_bin'])]:
            keys = ['subject'] + grp
            o = (d.assign(y=d['chose_risky'].astype(float))
                   .groupby(keys + ['stim'])['y'].mean().unstack('stim'))
            o = (o['ips'] - o['vertex']).dropna()
            obs = o.groupby(grp).agg(['mean', 'sem']).rename(
                columns={'mean': 'observed', 'sem': 'observed_sem'})
            pd_ = (pd.DataFrame(sim, index=pd.MultiIndex.from_frame(d[keys + ['stim']]))
                     .groupby(level=keys + ['stim']).mean())
            eff = (pd_.xs('ips', level='stim') - pd_.xs('vertex', level='stim'))
            eff = eff.groupby(grp).mean()
            out = pd.DataFrame({'model': eff.mean(axis=1),
                                'lo': eff.quantile(.025, axis=1),
                                'hi': eff.quantile(.975, axis=1),
                                'p_gt0': (eff > 0).mean(axis=1)}).join(obs)
            out = out.join(d.groupby(grp)['frac'].mean().rename('frac'))
            out = out.join(d.groupby(grp)['stake'].mean().rename('stake_chf'))
            out = out.reset_index()
            out.insert(0, 'label', label)
            out.to_csv(out_dir / f'ppc_anchor.delta_{name}.{label}.tsv',
                       sep='\t', index=False)

        # Psychometric SLOPE per posterior draw. The pooled P(risky) curves
        # for the two stimulation conditions overlap almost exactly, which is
        # an accurate picture of a one-percentage-point separation and an
        # unreadable one; and pooled over stake the slope contrast even comes
        # out with the WRONG SIGN, because the fitted cTBS effect rotates the
        # noise function (up at low payoffs, down at high) and the high-stake
        # trials dominate. Split by stake and expressed as a slope, the model
        # and the observed probit are finally the same quantity on the same
        # axis. Slope is fitted per draw, so what comes out is a posterior.
        sg2 = ['order', 'stake_bin', 'stim']
        X = np.column_stack([np.ones(len(d)), np.log(d['frac'].values)])
        rows = []
        for gk, gi in d.groupby(sg2).indices.items():
            Xg, S = X[gi], sim[gi]                    # (n, 2), (n, n_draw)
            # logit slope by IRLS is overkill on 3-point-per-cell data; a
            # linear probability fit on the same design has the same sign and
            # is stable, and it is only ever compared LIKE FOR LIKE (model
            # against model, and against the probit's own slope contrast).
            beta = np.linalg.lstsq(Xg, S, rcond=None)[0][1]     # (n_draw,)
            yg = d['chose_risky'].values[gi].astype(float)
            b_obs = float(np.linalg.lstsq(Xg, yg, rcond=None)[0][1])
            rows.append(dict(zip(sg2, gk)) | dict(
                slope=float(beta.mean()),
                lo=float(np.quantile(beta, .025)),
                hi=float(np.quantile(beta, .975)),
                observed=b_obs,
                stake_chf=float(d['stake'].values[gi].mean()),
                n_trials=len(gi)))
        sl = pd.DataFrame(rows)
        # the contrast, per draw, so it is not a difference of summaries
        crows = []
        for gk, gi in d.groupby(['order', 'stake_bin']).indices.items():
            sub = d.iloc[gi]
            b = {}
            for st in ('ips', 'vertex'):
                ii = gi[(sub['stim'] == st).values]
                b[st] = np.linalg.lstsq(X[ii], sim[ii], rcond=None)[0][1]
            dd_ = b['ips'] - b['vertex']
            bo = {}
            for st in ('ips', 'vertex'):
                ii = gi[(sub['stim'] == st).values]
                bo[st] = float(np.linalg.lstsq(
                    X[ii], d['chose_risky'].values[ii].astype(float),
                    rcond=None)[0][1])
            crows.append(dict(order=gk[0], stake_bin=gk[1],
                              d_slope_observed=bo['ips'] - bo['vertex'],
                              d_slope=float(dd_.mean()),
                              lo=float(np.quantile(dd_, .025)),
                              hi=float(np.quantile(dd_, .975)),
                              p_lt0=float((dd_ < 0).mean()),
                              stake_chf=float(sub['stake'].mean())))
        sl = sl.merge(pd.DataFrame(crows), on=['order', 'stake_bin'],
                      suffixes=('', '_contrast'), how='left')
        sl.insert(0, 'label', label)
        sl.to_csv(out_dir / f'ppc_anchor.slope.{label}.tsv', sep='\t',
                  index=False)

        # Also aggregate by the SAFE payoff. Stake terciles are the paper's
        # convention, but the model's perceived-value panels are indexed by safe
        # payoff, and a causal-chain figure has to put the consequence on the
        # same x-axis as the cause.
        for name, grp in [('rung', ['order', 'rung', 'stim']),
                          ('stake', ['order', 'stake_bin', 'stim']),
                          ('safe', ['order', 'n_safe', 'stim'])]:
            keys = ['subject'] + grp
            obs = (d.assign(y=d['chose_risky'].astype(float))
                     .groupby(keys)['y'].mean()
                     .groupby(grp).agg(['mean', 'sem'])
                     .rename(columns={'mean': 'observed', 'sem': 'observed_sem'}))
            per_draw = (pd.DataFrame(sim, index=pd.MultiIndex.from_frame(d[keys]))
                          .groupby(level=keys).mean().groupby(grp).mean())
            mod = pd.DataFrame({'model': per_draw.mean(1),
                                'lo': per_draw.quantile(.025, axis=1),
                                'hi': per_draw.quantile(.975, axis=1)})
            xpos = d.groupby(grp)['frac'].mean().rename('frac')
            stake = d.groupby(grp)['stake'].mean().rename('stake_chf')
            safe = d.groupby(grp)['n_safe'].mean().rename('n_safe_chf')
            out = mod.join(obs).join(xpos).join(stake).join(safe).reset_index()
            out.insert(0, 'label', label)
            out.to_csv(out_dir / f'ppc_anchor.{name}.{label}.tsv',
                       sep='\t', index=False)

        rmse = float(np.sqrt(((out['model'] - out['observed']) ** 2).mean()))
        summary.append(dict(label=label, space=a['tms_risk_space'],
                            form=a['tms_risk_noise_form'],
                            placement=a['tms_risk_placement'],
                            grand_mean_gap=gap, rmse_stake=rmse,
                            n_draws=int(p_risky.shape[1])))
        print(f'  wrote ppc_anchor.*.{label}.tsv   RMSE {rmse:.4f}')

    pd.DataFrame(summary).to_csv(out_dir / 'ppc_anchor_summary.tsv',
                                 sep='\t', index=False)
    print(f'wrote {out_dir / "ppc_anchor_summary.tsv"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='+')
    ap.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-tmsrisk')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_dir', default='ppc_anchor')
    ap.add_argument('--n_draws', default=400, type=int)
    ap.add_argument('--max_gap', default=0.02, type=float)
    args = ap.parse_args()
    for lbl in args.labels:
        parse_label(BASE(lbl))
    main(args.labels, args.bids_folder, args.trace_dir, args.out_dir,
         args.n_draws, args.max_gap)
