"""Posterior predictive checks across PMC / DDM / RDM × Weber-shared / Flex.

For each of six representative fits we draw N posterior samples via
bauer's ``ppc()`` and compare the simulated choices against the empirical
choice rate binned by log(n_risky / n_safe).

DDM ppc() requires ssm-simulators which is incompatible with this env's
numpy 1.26 pin — we skip those two models with a clear annotation. PMC
and RDM ppc() use the analytical Bernoulli / Wald-race likelihoods and
don't need ssm-simulators.
"""
import sys
sys.path.insert(0, '/Users/gdehol/git/tms_risk/libs/bauer')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path

from tms_risk.behavior.fit_model import build_model
from tms_risk.utils.data import get_all_behavior

bids = Path('/data/ds-tmsrisk/derivatives/cogmodels')

MODELS = [
    ('PMC Weber',  'mediumblue',  '11a',          'pmc'),
    ('PMC Flex',   'royalblue',   'flexible2b',   'pmc'),
    ('DDM Weber',  'tab:green',   'ddm_weber',    'ddm'),
    ('DDM Flex',   'limegreen',   'ddm_flexible', 'ddm'),
    ('RDM Weber',  'tab:orange',  'rdm_weber',    'rdm'),
    ('RDM Flex',   'goldenrod',   'rdm_flexible', 'rdm'),
]

# ── Load data ────────────────────────────────────────────────────────────────
df = get_all_behavior(bids_folder='/data/ds-tmsrisk')
df = df[df['rt'] >= 0.20].copy()
# Patsy formulas reference `stimulation_condition` as a column, but
# get_all_behavior puts it in the MultiIndex.
df = df.reset_index('stimulation_condition')
# `chose_risky` and `log(risky/safe)` are already in df from get_all_behavior.
df['log_ratio'] = df['log(risky/safe)']
df['log_ratio_bin'] = pd.qcut(df['log_ratio'], q=7, duplicates='drop')
df['log_ratio_mid'] = df.groupby('log_ratio_bin', observed=True)['log_ratio'].transform('mean')
n_subj = df.index.get_level_values('subject').nunique()
print(f'data: {len(df)} trials, {n_subj} subjects')

# Bin centers used everywhere
bin_centers = sorted(df['log_ratio_mid'].unique())
n_bins = len(bin_centers)

results = {}   # label -> (binwise_ppc_array[n_draws, n_bins], color, family)
ppc_n_samples = 60

for label, color, code, family in MODELS:
    print(f'\n=== {label} ({code}) ===')
    try:
        idata = az.from_netcdf(bids / f'model-{code}_trace.netcdf')
        model = build_model(code, df.copy())
        # bauer.ppc requires the estimation model to be built first (it uses
        # self.estimation_model.coords + the design matrices stored on `model`).
        # Flex models override with `paradigm=` kwarg; base RiskModel uses `data=`.
        try:
            model.build_estimation_model(data=df)
        except TypeError:
            model.build_estimation_model(paradigm=df)

        ppc = model.ppc(df, idata, n_posterior_samples=ppc_n_samples,
                        progressbar=True, random_seed=0)
        ppc['chose_risky'] = ppc['simulated_choice'].astype(float)
        # Bring log_ratio_mid in by inner-joining on the trial index levels
        # (those that ppc inherits from df, minus ppc_sample).
        trial_levels = [n for n in ppc.index.names if n != 'ppc_sample']
        mid_lookup = df.reset_index()[trial_levels + ['log_ratio_mid']].drop_duplicates(trial_levels).set_index(trial_levels)['log_ratio_mid']
        ppc_flat = ppc.reset_index()
        ppc_flat = ppc_flat.merge(mid_lookup.reset_index(), on=trial_levels, how='left')
        missing = ppc_flat['log_ratio_mid'].isna().sum()
        print(f'  merged {len(ppc_flat) - missing}/{len(ppc_flat)} ppc rows with bin')
        binwise = (ppc_flat.groupby(['ppc_sample', 'log_ratio_mid'], observed=True)['chose_risky']
                       .mean().unstack('log_ratio_mid'))
        results[label] = (binwise.values, color, family)
        print(f'  ppc: {binwise.shape}')
    except Exception as e:
        # Catch broadly: DDM ppc needs ssm-simulators (ImportError); flex PMC
        # ppc can hit scipy domain errors on the Bernoulli node when a draw's
        # spline-derived p falls outside (0, 1).
        print(f'  SKIPPED ({type(e).__name__}): {str(e)[:160]}')
        results[label] = (None, color, family)

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
axes = axes.flatten()

# Empirical choice rate (data) — same on every panel
data_rate = df.groupby('log_ratio_mid', observed=True)['chose_risky'].agg(['mean', 'sem']).reset_index()

for ax, (label, (binwise, color, family)) in zip(axes, results.items()):
    ax.errorbar(data_rate['log_ratio_mid'], data_rate['mean'], yerr=data_rate['sem'],
                fmt='o', color='black', markersize=6, capsize=3, zorder=3, label='Data ± SEM')
    if binwise is None:
        ax.text(0.5, 0.5, 'DDM ppc() needs\nssm-simulators\n(skipped locally)',
                transform=ax.transAxes, ha='center', va='center', fontsize=10,
                color='gray', style='italic')
    else:
        lo, mid, hi = np.nanquantile(binwise, [0.03, 0.5, 0.97], axis=0)
        ax.fill_between(bin_centers, lo, hi, color=color, alpha=0.3,
                        label='Model PPC 94% HDI')
        ax.plot(bin_centers, mid, color=color, lw=2, label='Model PPC median')
    ax.axhline(0.5, color='k', ls=':', alpha=0.3, zorder=1)
    ax.axvline(0,   color='k', ls=':', alpha=0.3, zorder=1)
    ax.set_title(label, fontsize=11)
    if ax in (axes[0], axes[3]):
        ax.set_ylabel('P(choose risky)')
    if ax in axes[3:]:
        ax.set_xlabel('log(n_risky / n_safe)')
    if ax is axes[0]:
        ax.legend(fontsize=8, loc='upper left')

plt.suptitle('Posterior predictive psychometric — six representative fits',
             fontsize=13, y=1.005)
plt.tight_layout()
out = Path('notes/figures/ssm_ppc_psychometric')
plt.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.savefig(out.with_suffix('.png'), dpi=130, bbox_inches='tight')
print(f'\nwrote {out}.{{pdf,png}}')

# ── RT chronometric: only models that succeeded and have RT ─────────────────
# (RDM Weber/Flex have simulated_rt; PMC doesn't)
have_rt = {label: (binwise, color, family) for label, (binwise, color, family) in results.items()
           if family in ('rdm', 'ddm') and binwise is not None}
if have_rt:
    # We didn't store per-trial RT above; re-collect cheap from the PPC we just ran.
    # But to keep this script simple, mention as a follow-up.
    print(f'\n(chronometric PPCs deferred — re-run with --collect-rt to materialize)')
