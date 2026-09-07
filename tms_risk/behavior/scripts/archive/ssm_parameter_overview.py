"""Parameter overview across PMC / DDM / RDM × Weber-shared/Weber-indep/Flex
noise structures.

All bounded parameters (σ_*, a, t0, prior σ) are transformed to their natural
space (softplus) so they are directly comparable across structures. For Flex
models, σ_perceptual / σ_memory are evaluated at n=20 (mid-range) by composing
spline coefs × B-spline basis, then softplus — *not* by averaging raw coefs.
"""
import arviz as az
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
from pathlib import Path

bids = Path('/data/ds-tmsrisk/derivatives/cogmodels')
def load(l):
    return az.from_netcdf(bids / f'model-{l}_trace.netcdf')

n_grid = np.arange(7, 113, dtype=float)
B = np.asarray(patsy.dmatrix('bs(n, df=5, include_intercept=True) - 1',
                              {'n': n_grid}))
N_EVAL = 20   # geometric mean of empirical magnitudes (range 7–112,
              # log-mean ≈ 20.6, median = 20). Evaluating σ(n) here is
              # representative of a typical trial — n=50 would have been
              # near the right tail of the magnitude distribution.
B_neval = B[np.argmin(np.abs(n_grid - N_EVAL))]   # length-5 basis at n=N_EVAL

def softplus(x):  return np.log1p(np.exp(x))

MODELS = [
    # name, color, marker, label, prior_space ('log' for non-Flex, 'natural' for Flex)
    ('PMC Weber shared',  'mediumblue', 'o', '11a',          'log'),
    ('DDM Weber shared',  'tab:green',  'o', 'ddm_weber',    'log'),
    ('RDM Weber shared',  'tab:orange', 'o', 'rdm_weber',    'log'),
    ('DDM Weber indep',   'tab:green',  's', 'ddm_indep',    'log'),
    ('RDM Weber indep',   'tab:orange', 's', 'rdm_indep',    'log'),
    ('PMC Flex',          'mediumblue', '^', 'flexible2b',   'natural'),
    ('DDM Flex',          'tab:green',  '^', 'ddm_flexible', 'natural'),
    ('RDM Flex',          'tab:orange', '^', 'rdm_flexible', 'natural'),
]

def scalar_post(idata, par, transform=None):
    v = idata.posterior[f'{par}_mu']
    if v.ndim == 3:
        v = v.isel({v.dims[-1]: 0})
    x = v.values.reshape(-1)
    return transform(x) if transform else x

def flex_sigma_at_n50(idata, base):
    coefs = []
    for k in range(1, 6):
        v = idata.posterior[f'{base}_spline{k}_mu']
        if v.ndim == 3:
            v = v.isel({v.dims[-1]: 0})
        coefs.append(v.values.reshape(-1))
    pre = np.stack(coefs, axis=0).T @ B_neval
    return softplus(pre)

def _natural_prior_median(base):
    """Return median magnitude prior in natural space.

    Non-Flex models parameterize the prior in **log magnitude** (lognormal):
    median = exp(prior_mu). Flex models parameterize in **natural n**:
    median = prior_mu directly. The caller threads `prior_space` via the
    MODELS table so we know which transform to use.
    """
    def _(idata, prior_space):
        x = scalar_post(idata, base)
        return np.exp(x) if prior_space == 'log' else x
    return _

def get_noise(base):
    def _(idata):
        pvars = [v[:-3] for v in idata.posterior.data_vars if v.endswith('_mu')]
        if base in pvars:
            return scalar_post(idata, base, softplus)
        if any(p.startswith(f'{base}_spline') for p in pvars):
            return flex_sigma_at_n50(idata, base)
        return None
    return _

PARAMS = [
    ('σ_memory (n=20)',          'noise', get_noise('memory_noise_sd')),
    ('σ_perceptual (n=20)',      'noise', get_noise('perceptual_noise_sd')),
    ('σ_n1  (indep)',            'noise', get_noise('n1_evidence_sd')),
    ('σ_n2  (indep)',            'noise', get_noise('n2_evidence_sd')),
    # Priors. NB scale differs by model:
    #   non-Flex (PMC Weber, DDM/RDM Weber-shared, DDM/RDM Weber-indep) —
    #   evidence + prior in **log magnitude**. So safe_prior_mu = E[log n].
    #   Flex (PMC Flex, DDM/RDM Flex) — evidence + prior in **natural n**.
    # We exponentiate the non-Flex μ so all panels show the natural-space
    # *median* of the magnitude prior (log-normal median = exp(μ)).
    ('safe prior median',        'prior', _natural_prior_median('safe_prior_mu')),
    ('risky prior median',       'prior', _natural_prior_median('risky_prior_mu')),
    # σ panels are NOT directly comparable across spaces — kept untransformed
    # and labeled accordingly.
    ('safe prior σ †',           'prior', lambda i: scalar_post(i, 'safe_prior_sd', softplus)),
    ('risky prior σ †',          'prior', lambda i: scalar_post(i, 'risky_prior_sd', softplus)),
    ('a  (threshold)',           'ssm',   lambda i: scalar_post(i, 'a',  softplus) if 'a_mu'  in i.posterior else None),
    ('t0  (NDT, s)',             'ssm',   lambda i: scalar_post(i, 't0', softplus) if 't0_mu' in i.posterior else None),
    # RDM only (van Ravenzwaaij 2020 advantage decomposition):
    #   μ_i = w_0 + w_d·(tildeᵢ − tildeⱼ) + w_s·(tildeᵢ + tildeⱼ)
    # w_0 and w_d are softplus-transformed in bauer (positivity enforced —
    # negative w_d would mean drift away from the higher-utility option,
    # which is nonsensical). w_s is identity, so signed.
    ('w_0  (urgency / baseline)',  'ssm', lambda i: scalar_post(i, 'w_0', softplus) if 'w_0_mu' in i.posterior else None),
    ('w_d  (drift on Δ utility)',  'ssm', lambda i: scalar_post(i, 'w_d', softplus) if 'w_d_mu' in i.posterior else None),
    ('w_s  (drift on total stake)','ssm', lambda i: scalar_post(i, 'w_s') if 'w_s_mu' in i.posterior else None),
]

rows = []
for label, color, marker, code, prior_space in MODELS:
    idata = load(code)
    for name, fam, getter in PARAMS:
        try:
            x = getter(idata, prior_space) if 'prior median' in name else getter(idata)
        except TypeError:
            x = getter(idata)
        if x is None: continue
        rows.append(dict(model=label, color=color, marker=marker, prior_space=prior_space,
                         family=fam, param=name,
                         mean=x.mean(), lo=np.quantile(x, 0.03), hi=np.quantile(x, 0.97)))
df = pd.DataFrame(rows)

fams = ['noise', 'prior', 'ssm']
params_per_fam = {f: list(df[df.family == f]['param'].unique()) for f in fams}
ncols = max(len(ps) for ps in params_per_fam.values())

fig, axes = plt.subplots(len(fams), ncols, figsize=(ncols * 3.0, 11.0),
                         squeeze=False, sharey=False)

for r, fam in enumerate(fams):
    fam_params = params_per_fam[fam]
    # Per-row y-zoom: trim outliers so bounded-noise vs SSM scales don't collapse
    # everything to a flat line.
    for c in range(ncols):
        ax = axes[r, c]
        if c >= len(fam_params):
            ax.axis('off'); continue
        par = fam_params[c]
        sub = df[df['param'] == par].reset_index(drop=True)
        if len(sub) == 0:
            ax.axis('off'); continue
        # Shade the background to mark log-space vs natural-space prior models —
        # only meaningful for prior panels, but harmless elsewhere.
        if fam == 'prior':
            for i, row in sub.iterrows():
                bg = '#fff4e6' if row['prior_space'] == 'log' else '#e7f4ff'
                ax.axvspan(i - 0.45, i + 0.45, color=bg, zorder=0)
        for i, row in sub.iterrows():
            ax.errorbar(i, row['mean'],
                        yerr=[[row['mean'] - row['lo']],
                              [row['hi'] - row['mean']]],
                        fmt=row['marker'], color=row['color'],
                        markersize=8, capsize=4, mew=1.2, mfc='white')
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub['model'], rotation=42, ha='right', fontsize=8.5)
        ax.set_title(par, fontsize=11)
        if fam == 'noise':
            # σ ≥ 0; clip y to the 97% HDIs of non-flex fits so PMC/DDM Flex's
            # huge spline-σ at n=20 doesn't crush the rest. Mark out-of-frame
            # points with a downward marker at the top so the comparison stays
            # honest.
            non_flex = sub[~sub['model'].str.contains('Flex')]
            if len(non_flex) > 0:
                ymax = non_flex['hi'].max() * 2.5
                # Don't clip below the smallest Flex band, but cap at 3× max
                ax.set_ylim(0, max(ymax, 1.0))
                for i, row in sub.iterrows():
                    if row['hi'] > ymax:
                        ax.annotate(f'↑ {row["mean"]:.1f}', xy=(i, ymax * 0.95),
                                    ha='center', va='top', fontsize=8,
                                    color=row['color'], fontweight='bold')
        elif fam == 'ssm':
            ax.axhline(0, color='k', ls='--', lw=0.6, alpha=0.4)
        ax.grid(True, alpha=0.2, axis='y')

    axes[r, 0].set_ylabel({'noise': 'σ at n=20 (natural)',
                            'prior': 'value',
                            'ssm':   'value (softplus where bounded)'}[fam],
                           fontsize=9)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [
    Line2D([0],[0], marker='o', color='mediumblue', mfc='white', ls='', label='PMC'),
    Line2D([0],[0], marker='o', color='tab:green',  mfc='white', ls='', label='DDM'),
    Line2D([0],[0], marker='o', color='tab:orange', mfc='white', ls='', label='RDM'),
    Line2D([0],[0], marker='o', color='dimgray', mfc='white', ls='', label='Weber shared'),
    Line2D([0],[0], marker='s', color='dimgray', mfc='white', ls='', label='Weber indep'),
    Line2D([0],[0], marker='^', color='dimgray', mfc='white', ls='', label='Flex @ n=20'),
    Patch(facecolor='#fff4e6', edgecolor='none', label='Prior in log magnitude (exp transformed to natural median)'),
    Patch(facecolor='#e7f4ff', edgecolor='none', label='Prior in natural magnitude'),
]
fig.legend(handles=handles, loc='upper center', ncol=4,
           bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=9)

plt.suptitle('Parameter overview — group-μ at IPS reference, natural space (94% HDI)\n'
             '† σ values not directly comparable across log/natural prior parametrizations',
             fontsize=12, y=1.02)
plt.tight_layout()
out = Path('notes/figures/ssm_parameter_overview')
plt.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.savefig(out.with_suffix('.png'), dpi=130, bbox_inches='tight')
print(f'wrote {out}.{{pdf,png}}')

Path('notes/data').mkdir(exist_ok=True, parents=True)
df.to_csv('notes/data/ssm_parameter_overview.tsv', sep='\t', index=False)
print('wrote notes/data/ssm_parameter_overview.tsv')
