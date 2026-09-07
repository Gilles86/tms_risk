"""Regenerate the supplementary perceptual-distortion heatmaps (Figs. S1.1-S1.4).

These four 2x3 heatmaps were originally made in
``tms_risk/behavior/notebooks/nov25/2d_distortion_curces.ipynb``. The
computation here is a verbatim port of cells 0-20 of that notebook; the only
substantive change is in the labelling of the right-hand column.

**Why this script exists.** In S1.1-S1.3 the right-hand column is computed as
``ips / vertex`` (a ratio, centred on 1), but the panel titles said
"IPS - Vertex", which reads as a difference. The colour scales give it away:
they run 0.90-1.10 (or 0.95-1.05), i.e. centred on 1, not 0. S1.4 is the one
figure whose right column really is a subtraction (``noise_ips - noise_vertex``,
colour scale +/- 0.75), and it is labelled as such.

So: S1.1, S1.2, S1.3 -> "IPS / Vertex (ratio)"; S1.4 -> "IPS - Vertex (difference)".

The heavy part (loading the flexible2 trace, evaluating noise curves and
posterior percepts for every subject) is cached to a pickle so that re-running
for label/style tweaks is instant. Use ``--recompute`` to force a refit.

Usage
-----
    python -m tms_risk.behavior.scripts.plot_perceptual_heatmaps \
        --bids_folder /data/ds-tmsrisk --out_dir notes/figures
"""

import argparse
import os
import os.path as op
import pickle

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

# Grid shared by all four figures (notebook cell 7).
N_SAFE = np.arange(7, 28.5, 0.5)
FRAC = np.linspace(1.0, 4.0, 50)
N_SAFE_TICKS = [7, 10, 14, 20, 28]

# The risky option pays out with p = 0.55, so EV_risky = 0.55 * n_risky.
P_RISKY = 0.55

CMAP = 'inferno'
DIFF_CMAP = 'coolwarm'


def _build_model(df, model_label):
    """``build_model`` with a fallback for the pre-refactor bauer API.

    The trace this script reads was fitted with the bauer version that called
    the spline order ``polynomial_order``; ``fit_model.build_model`` targets the
    refactored library, where it is ``spline_order``. Whichever bauer is
    installed, build the same model.
    """
    from tms_risk.behavior.fit_model import build_model

    try:
        return build_model(df=df, model_label=model_label)
    except TypeError as e:
        if 'spline_order' not in str(e):
            raise

    from bauer.models import FlexibleNoiseRiskRegressionModel

    if model_label != 'flexible2':
        raise NotImplementedError(
            f'No pre-refactor fallback for model_label={model_label!r}')

    return FlexibleNoiseRiskRegressionModel(
        df,
        regressors={'memory_noise_sd': 'stimulation_condition',
                    'perceptual_noise_sd': 'stimulation_condition'},
        polynomial_order=5,
        memory_model='shared_perceptual_noise',
        prior_estimate='full')


def build_cache(bids_folder, model_label='flexible2', thin=25):
    """Port of notebook cells 0-15: noise curves + posterior percepts."""
    from bauer.utils.bayes import get_posterior

    from tms_risk.behavior.fit_model import get_data

    conditions = pd.DataFrame([{'stimulation_condition': 'vertex'},
                               {'stimulation_condition': 'ips'}])

    df = get_data(model_label=model_label, bids_folder=bids_folder)
    model = _build_model(df, model_label)
    idata = az.from_netcdf(op.join(bids_folder, 'derivatives', 'cogmodels',
                                   f'model-{model_label}_trace.netcdf'))
    model.build_estimation_model()

    idata_thin = idata.sel(draw=slice(None, None, thin))

    curves = model.get_sd_curve(conditions, idata=idata_thin,
                                x=np.arange(7, 113), variable='both', group=False)

    prior = model.get_conditionwise_parameters(conditions=conditions,
                                               idata=idata_thin, group=False)
    mean_priors = prior.loc[['risky_prior_mu', 'risky_prior_sd',
                             'safe_prior_mu', 'safe_prior_sd']]
    mean_priors = mean_priors.groupby(['subject', 'parameter']).mean().stack().to_frame('value')
    mean_priors = mean_priors.unstack('parameter').droplevel(0, axis=1)

    # Option 1 carries perceptual + memory noise; option 2 only perceptual.
    curves['n1_evidence_sd'] = curves['perceptual_noise_sd'] + curves['memory_noise_sd']
    curves['n2_evidence_sd'] = curves['perceptual_noise_sd']
    mean_curves = curves.groupby(['subject', 'x', 'stimulation_condition']).mean()

    # Interpolate the per-subject noise curves onto the (n_safe, frac) grid.
    n_safe = N_SAFE[:, np.newaxis]
    frac = np.repeat(FRAC[np.newaxis, :], n_safe.shape[0], axis=0)
    n_risky = n_safe * FRAC[np.newaxis, :]

    def _interp(d, x_new, index):
        f = interp1d(d.index.get_level_values('x').values, d, axis=0)
        return pd.DataFrame(f(x_new.ravel()), index=index, columns=d.columns)

    safe_curves = mean_curves.groupby(['subject', 'stimulation_condition']).apply(
        lambda d: _interp(d, n_safe, pd.Index(n_safe.flatten(), name='x')))

    n_safe_grid = np.repeat(n_safe, frac.shape[1], axis=1)
    risky_index = pd.MultiIndex.from_arrays(
        (n_safe_grid.flatten(), frac.flatten()), names=('n_safe', 'frac'))
    risky_curves = mean_curves.groupby(['subject', 'stimulation_condition']).apply(
        lambda d: _interp(d, n_risky, risky_index))

    for frame, prefix in ((safe_curves, 'safe'), (risky_curves, 'risky')):
        frame['prior_mu'] = mean_priors[f'{prefix}_prior_mu']
        frame['prior_sd'] = mean_priors[f'{prefix}_prior_sd']

    # Bayesian percept = posterior mean given noisy evidence + subject's prior.
    safe_x = safe_curves.index.get_level_values('x').values
    risky_x = (risky_curves.index.get_level_values('n_safe').values
               * risky_curves.index.get_level_values('frac').values)

    safe_est_safe_first, _ = get_posterior(safe_x, safe_curves['n1_evidence_sd'],
                                           safe_curves['prior_mu'], safe_curves['prior_sd'])
    safe_est_risky_first, _ = get_posterior(safe_x, safe_curves['n2_evidence_sd'],
                                            safe_curves['prior_mu'], safe_curves['prior_sd'])
    risky_est_safe_first, _ = get_posterior(risky_x, risky_curves['n2_evidence_sd'],
                                            risky_curves['prior_mu'], risky_curves['prior_sd'])
    risky_est_risky_first, _ = get_posterior(risky_x, risky_curves['n1_evidence_sd'],
                                             risky_curves['prior_mu'], risky_curves['prior_sd'])

    cache = {
        'safe_estimates_safe_first': safe_est_safe_first.groupby(['stimulation_condition', 'x']).mean(),
        'safe_estimates_risky_first': safe_est_risky_first.groupby(['stimulation_condition', 'x']).mean(),
        'risky_estimates_safe_first': risky_est_safe_first.groupby(['stimulation_condition', 'n_safe', 'frac']).mean(),
        'risky_estimates_risky_first': risky_est_risky_first.groupby(['stimulation_condition', 'n_safe', 'frac']).mean(),
        'safe_noise': safe_curves.groupby(['stimulation_condition', 'x']).mean(),
        'risky_noise': risky_curves.groupby(['stimulation_condition', 'n_safe', 'frac']).mean(),
        'n_subjects': curves.index.get_level_values('subject').nunique(),
    }
    return cache


def get_cache(bids_folder, model_label='flexible2', recompute=False):
    path = op.join(bids_folder, 'derivatives', 'cogmodels',
                   f'model-{model_label}_perceptual-heatmaps.pkl')
    if op.exists(path) and not recompute:
        with open(path, 'rb') as f:
            return pickle.load(f)

    cache = build_cache(bids_folder, model_label=model_label)
    with open(path, 'wb') as f:
        pickle.dump(cache, f)
    print(f'Wrote cache to {path}')
    return cache


def _condition_frames(cache, condition):
    """Percepts of (safe, risky) option for one presentation order.

    ``condition='risky_first'`` = the risky option came first, so it is the one
    held in working memory (n1, perceptual + memory noise) and the safe option
    is seen second (n2, perceptual noise only).
    """
    suffix = 'risky_first' if condition == 'risky_first' else 'safe_first'
    return (cache[f'safe_estimates_{suffix}'], cache[f'risky_estimates_{suffix}'])


def _noise(cache, condition):
    """Total decision noise sqrt(sd_safe^2 + sd_risky^2) per stimulation site."""
    safe_col, risky_col = ('n2_evidence_sd', 'n1_evidence_sd') if condition == 'risky_first' \
        else ('n1_evidence_sd', 'n2_evidence_sd')
    safe_noise = cache['safe_noise'][safe_col]
    risky_noise = cache['risky_noise'][risky_col]

    out = {}
    for site in ('vertex', 'ips'):
        out[site] = np.sqrt(safe_noise.loc[site].values[np.newaxis, :] ** 2
                            + risky_noise.loc[site].unstack('n_safe').values ** 2)
    return out


def _panel(ax, data, extent, vmin, vmax, cmap, levels, contour_color, fmt,
           title, xticks, yticks, ylabel, fig):
    im = ax.imshow(data, aspect='auto', origin='lower', extent=extent,
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    contours = ax.contour(data, levels=levels, extent=extent,
                          colors=contour_color, linewidths=0.7, alpha=0.9)
    ax.clabel(contours, levels=levels, fmt=fmt, fontsize=8, colors=contour_color)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel('Safe payoff')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return im


def _save(fig, out_dir, name):
    for ext in ('pdf', 'png'):
        path = op.join(out_dir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300 if ext == 'png' else None)
        print(f'Wrote {path}')
    plt.close(fig)


# ---------------------------------------------------------------- S1.1 / S1.2

def plot_percept(cache, out_dir, which):
    """S1.1 (risky option percept) and S1.2 (safe option percept).

    Right column = ips / vertex, i.e. a RATIO.
    """
    if which == 'risky':
        suptitle = 'TMS-induced changes in posterior percepts of risky payoffs'
        name, diff_lims = 'sfig_s1_1_risky_percept', (0.9, 1.1)
    else:
        suptitle = 'TMS-induced changes in posterior percepts of safe payoffs'
        name, diff_lims = 'sfig_s1_2_safe_percept', (0.95, 1.05)

    titles = {'risky_first': 'Risky first', 'safe_first': 'Risky second'}
    vmin, vmax = 5, 28
    levels = [5, 7, 10, 14, 20, 28]
    diff_levels = np.linspace(*diff_lims, 7)
    extent = [N_SAFE.min(), N_SAFE.max(), FRAC.min(), FRAC.max()]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, condition in enumerate(('risky_first', 'safe_first')):
        safe_estimate, risky_estimate = _condition_frames(cache, condition)

        if which == 'risky':
            est = {site: risky_estimate.loc[site].unstack('n_safe').values * P_RISKY
                   for site in ('vertex', 'ips')}
        else:
            est = {site: np.repeat(safe_estimate.loc[site].values[np.newaxis, :], len(FRAC), axis=0)
                   for site in ('vertex', 'ips')}

        ratio = est['ips'] / est['vertex']
        order = titles[condition]

        im1 = _panel(axes[i, 0], est['vertex'], extent, vmin, vmax, CMAP, levels,
                     'white', '%1.1f', f'Vertex posterior estimate ({order})',
                     N_SAFE_TICKS, [1, 2, 3, 4], 'Risky-safe ratio', fig)
        im2 = _panel(axes[i, 1], est['ips'], extent, vmin, vmax, CMAP, levels,
                     'white', '%1.1f', f'IPS posterior estimate ({order})',
                     N_SAFE_TICKS, [1, 2, 3, 4], 'Risky-safe ratio', fig)
        im3 = _panel(axes[i, 2], ratio, extent, diff_lims[0], diff_lims[1],
                     DIFF_CMAP, diff_levels, 'black', '%1.2f',
                     f'IPS / Vertex ratio ({order})',
                     N_SAFE_TICKS, [1, 2, 3, 4], 'Risky-safe ratio', fig)

        fig.colorbar(im1, ax=axes[i, 0], label='Perceived payoff')
        fig.colorbar(im2, ax=axes[i, 1], label='Perceived payoff')
        fig.colorbar(im3, ax=axes[i, 2], label='IPS / Vertex ratio')

    fig.tight_layout()
    fig.suptitle(suptitle, fontsize=24, y=1.025)
    _save(fig, out_dir, name)


# ------------------------------------------------------------------ S1.3

def plot_percept_ratio(cache, out_dir):
    """S1.3: perceived risky/safe EV ratio. Right column = ips / vertex."""
    frac_ev = FRAC * P_RISKY
    extent = [N_SAFE.min(), N_SAFE.max(), frac_ev.min(), frac_ev.max()]
    frac_ticks = [.7, 1.0, 1.3, 1.6, 1.9]

    vmin, vmax = 0.75, 1.95
    levels = np.linspace(vmin, vmax, 25)
    levels = levels[~np.isclose(levels, 1.0, atol=1e-6)]
    diff_vmin, diff_vmax = 0.9, 1.1
    diff_levels = np.linspace(diff_vmin, diff_vmax, 7)

    titles = {'risky_first': 'Risky first', 'safe_first': 'Risky second'}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, condition in enumerate(('risky_first', 'safe_first')):
        safe_estimate, risky_estimate = _condition_frames(cache, condition)

        est = {}
        for site in ('vertex', 'ips'):
            risky = risky_estimate.loc[site].unstack('n_safe').values * P_RISKY
            safe = safe_estimate.loc[site].values[np.newaxis, :]
            est[site] = risky / safe

        ratio = est['ips'] / est['vertex']
        order = titles[condition]

        im1 = _panel(axes[i, 0], est['vertex'], extent, vmin, vmax, CMAP, levels,
                     'white', '%1.2f', f'Vertex posterior estimate ({order})',
                     N_SAFE_TICKS, frac_ticks, 'Risky-safe ratio (EV)', fig)
        im2 = _panel(axes[i, 1], est['ips'], extent, vmin, vmax, CMAP, levels,
                     'white', '%1.2f', f'IPS posterior estimate ({order})',
                     N_SAFE_TICKS, frac_ticks, 'Risky-safe ratio (EV)', fig)
        im3 = _panel(axes[i, 2], ratio, extent, diff_vmin, diff_vmax, DIFF_CMAP,
                     diff_levels, 'black', '%1.2f',
                     f'IPS / Vertex ratio ({order})',
                     N_SAFE_TICKS, frac_ticks, 'Risky-safe ratio (EV)', fig)

        # Perceptual indifference: where the perceived EV ratio equals 1.
        for ax, data in ((axes[i, 0], est['vertex']), (axes[i, 1], est['ips'])):
            contour = ax.contour(data, levels=[1.0], extent=extent, colors='white',
                                 linewidths=3.0, linestyles='-', alpha=1.0)
            ax.clabel(contour, levels=[1.0], fmt={1.0: 'Indifference'},
                      fontsize=10, colors='white')

        for ax in axes[i]:
            ax.axhline(y=1., c='white', ls='--', lw=2)

        fig.colorbar(im1, ax=axes[i, 0], label='Perceived risky/safe EV')
        fig.colorbar(im2, ax=axes[i, 1], label='Perceived risky/safe EV')
        fig.colorbar(im3, ax=axes[i, 2], label='IPS / Vertex ratio')

    fig.tight_layout()
    fig.suptitle('TMS-induced changes in ratio of perceived risky/safe EV',
                 fontsize=24, y=1.025)
    _save(fig, out_dir, 'sfig_s1_3_percept_ratio')


# ------------------------------------------------------------------ S1.4

def plot_total_noise(cache, out_dir):
    """S1.4: total decision noise. Right column really is ips - vertex."""
    extent = [N_SAFE.min(), N_SAFE.max(), FRAC.min(), FRAC.max()]
    vmin, vmax = 0, 7
    levels = np.arange(vmin, vmax, .5)
    diff_levels = np.linspace(-0.75, 0.75, 7)
    titles = {'risky_first': 'Risky first', 'safe_first': 'Risky second'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, condition in enumerate(('risky_first', 'safe_first')):
        noise = _noise(cache, condition)
        diff = noise['ips'] - noise['vertex']
        order = titles[condition]

        im1 = _panel(axes[i, 0], noise['vertex'], extent, vmin, vmax, CMAP, levels,
                     'white', '%1.1f', f'Vertex total noise ({order})',
                     N_SAFE_TICKS, [1, 2, 3, 4], 'Risky-safe ratio', fig)
        im2 = _panel(axes[i, 1], noise['ips'], extent, vmin, vmax, CMAP, levels,
                     'white', '%1.1f', f'IPS total noise ({order})',
                     N_SAFE_TICKS, [1, 2, 3, 4], 'Risky-safe ratio', fig)
        im3 = _panel(axes[i, 2], diff, extent, -0.75, 0.75, DIFF_CMAP, diff_levels,
                     'black', '%1.2f', f'IPS − Vertex difference ({order})',
                     N_SAFE_TICKS, [1, 2, 3, 4], 'Risky-safe ratio', fig)

        fig.colorbar(im1, ax=axes[i, 0], label='Evidence noise (SD)')
        fig.colorbar(im2, ax=axes[i, 1], label='Evidence noise (SD)')
        fig.colorbar(im3, ax=axes[i, 2], label='IPS − Vertex difference')

    fig.tight_layout()
    fig.suptitle('TMS-induced changes in noise', fontsize=24, y=1.025)
    _save(fig, out_dir, 'sfig_s1_4_total_noise')


def main(bids_folder, out_dir, model_label, recompute, figures):
    sns.set(font_scale=1.25, style='white', palette=sns.color_palette())
    os.makedirs(out_dir, exist_ok=True)

    cache = get_cache(bids_folder, model_label=model_label, recompute=recompute)
    print(f"n_subjects = {cache['n_subjects']}")

    if 's1.1' in figures:
        plot_percept(cache, out_dir, 'risky')
    if 's1.2' in figures:
        plot_percept(cache, out_dir, 'safe')
    if 's1.3' in figures:
        plot_percept_ratio(cache, out_dir)
    if 's1.4' in figures:
        plot_total_noise(cache, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--out_dir', default='notes/figures')
    parser.add_argument('--model_label', default='flexible2')
    parser.add_argument('--recompute', action='store_true',
                        help='Ignore the cached curves and re-evaluate the trace.')
    parser.add_argument('--figures', default='s1.1,s1.2,s1.3,s1.4',
                        help='Comma-separated subset of s1.1,s1.2,s1.3,s1.4')
    args = parser.parse_args()

    main(args.bids_folder, args.out_dir, args.model_label, args.recompute,
         [f.strip().lower() for f in args.figures.split(',')])
