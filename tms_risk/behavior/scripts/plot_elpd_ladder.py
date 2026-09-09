"""Model comparison for the paper's two modelling questions.

The paper asks two things of the model space, and they are separate questions
that a single ranked list conflates:

a  WHERE the cTBS effect acts. Noise on the first-presented option, the
   second, both, or on a perceptual/memory decomposition of the same two
   quantities. Noise form fixed to the power law throughout.
b  WHAT SHAPE the noise function takes. Weber (constant in log space) through
   to a five-knot spline, with the cTBS effect on both options throughout.

Priors are held fixed across sessions in every model shown: the perturbation
reduced choice consistency, which is a slope effect and therefore a noise
effect, and a prior change would have moved preferred numerosity, which it did
not (see Results).

Differences are PAIRED against the reported model, with the standard error of
the difference; the marginal SE of each ELPD is several times larger and would
make every comparison look inconclusive. Models that fail the convergence gate
are drawn in outline and labelled: an ELPD from a posterior that never mixed is
not a number to rank.

    python -m tms_risk.behavior.scripts.plot_elpd_ladder
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]
REF = 'log-power-perc'
GOOD, BAD, DEAD = '0.25', '#C44E52', '0.72'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 0,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})

PANELS = [
    # a  WHERE the cTBS effect acts, noise form held at the power law
    ('Where the cTBS effect acts', [
        ('log-power-n1n2',    'Both options'),
        ('log-power-percmem', 'Perceptual + memory'),
        ('log-power-perc',    'Perceptual only'),
        ('log-power-n2',      'Second option only'),
        ('log-power-n1',      'First option only'),
        ('log-power-mem',     'Memory only'),
        ('log-power-nullind', 'No cTBS effect'),
    ]),
    # b  WHAT SHAPE the noise function takes, placement held at both options.
    # Weber is the constant-nu null the paper argues against; genweber is
    # Weber plus an additive constant; power is the reported form. The splines
    # are NOT here -- they answer "is two anchors enough?", a different
    # question with its own figure, and mixing three-to-five-parameter
    # channels into a shape comparison invites the reader to read the ranking
    # as flexibility rather than shape.
    ('Shape of the noise function', [
        ('log-power-n1n2',    'Power law'),
        ('log-affine-n1n2',   'Affine'),
        ('log-genweber-n1n2', 'Gen. Weber'),
        ('log-weber-n1n2',    'Weber, constant ν'),
        ('log-weber-nullind', 'Weber, no cTBS effect'),
    ]),
]

#: The spline family, plotted separately by `--splines`. These ask whether two
#: anchors are enough, which is a question about RESOLUTION rather than shape:
#: spl3 and spl5 place three and five anchors over the same payoff range, so a
#: flat ladder here says the two-anchor power law already has the resolution
#: the data support.
SPLINES = [
    ('Is two anchors enough?', [
        ('log-power-n1n2', 'Power law, 2 anchors'),
        ('log-spl3-n1n2',  'Spline, 3 anchors'),
        ('log-spl5-n1n2',  'Spline, 5 anchors'),
    ]),
    ('… and without a cTBS effect', [
        ('log-power-nullind', 'Power law, 2 anchors'),
        ('log-spl3-nullind',  'Spline, 3 anchors'),
        ('log-spl5-nullind',  'Spline, 5 anchors'),
    ]),
]




#: Preference order over the sampler/init variants of one model. A bare label
#: like `log-power-perc` names a MODEL; on disk there may be several fits of it
#: differing only in init or in a convergence-prior override, and the ladder
#: must pick one deterministically. Converged variants win, then this order.
# Sampler/init variants only. Fits under a DIFFERENT PRIOR (tn*, ti*, sps*)
# are deliberately absent: a ladder must compare models on one prior spec, and
# swapping in the better-mixing tau_noise fit for a single rung would make its
# ELPD incommensurable with every other bar. The mixing of the rung that needs
# it is reported as numbers on the row instead.
VARIANTS = ['.mapjitter.klw', '.pathfinder.klw', '.klw']


def resolve(base, ld, chk):
    """Bare model label -> the KLW trace to plot, or None.

    Raises rather than silently falling back to a NON-KLW file: the raw-rule
    fits normalise a shrunken numerator by an unshrunken denominator, so their
    ELPD is not on the same footing as a KLW fit and a ladder mixing the two is
    meaningless. That is exactly what happened once -- a KLW reference against
    raw-rule comparators -- and it looked completely plausible.
    """
    cands = [base + v for v in VARIANTS
             if (ld / f'looi.{base + v}.npy').exists()]
    if not cands:
        if (ld / f'looi.{base}.npy').exists():
            raise SystemExit(
                f'{base}: only a non-KLW fit is available. Refusing to mix '
                f'choice rules in one ladder.')
        return None
    # prefer a variant that passes; otherwise the best-mixing one, so a rung
    # is never represented by a worse fit than exists at the same prior
    scored = sorted(cands, key=lambda c: (
        not (c in chk.index and bool(chk.loc[c, 'ok'])),
        float(chk.loc[c, 'max_rhat']) if c in chk.index else 9.0))
    return scored[0]


def main(data_dir, out_stem, ref, panels=None):
    dd = Path(data_dir)
    ld = dd / 'loo_anchor'
    _chkf = dd / 'all_klw_check.tsv'
    chk = pd.read_csv(_chkf if _chkf.exists() else dd / 'all_anchor_check.tsv',
                      **READ).set_index('trace')
    piw = lambda l: (np.load(ld / f'looi.{l}.npy')
                     if (ld / f'looi.{l}.npy').exists() else None)
    ref = resolve(ref, ld, chk) or ref
    a = piw(ref)
    if a is None:
        raise SystemExit(f'no pointwise LOO for the reference {ref}')
    print(f'reference: {ref}')

    fig, AX = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True,
                           sharex=True)
    for ax, (title, rows) in zip(AX, panels or PANELS):
        ys, seen = [], []
        for k, (base, nm) in enumerate(rows):
            lab = resolve(base, ld, chk)
            if lab is None:
                print(f'  no KLW fit for {base}, skipped')
                continue
            b = piw(lab)
            if lab == ref:
                d = se = 0.0
            elif a is not None and b is not None and b.shape == a.shape:
                diff = b - a
                d = float(diff.sum())
                se = float(np.std(diff, ddof=1) * np.sqrt(len(diff)))
            else:
                continue
            # Diagnostics, not a verdict. r_hat 1.02 / ESS 384 and r_hat 1.12 /
            # ESS 42 are not the same object, and one label for both is
            # misleading in whichever direction the reader happens to guess.
            r_ = float(chk.loc[lab, 'max_rhat']) if lab in chk.index else np.nan
            e_ = float(chk.loc[lab, 'min_ess_bulk']) if lab in chk.index else np.nan
            ok = bool(chk.loc[lab, 'ok']) if lab in chk.index else True
            seen.append((nm, d, se, lab == ref, ok, r_, e_))
        y = np.arange(len(seen))[::-1]
        # names sit clear of the longest POSITIVE bar+whisker, so a model that
        # beats the reference does not draw its bar through its own label
        xn = max([r[1] + r[2] for r in seen] + [0]) + 5
        for yy, (nm, d, se, is_ref, ok, r_, e_) in zip(y, seen):
            # grey out only what is genuinely unusable; a near-miss keeps its
            # ink and carries its numbers
            poor = (not ok) and (r_ > 1.05 or e_ < 100)
            col = GOOD if is_ref else (DEAD if poor else BAD)
            ax.barh(yy, d, height=.62, color=col, alpha=.30 if poor else .85,
                    lw=.8 if poor else 0, edgecolor=col, zorder=2)
            if se > 0:
                ax.plot([d - se, d + se], [yy] * 2,
                        color=DEAD if poor else '0.2',
                        lw=1.0, zorder=3, solid_capstyle='butt')
            txt = 'reference' if is_ref else f'{d:+.0f}'
            ax.text(min(d - se, 0) - 4, yy, txt, ha='right', va='center',
                    fontsize=7.5, color=col,
                    fontweight='bold' if is_ref else 'normal')
            diag = '' if ok else f'r̂ {r_:.2f} · ESS {e_:.0f}'
            ax.text(xn, yy, nm, fontsize=7.5, va='center',
                    color=DEAD if poor else '0.2')
            if diag:
                # flush right, so it can never run into a model name
                ax.text(.995, yy, diag.strip(), fontsize=6.4, va='center',
                        ha='right', transform=ax.get_yaxis_transform(),
                        color=DEAD if poor else '0.45')
        ax.axvline(0, color='0.4', lw=.9, zorder=1)
        ax.set_yticks([])
        ax.set_ylim(-1.5, len(seen) - .35)
        ax.set_xlim(-125, 175)
        ax.set_xticks([-100, -50, 0])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('ELPD relative to the reported model (nats)')
    # one glyph key, on the panel with room for it
    ax = AX[0]
    ax.plot([.06, .13], [.04, .04], transform=ax.transAxes, color='0.2', lw=1.0)
    ax.text(.15, .04, '±1 SE of the paired difference', transform=ax.transAxes,
            fontsize=6.8, va='center', color='0.45')
    for letter, ax in zip('ab', AX):
        ax.text(-.02, 1.06, letter, transform=ax.transAxes, fontsize=9,
                fontweight='bold', family='Arial', va='bottom', ha='right')
    sns.despine(fig=fig, left=True, offset={'bottom': 4})
    fig.savefig(f'{out_stem}.pdf')
    fig.savefig(f'{out_stem}.png', dpi=200)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--reference', default=REF)
    ap.add_argument('--splines', action='store_true',
                    help='plot the spline-resolution comparison instead')
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    stem = a.out_stem or str(REPO / ('notes/figures/supp_elpd_'
                                     + ('splines' if a.splines else 'ladder')))
    main(a.data_dir, stem, a.reference, SPLINES if a.splines else PANELS)
