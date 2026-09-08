"""Figure 5, recomposed after an external design audit of the 3x3 version.

What changed and why.

* **Columns meant different things in different rows.** In the 3x3 the top row
  was indexed by OPTION POSITION (first / second / difference) and the lower
  rows by PRESENTATION ORDER (risky first / risky second), so "Second-presented
  option" sat directly above "Risky first" and a reader scanning a column
  conflated the two -- the exact confusion this paper has to avoid. Here every
  row has one logic, and the two rows are the two halves of the argument: what
  cTBS did to the representation (top), and what that did to choices (bottom).

* **Panel g is gone.** Its four rows were the x = 7 and x = 112 endpoints of the
  noise bands, so it restated panel a as a table. The only thing in it that was
  not already drawn -- the directional posterior probabilities -- is now
  annotated at the anchors on the difference panel, where the curve it refers
  to actually is.

* **The difference panel is the biggest.** It carries the claim, and in the 3x3
  it was the thinnest-ink panel on the page.

* **The behavioural consequence is shown.** The mechanism panels say the
  perceived risky/safe ratio rises by up to 8%, and the old figure never showed
  whether people then chose the risky option more often. That was the paper's
  headline effect, missing from its own mechanism figure.

* One estimand throughout: `--mechanism_level group` reads the `.group`
  mechanism table, evaluated at the group-level parameters, the same quantity
  the noise panels plot.

    python -m tms_risk.behavior.scripts.plot_fig5_recomposed \\
        --model_label log-power-n1n2.mapjitter.klw
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
IPS, VERTEX = '#d62728', '#2ca02c'
RATIO_C, NOISE_C, RISKY_C, SAFE_C = '0.15', '#3B5BA5', '#8172B2', '0.55'
ORDERS = ['Risky first', 'Risky second']
TICKS = [7, 14, 28, 56, 112]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def glyph_key(ax, entries, x=.04, y=.96, dy=.085, seg=.07, fs=6.5):
    """Inline legend drawn as the real marks, never as words describing them."""
    for i, (lab, col, kind, o) in enumerate(entries):
        yy = y - i * dy
        tf = ax.transAxes
        if kind == 'band':
            ax.add_patch(plt.Rectangle((x, yy - .020), seg, .040, transform=tf,
                                       facecolor=col, alpha=o.get('alpha', .2),
                                       lw=0, clip_on=False))
        elif kind == 'marker':
            ax.plot(x + seg / 2, yy, o.get('marker', 'o'), transform=tf,
                    ms=o.get('ms', 3.6), color=col, clip_on=False, mew=0)
        else:
            ax.plot([x, x + seg], [yy, yy], transform=tf, color=col,
                    ls=o.get('ls', '-'), lw=o.get('lw', 1.4),
                    alpha=o.get('alpha', 1), solid_capstyle='butt',
                    clip_on=False)
        ax.text(x + seg + .025, yy, lab, transform=tf, color=o.get('tc', col),
                fontsize=fs, va='center')


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks(TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.minorticks_off()
    ax.set_xlim(6.5, 122)


def main(data_dir, out_stem, label, mech_level):
    dd = Path(data_dir)
    c = pd.read_csv(dd / 'anchor_curves.tsv', **READ)
    c = c[c.label == label]
    chans = [ch for ch in ('n1', 'n2', 'perc', 'mem') if ch in set(c.channel)]
    NAME = {'n1': 'First-presented', 'n2': 'Second-presented',
            'perc': 'Perceptual', 'mem': 'Memory'}
    pri = pd.read_csv(dd / 'anchor_priors.tsv', **READ)
    pri = pri[pri.label == label]
    mf = dd / (f'anchor_mechanism.{label}'
               + ('.group' if mech_level == 'group' else '') + '.tsv')
    mech = pd.read_csv(mf, **READ)
    slp = pd.read_csv(dd / f'ppc_anchor/ppc_anchor.slope.{label}.tsv', **READ)
    dlt = pd.read_csv(dd / f'ppc_anchor/ppc_anchor.delta_stake.{label}.tsv', **READ)

    # Explicit gutters rather than constrained_layout: every panel carries its
    # own y-label and a row-2 title, and the engine was leaving them in the
    # neighbouring axes.
    fig = plt.figure(figsize=(7.6, 5.4))
    gs = fig.add_gridspec(2, 12, height_ratios=[1, 1], wspace=3.1, hspace=.42,
                          left=.075, right=.985, top=.91, bottom=.085)
    A = {'a': fig.add_subplot(gs[0, 0:4]),
         'b': fig.add_subplot(gs[0, 4:9]),
         'c': fig.add_subplot(gs[0, 9:12]),
         'd': fig.add_subplot(gs[1, 0:4]),
         'e': fig.add_subplot(gs[1, 4:8]),
         'f': fig.add_subplot(gs[1, 8:12])}

    # -- a: the noise functions, both channels, both conditions ------------
    ax = A['a']
    LS = {chans[0]: (0, (3, 1.6)), chans[-1]: '-'}
    for ch in chans:
        for cond, col in (('vertex', VERTEX), ('ips', IPS)):
            q = c[(c.channel == ch) & (c.condition == cond)].sort_values('x')
            if not len(q):
                continue
            ax.fill_between(q.x, q.lo, q.hi, color=col, alpha=.15, lw=0)
            ax.plot(q.x, q['mid'], color=col, ls=LS[ch], lw=1.4)
    logx(ax)
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Representational noise ν')
    glyph_key(ax, [('IPS', IPS, 'line', {}), ('Vertex', VERTEX, 'line', {}),
                   (NAME[chans[-1]], '0.3', 'line', dict(ls='-')),
                   (NAME[chans[0]], '0.3', 'line', dict(ls=(0, (3, 1.6))))],
              x=.04, y=.96, dy=.082)

    # -- b: the cTBS effect, with the directional probabilities at the anchors
    ax = A['b']
    ax.axhline(0, color='0.45', lw=1.1, zorder=0)
    dsel = c[c.condition == 'delta']
    for ch, col in zip(chans, ('0.58', '0.12')):
        q = dsel[dsel.channel == ch].sort_values('x')
        if not len(q) or float(np.abs(q['mid']).max()) < 1e-9:
            continue
        ax.fill_between(q.x, q.lo, q.hi, color=col, alpha=.16, lw=0)
        ax.plot(q.x, q['mid'], color=col, ls=LS[ch], lw=1.1)
        cred = (q.p_gt0 > .95).values
        if cred.any():
            i0, i1 = np.flatnonzero(cred)[[0, -1]]
            ax.plot(q.x.values[i0:i1 + 1], q['mid'].values[i0:i1 + 1],
                    color=col, ls=LS[ch], lw=2.4, solid_capstyle='round',
                    zorder=5)
        # the probabilities panel g used to tabulate, at the anchors they
        # belong to
        for xa in (q.x.min(), q.x.max()):
            j = int(np.argmin(np.abs(q.x.values - xa)))
            pg = float(q.p_gt0.values[j])
            ax.annotate(f'P = {pg:.2f}'.rstrip('0').rstrip('.') if pg not in
                        (0, 1) else f'P = {pg:.2f}',
                        (q.x.values[j], q['mid'].values[j]),
                        xytext=(3 if xa == q.x.min() else -3,
                                9 if ch == chans[-1] else -12),
                        textcoords='offset points', fontsize=6.4, color=col,
                        ha='left' if xa == q.x.min() else 'right',
                        annotation_clip=False)
    logx(ax)
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Δν,  IPS − vertex')
    ax.set_title('cTBS effect on noise', fontsize=8)
    ax.text(.5, 1.14, 'Raised on the second option, at small payoffs',
            transform=ax.transAxes, fontsize=6.8, color='0.35', ha='center')
    glyph_key(ax, [('P(Δν > 0) > 0.95', '0.12', 'line', dict(lw=2.4))],
              x=.04, y=.10, dy=.08)

    # -- c: where the priors sit -------------------------------------------
    ax = A['c']
    for k, (which, col) in enumerate((('risky', RISKY_C), ('safe', SAFE_C))):
        r = pri[pri.which == which]
        if not len(r):
            continue
        r = r.iloc[0]
        y = 1 - k
        ax.plot([np.exp(r.mu - r.sd), np.exp(r.mu + r.sd)], [y, y], color=col,
                lw=3.2, alpha=.5, solid_capstyle='butt')
        ax.plot(np.exp(r.mu), y, 'o', ms=5.5, color=col)
        ax.text(np.exp(r.mu), y + .22, f'{np.exp(r.mu):.0f} CHF', color=col,
                fontsize=6.6, ha='center')
    logx(ax)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['Risky', 'Safe'], fontsize=7)
    ax.set_ylim(-.75, 1.75)
    ax.set_xlabel('Payoff (CHF)')
    ax.set_title('Magnitude priors', fontsize=8)
    ax.text(.5, .04, 'Bar: ±1σ of the prior', transform=ax.transAxes,
            fontsize=6.2, color='0.45', ha='center')

    # -- d, e: the mechanism, one panel per presentation order -------------
    TR = [(RISKY_C, 'risky', (0, (2.6, 1.4)), 'Risky option'),
          (SAFE_C, 'safe', (0, (1.2, 1.2)), 'Safe option'),
          (RATIO_C, 'ratio', '-', 'Perceived ratio'),
          (NOISE_C, 'noise', '-', 'Decision noise')]
    xs = np.sort(mech.n_safe.unique())
    for k, order in zip(('d', 'e'), ORDERS):
        ax, q = A[k], mech[mech.order == order].sort_values('n_safe')
        ax.axhline(0, color='0.45', lw=1.0, zorder=0)
        for col, key, ls, _ in TR:
            if key not in q:
                continue
            if f'{key}_lo' in q:
                ax.fill_between(np.arange(len(q)), q[f'{key}_lo'],
                                q[f'{key}_hi'], color=col, alpha=.13, lw=0)
            ax.plot(np.arange(len(q)), q[key], color=col, ls=ls, lw=1.5,
                    marker='o', ms=3.2)
        ax.set_xticks(np.arange(len(xs)))
        ax.set_xticklabels([f'{v:.0f}' for v in xs])
        ax.set_xlabel('Safe payoff (CHF)')
        ax.set_title(order, fontsize=7.5)
        if k == 'd':
            ax.set_ylabel('cTBS effect (%)')
            glyph_key(ax, [(lab, col, 'line', dict(ls=ls))
                           for col, key, ls, lab in TR if key in q],
                      x=.04, y=.96, dy=.078)
        else:
            ax.tick_params(labelleft=False)
            ax.text(.5, .04, 'Ratio up: more risky choices\n'
                             'Noise up: flatter curve',
                    transform=ax.transAxes, fontsize=6.2, color='0.45',
                    ha='center', va='bottom', linespacing=1.6)

    # -- f: the behavioural consequence ------------------------------------
    ax = A['f']
    ax.axhline(0, color='0.45', lw=1.0, zorder=0)
    for order, mk, ls in zip(ORDERS, ('o', 'o'), ((0, (3, 1.6)), '-')):
        q = dlt[dlt.order == order].sort_values('stake_chf')
        x = np.arange(len(q))
        col = '0.62' if order == ORDERS[0] else '0.12'
        ax.fill_between(x, 100 * q.lo, 100 * q.hi, color=col, alpha=.14, lw=0)
        ax.plot(x, 100 * q.model, color=col, ls=ls, lw=1.4)
        ax.errorbar(x, 100 * q.observed, yerr=100 * q.observed_sem, fmt=mk,
                    ms=4.2, color=col, ecolor=col, elinewidth=.9, capsize=2,
                    zorder=5)
        ax.annotate(order, (x[-1], 100 * q.observed.iloc[-1]),
                    xytext=(4, 0), textcoords='offset points', color=col,
                    fontsize=6.6, va='center', annotation_clip=False)
    q0 = dlt[dlt.order == ORDERS[0]].sort_values('stake_chf')
    ax.set_xticks(np.arange(len(q0)))
    ax.set_xticklabels([f'{v:.0f}' for v in q0.stake_chf])
    ax.set_xlim(-.4, len(q0) - .35)
    ax.set_xlabel('Stake (CHF)')
    ax.set_ylabel('Δ P(chose risky), points')
    glyph_key(ax, [('Observed ±1 s.e.m.', '0.12', 'marker', dict(ms=4.2)),
                   ('Model, 95% predictive', '0.5', 'band', dict(alpha=.14))],
              x=.04, y=.14, dy=.085)

    for letter, k in zip('abcdef', 'abcdef'):
        A[k].text(-.16 if k in 'ad' else -.13, 1.05, letter,
                  transform=A[k].transAxes, fontsize=8.5, fontweight='bold',
                  family='Arial', va='bottom', ha='right')
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2.mapjitter.klw')
    ap.add_argument('--mechanism_level', default='subject',
                    choices=['subject', 'group'])
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    main(a.data_dir, a.out_stem or str(REPO / 'notes/figures/fig5_recomposed'),
         a.model_label, a.mechanism_level)
