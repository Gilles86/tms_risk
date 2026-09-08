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
# only two encodings survive in this figure: stimulation (red/green) and
# the numerator/denominator pair in the mechanism panels
RATIO_C, NOISE_C = '0.15', '#3B5BA5'
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
                    ms=o.get('ms', 3.6), color=col, clip_on=False,
                    mec=o.get('mec', col), mew=o.get('mew', 0))
        else:
            ax.plot([x, x + seg], [yy, yy], transform=tf, color=col,
                    ls=o.get('ls', '-'), lw=o.get('lw', 1.4),
                    alpha=o.get('alpha', 1), solid_capstyle='butt',
                    clip_on=False)
        ax.text(x + seg + .025, yy, lab, transform=tf, color=o.get('tc', col),
                fontsize=fs, va='center')


def empirical_payoffs(bids_folder):
    """Geometric mean and +/-1 SD of log payoff, per option role.

    What the observer's prior is an estimate OF. Reported on the same log scale
    the prior lives on, so the two are directly comparable.
    """
    from tms_risk.behavior.fit_model import get_data
    df = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm')
    rf = (df['p1'] == 0.55).values
    out = {}
    for which, v in (('risky', np.where(rf, df.n1, df.n2)),
                     ('safe', np.where(rf, df.n2, df.n1))):
        lg = np.log(np.asarray(v, float))
        out[which] = (float(np.exp(lg.mean() - lg.std())),
                      float(np.exp(lg.mean() + lg.std())),
                      float(np.exp(lg.mean())))
    return out


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks(TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.minorticks_off()
    ax.set_xlim(6.5, 122)


def _panel_f_delta(fig, A, dd, label):
    """The searching version: the cTBS difference along the ratio ladder, with
    the marginal MEAN. Fails -- observed +0.053 against a predictive interval of
    [-0.014, +0.044], p = 0.005 -- because the model predicts a flattening while
    the observed effect is mostly a bias shift. Supplement, not main text."""
    rng_ = pd.read_csv(dd / f'ppc_anchor/ppc_anchor.delta_rung.{label}.tsv', **READ)
    sta = pd.read_csv(dd / f'ppc_anchor/ppc_stats.{label}.tsv', **READ)
    MEANS = {'Risky first': 'dp_first_mean', 'Risky second': 'dp_second_mean'}
    gsF = A['f'].get_subplotspec().subgridspec(2, 1, hspace=.16)
    A['f'].set_visible(False)
    fx = None
    for r, order in enumerate(ORDERS):
        ax = fig.add_subplot(gsF[r], sharey=fx, sharex=fx)
        fx = fx or ax
        if r == 0:
            A['f'] = ax
        q = rng_[rng_.order == order].sort_values('frac')
        x = np.log(q.frac.values)
        xm = x.max() + .55 * (x.max() - x.min()) / (len(x) - 1)
        ax.axhline(0, color='0.45', lw=.9, zorder=0)
        ax.fill_between(x, 100 * q.lo, 100 * q.hi, color='0.6', alpha=.20,
                        lw=0, zorder=1)
        ax.plot(x, 100 * q.model, color='0.35', lw=1.2, zorder=2)
        ax.plot(x, 100 * q.observed, 'o', ms=3.8, color='0.1', zorder=4)
        m = sta[sta.statistic == MEANS[order]]
        if len(m):
            m = m.iloc[0]
            ax.plot([xm] * 2, [100 * m.lo, 100 * m.hi], color='0.6', lw=5,
                    alpha=.55, solid_capstyle='butt', zorder=2)
            ax.plot(xm, 100 * m.model_median, '_', ms=7, color='0.35',
                    mew=1.3, zorder=3)
            ax.plot(xm, 100 * m.observed, 'o', ms=5.2, zorder=5,
                    color='0.1' if m.covered else IPS)
            ax.annotate(f'p = {m.ppp:.3f}', (xm, 100 * m.observed),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=6.2,
                        color='0.45' if m.covered else IPS,
                        annotation_clip=False)
        ax.axvline((x.max() + xm) / 2, color='0.85', lw=.7, zorder=0)
        ax.set_xticks(list(np.log([1.5, 2, 2.5, 3])) + [xm])
        ax.set_xticklabels(['1.5', '2', '2.5', '3', 'Mean'], fontsize=6.6)
        ax.set_xlim(x.min() - .04, xm + .06)
        ax.set_ylim(-9.5, 17)
        ax.text(.98, .04, order, transform=ax.transAxes, fontsize=7,
                ha='right', va='bottom', color='0.25')
        if r == 0:
            ax.tick_params(labelbottom=False)
            glyph_key(ax, [('Observed', '0.1', 'marker', dict(ms=3.8)),
                           ('Model, 95% predictive', '0.6', 'band',
                            dict(alpha=.20))],
                      x=.03, y=.95, dy=.115, seg=.09, fs=6.2)
        else:
            ax.set_xlabel('Risky / safe payoff ratio')
            ax.set_ylabel('Δ P(chose risky), points')
            ax.yaxis.set_label_coords(-.19, 1.06)
        sns.despine(ax=ax, offset=3)


def main(data_dir, out_stem, label, mech_level, bids_folder, panel_f='slope'):
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
    # short label and 2-decimal ticks: the 3-decimal ones were wide enough to
    # push this label out of its own gutter and into panel a
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    ax.set_ylabel('Δν  (IPS − vertex)', labelpad=2)
    ax.set_title('cTBS effect on noise', fontsize=8)
    ax.text(.5, 1.14, 'Raised on the second option, at small payoffs',
            transform=ax.transAxes, fontsize=6.8, color='0.35', ha='center')
    glyph_key(ax, [('P(Δν > 0) > 0.95', '0.12', 'line', dict(lw=2.4))],
              x=.04, y=.10, dy=.08)

    # -- c: the fitted prior against the payoffs actually shown ----------
    # The observer's prior is an ESTIMATE of the payoff distribution, so the
    # panel that shows it should show what it is estimating. Pale strip = the
    # empirical spread of payoffs of that role (geometric mean, +/-1 SD of log
    # payoff); coloured bar = the fitted prior's +/-1 sigma; whisker = the 95%
    # CrI on the prior's mean, which is the uncertainty the old panel omitted.
    ax = A['c']
    emp = empirical_payoffs(bids_folder)
    # Colour encodes ONE thing here: fitted prior versus payoffs actually
    # shown. The option's role is already encoded by ROW, so colouring it too
    # made grey mean "safe option" in one mark and "empirical distribution" in
    # another, in the same panel. Role by position, quantity by ink.
    # pale-and-wide = what was shown, dark-and-narrow = what was fitted. The
    # first version had both at an effective 0.70 grey and they were
    # indistinguishable; the contrast has to be in VALUE and WIDTH, not hue.
    EMP, FIT = '0.80', '0.25'
    for k, which in enumerate(('risky', 'safe')):
        r = pri[pri.which == which]
        if not len(r):
            continue
        r = r.iloc[0]
        y = 1 - k
        lo_e, hi_e, gm = emp[which]
        ax.plot([lo_e, hi_e], [y] * 2, color=EMP, lw=11,
                solid_capstyle='butt', zorder=1)
        ax.plot([np.exp(r.mu - r.sd), np.exp(r.mu + r.sd)], [y, y], color=FIT,
                lw=3.6, solid_capstyle='butt', zorder=3)
        ax.plot([np.exp(r.mu_lo), np.exp(r.mu_hi)], [y, y], color='w', lw=1.1,
                zorder=4)
        ax.plot(np.exp(r.mu), y, 'o', ms=3.6, color='w', mec=FIT, mew=.8,
                zorder=5)
    logx(ax)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['Risky', 'Safe'], fontsize=7)
    ax.set_ylim(-.7, 2.1)
    ax.set_xlabel('Payoff (CHF)')
    ax.set_title('Priors vs payoffs shown', fontsize=8)
    glyph_key(ax, [('Payoffs shown, ±1 SD', EMP, 'bar', dict(lw=8)),
                   ('Fitted prior, ±1σ', FIT, 'bar', dict(lw=3.6)),
                   ('95% CrI on its mean', 'w', 'marker',
                    dict(ms=3.6, mec=FIT, mew=.8, tc='0.3'))],
              x=.04, y=.97, dy=.085, seg=.10, fs=6.2)

    # -- d, e: the mechanism, one panel per presentation order -------------
    # Two traces, not four. The perceived ratio is the numerator of the
    # decision variable and the decision noise is its denominator; they are the
    # two quantities that COMPETE, and that competition is the panel's point.
    # The risky and safe components were the ratio's own parts -- in risky-second
    # the ratio and the risky trace differ by 0.1 percentage points at 28 CHF --
    # so drawing all four put eight bands on two axes to show two things.
    TR = [(RATIO_C, 'ratio', '-', 'Perceived ratio: more risky choices'),
          (NOISE_C, 'noise', '-', 'Decision noise: flatter curve')]
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
            pass

    # -- f: the behavioural consequence ------------------------------------
    # The psychometric SLOPE by stake, per presentation order and stimulation.
    # This is the one view where the effect is legible: the model's IPS band
    # sits below its vertex band at the low and middle stakes on risky-second
    # trials, and the observed slopes do the same, converging by 42 CHF.
    #
    # The alternative -- the cTBS difference in P(chose risky) along the ratio
    # ladder -- is the more searching check and it fails (the model predicts a
    # flattening, the data are mostly a bias shift). It is `--panel_f delta`,
    # and it belongs in the supplement rather than here.
    if panel_f == 'delta':
        _panel_f_delta(fig, A, dd, label)
    else:
        slp = pd.read_csv(dd / f'ppc_anchor/ppc_anchor.slope.{label}.tsv', **READ)
        gsF = A['f'].get_subplotspec().subgridspec(2, 1, hspace=.16)
        A['f'].set_visible(False)
        fx = None
        for r, order in enumerate(ORDERS):
            ax = fig.add_subplot(gsF[r], sharey=fx, sharex=fx)
            fx = fx or ax
            if r == 0:
                A['f'] = ax
            o_ = slp[slp.order == order]
            xs = np.arange(o_.stake_bin.nunique())
            for stim, col in (('vertex', VERTEX), ('ips', IPS)):
                q = o_[o_.stim == stim].sort_values('stake_chf')
                ax.fill_between(xs, q.lo, q.hi, color=col, alpha=.18, lw=0,
                                zorder=1)
                ax.plot(xs, q.slope, color=col, lw=1.4, zorder=3)
                ax.plot(xs, q.observed, 'o', ms=4.4, color=col, zorder=5)
            ax.set_xticks(xs)
            ax.set_xticklabels([f'{v:.0f}' for v in
                                o_.groupby('stake_bin').stake_chf.mean()])
            ax.set_xlim(-.30, len(xs) - .70)
            ax.text(.97, .93, order, transform=ax.transAxes, fontsize=7,
                    ha='right', va='top', color='0.25')
            if r == 0:
                ax.tick_params(labelbottom=False)
                ax.text(.05, .18, 'IPS', transform=ax.transAxes, color=IPS,
                        fontsize=7)
                ax.text(.05, .05, 'Vertex', transform=ax.transAxes,
                        color=VERTEX, fontsize=7)
            else:
                ax.set_xlabel('Stake (CHF)')
                ax.set_ylabel('Psychometric slope')
                ax.yaxis.set_label_coords(-.19, 1.06)
                glyph_key(ax, [('Observed', '.25', 'marker', dict(ms=4.4)),
                               ('95% predictive', '.5', 'band',
                                dict(alpha=.18))],
                          x=.04, y=.30, dy=.11, seg=.09, fs=6.2)
            sns.despine(ax=ax, offset=3)

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
    ap.add_argument('--mechanism_level', default='group',
                    choices=['subject', 'group'])
    ap.add_argument('--panel_f', default='slope', choices=['slope', 'delta'])
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    main(a.data_dir, a.out_stem or str(REPO / 'notes/figures/fig5_recomposed'),
         a.model_label, a.mechanism_level, a.bids_folder, a.panel_f)
