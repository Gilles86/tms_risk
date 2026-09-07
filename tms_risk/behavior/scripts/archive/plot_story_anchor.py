"""One-page account of the cTBS behavioural result, on the anchor parameterisation.

Same nine-panel argument as `plot_story_full`, but every quantity now comes from
a model whose parameters ARE the noise SDs at named payoffs, so panels A, B and
E can be read directly off the posterior instead of through a softplus of spline
coefficients.

Row 1  the two fitted noise terms, and what cTBS does to each.
Row 2  where the priors sit, the group parameter forest, the per-subject effect.
Row 3  the consequences: perceived value, choices, model comparison.

One thing the old version had to warn about disappears here: the anchor model
composes sigma_n1 = sigma_perc + sigma_mem as a plain sum of two positive
functions, so the per-term curves in A and B ARE additive and panel C's total is
their sum. No softplus, no non-additivity caveat.

    python -m tms_risk.behavior.scripts.plot_story_anchor --model_label log-spl3-percmem
"""
import argparse
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
IPS, VERTEX = '#d62728', '#2ca02c'
MEM, PERC, TOT = '#3B5BA5', '#8DA0CB', '0.15'
RISKY, SAFE = '#8172B2', '0.25'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
XT = [7, 14, 28, 56, 112]


def logx(ax, ticks=XT):
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def band(ax, s, col, ls='-', lw=1.3, alpha=.18):
    s = s.sort_values('x')
    ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=alpha, lw=0)
    ax.plot(s.x, s['mid'], color=col, ls=ls, lw=lw)


def main(data_dir, out_stem, label):
    dd = Path(data_dir)
    c = pd.read_csv(dd / 'anchor_curves.tsv', **READ)
    c = c[c.label == label]
    pri = pd.read_csv(dd / 'anchor_priors.tsv', **READ)
    pri = pri[pri.label == label]
    pay = pd.read_csv(dd / 'anchor_payoffs.tsv', **READ)
    sub = pd.read_csv(dd / f'subject_shifts/subject_shifts.{label}.tsv', **READ)
    dspf = dd / f'decision_space/decision_space.{label}.tsv'
    dsp = pd.read_csv(dspf, **READ) if dspf.exists() else None
    # The AVERAGE SUBJECT -- the model at its group-level parameters -- not the
    # average over subjects. Only this object satisfies the chain rule the
    # causal chain in G->H draws; see notes/audit_dp_magnitude.md audit 2.
    avgf = dd / f'decision_map.avg/decision_map.{label}.tsv'
    avg = pd.read_csv(avgf, **READ) if avgf.exists() else None
    dff = dd / f'decision_function/decision_function.{label}.tsv'
    dfun = pd.read_csv(dff, **READ) if dff.exists() else None
    stake = pd.read_csv(dd / f'ppc_anchor/ppc_anchor.delta_stake.{label}.tsv',
                        **READ)
    # prefer the SAFE-payoff aggregation: it puts the choice panels on the same
    # x-axis as the perceived-value panels, so the causal chain can be read
    # straight down a column instead of across a change of variable
    safef = dd / f'ppc_anchor/ppc_anchor.safe.{label}.tsv'
    absf = dd / f'ppc_anchor/ppc_anchor.stake.{label}.tsv'
    if safef.exists():
        stake_abs, XKEY, XLAB = pd.read_csv(safef, **READ), 'n_safe', 'Safe payoff (CHF)'
    elif absf.exists():
        stake_abs, XKEY, XLAB = pd.read_csv(absf, **READ), 'stake_chf', 'Stake (CHF)'
    else:
        stake_abs, XKEY, XLAB = None, 'stake_chf', 'Stake (CHF)'
    loo = pd.concat([pd.read_csv(f, **READ)
                     for f in glob.glob(str(dd / 'loo_anchor/loo.*.tsv'))],
                    ignore_index=True)

    # Layout: the noise functions and their cTBS effect across the top; then
    # two tall panels on the left (where the priors sit, model comparison) and,
    # on the right, the CONSEQUENCE COLUMNS -- perceived value above choices for
    # the same presentation order, so the causal chain reads straight down.
    fig = plt.figure(figsize=(9.2, 5.9), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=.02, h_pad=.03, wspace=.02, hspace=.05)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1, 1])
    AX = {k: fig.add_subplot(gs[0, i]) for i, k in enumerate('ABCD')}
    AX['E'] = fig.add_subplot(gs[1:, 0])
    AX['L'] = fig.add_subplot(gs[1:, 1])
    AX['H'] = fig.add_subplot(gs[1, 2])
    AX['I'] = fig.add_subplot(gs[1, 3])
    AX['J'] = fig.add_subplot(gs[2, 2])
    AX['K'] = fig.add_subplot(gs[2, 3])

    def sig_strip(ax, s, col):
        """Mark where the difference credibly excludes zero.

        `p_gt0` is the posterior probability that the difference is positive, so
        the two-sided 95% credible interval excludes zero exactly where it is
        above .975 or below .025. Drawn as a rug along the top rather than as
        stars, because the quantity is a curve and the reader needs to see WHERE
        it is credible, not just that it is somewhere.
        """
        if 'p_gt0' not in s.columns or not len(s):
            return
        s = s.sort_values('x')
        m = (s.p_gt0 > .975) | (s.p_gt0 < .025)
        if not m.any():
            return
        y = ax.get_ylim()[1]
        ax.plot(s.x[m], np.full(m.sum(), y * .93), '|', color=col, ms=4,
                mew=1.2, clip_on=False)

    # Which two channels the model actually has. The shared family decomposes
    # the noise into a perceptual and a memory term; the independent family
    # parameterises the two PRESENTATION POSITIONS directly and has no
    # memory/perceptual split to show. Label what the model says, not what we
    # would like it to say.
    shared = set(c.channel) >= {'perc', 'mem'}
    CH = ([('mem', 'Memory term'), ('perc', 'Perceptual term')] if shared else
          [('n1', 'First-presented option'),
           ('n2', 'Second-presented option')])
    TOTLAB = 'total (1st option)' if shared else 'first-presented'

    # -- A, B: the two noise terms ---------------------------------------
    for k, (chan, nm) in zip('AB', CH):
        ax = AX[k]
        for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
            band(ax, c[(c.channel == chan) & (c.condition == cond)], col)
        logx(ax)
        ax.set_title(nm, fontsize=8)
        ax.set_ylabel('Noise SD (log units)')
        ax.set_xlabel('Payoff (CHF)')
    AX['A'].text(.04, .10, 'IPS', color=IPS, transform=AX['A'].transAxes,
                 fontsize=7)
    AX['A'].text(.04, .02, 'Vertex', color=VERTEX, transform=AX['A'].transAxes,
                 fontsize=7)

    # -- C, D: the cTBS effect on each term, one panel each ---------------
    dsel = c[c.condition == 'delta']
    dm = 1.15 * max(abs(dsel.lo.min()), abs(dsel.hi.max())) if len(dsel) else .05
    for k, ((chan, base), col) in zip('CD', zip(CH, (MEM, PERC))):
        nm = 'cTBS on ' + base.lower().replace('-presented option', '-presented')
        ax = AX[k]
        ax.axhline(0, color='0.75', lw=.7, ls='--', zorder=0)
        sd_ = dsel[dsel.channel == chan]
        band(ax, sd_, col)
        ax.set_ylim(-dm, dm)
        logx(ax)
        sig_strip(ax, sd_, col)
        ax.set_title(nm, fontsize=8)
        ax.set_ylabel('Δ noise, IPS − vertex')
        ax.set_xlabel('Payoff (CHF)')
    AX['D'].text(.03, .03, 'Ticks: 95% CrI excludes 0', transform=AX['D'].transAxes,
                 fontsize=5.8, color='0.5')

    # -- D: where the priors sit -----------------------------------------
    ax = AX['E']
    for i, (which, col) in enumerate([('risky', RISKY), ('safe', SAFE)]):
        r = pri[pri.which == which]
        if not len(r):
            continue
        r = r.iloc[0]
        v = pay[pay.which == which]
        obs = np.repeat(v.payoff.values, v.n.values)
        y = 1 - i
        # Two different things, previously conflated in one bar:
        #   thick pale bar   the payoffs actually shown
        #   medium bar       the observer's PRIOR, mu +/- one prior SD -- a
        #                    property of the observer, not an error bar
        #   thin capped bar  the 95% CrI on mu -- how well we know WHERE the
        #                    prior sits, which is a different question and is
        #                    the one that says whether the offset from the
        #                    presented mean is credible
        ax.plot([obs.min(), obs.max()], [y + .22] * 2, color=col, lw=4,
                alpha=.22, solid_capstyle='butt')
        ax.plot([np.exp(r.mu - r.sd), np.exp(r.mu + r.sd)], [y] * 2, color=col,
                lw=3, alpha=.45, solid_capstyle='butt')
        ax.plot([np.exp(r.mu_lo), np.exp(r.mu_hi)], [y] * 2, color=col, lw=1.1,
                solid_capstyle='butt')
        for e in (r.mu_lo, r.mu_hi):
            ax.plot([np.exp(e)] * 2, [y - .05, y + .05], color=col, lw=1.1)
        ax.plot(np.exp(r.mu), y, 'o', ms=5, color=col)
        ax.plot(obs.mean() if False else np.exp(np.mean(np.log(obs))), y + .22,
                'v', ms=3.5, color=col, clip_on=False)
        ax.text(np.exp(r.mu), y - .19,
                f'μ {np.exp(r.mu):.0f} CHF [{np.exp(r.mu_lo):.0f}–'
                f'{np.exp(r.mu_hi):.0f}]   σ ×/ {np.exp(r.sd):.2f} '
                f'[{np.exp(r.sd_lo):.2f}–{np.exp(r.sd_hi):.2f}]',
                fontsize=5.6, color=col, ha='center', va='top')
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['Risky option', 'Safe option'], fontsize=7)
    ax.set_ylim(-.5, 1.5)
    logx(ax)
    ax.set_title('Where the priors sit', fontsize=8)
    ax.set_xlabel('Payoff (CHF)')
    ax.text(.5, .02, 'Pale: payoffs shown.  Wide: prior μ ± σ.  Thin: 95% CrI on μ.',
            transform=ax.transAxes, ha='center', fontsize=5.4, color='0.5')

    # -- H, I: the two ingredients of the decision variable ---------------
    # The choice index is
    #     (perceived log ratio + log p) / decision SD
    # and cTBS moves BOTH parts of it. Showing each option's perceived value
    # separately (the previous version) shows neither: what drives choice is the
    # ratio, and the decision SD divides it. These are the two upstream
    # quantities, in percent so they share an axis, and they compete -- which is
    # why the consequence in J/K is smaller than either.
    if dfun is not None:
        q = dfun[dfun.quantity.isin(['num_shift', 'den_ratio'])].copy()
        q['pct'] = np.where(q.quantity == 'num_shift',
                            100 * np.expm1(q['mid']), 100 * (q['mid'] - 1))
        agg = q.groupby(['quantity', 'order', 'n_safe'])['pct'].mean()
        m = 1.25 * agg.abs().max()
    for k, order in zip('HI', ['Risky first', 'Risky second']):
        ax = AX[k]
        ax.axhline(0, color='0.75', lw=.7, ls='--', zorder=0)
        if dfun is not None:
            for qn, col, nm in [('num_shift', RISKY, 'Perceived ratio'),
                                ('den_ratio', SAFE, 'Decision noise')]:
                v = agg.xs((qn, order))
                ax.plot(np.arange(len(v)), v.values, 'o-', ms=3.4, color=col,
                        lw=1.3, label=nm)
            ax.set_ylim(-m, m)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([])
        ax.set_title(order, fontsize=8)
        if k == 'H':
            ax.set_ylabel('cTBS effect (%)')
            ax.legend(loc='upper right', fontsize=5.8, handlelength=1.1,
                      labelspacing=.15, borderaxespad=.15)
        else:
            ax.set_yticklabels([])

    # -- J, K: consequence for choice, in the published Figure-3 form -----
    # Stake on x, stimulation as the hue, presentation order as the panel. The
    # absolute choice curves rather than their difference: the difference panel
    # spends its whole axis on a 2-point effect and reads as noise, while the
    # red-above-green separation in one panel and not the other is the actual
    # claim and is legible at a glance.
    for k, order in zip('JK', ['Risky first', 'Risky second']):
        ax = AX[k]
        o = stake_abs[stake_abs.order == order] if stake_abs is not None else None
        if o is None or not len(o):
            ax.text(.5, .5, 'no PPC', transform=ax.transAxes, ha='center',
                    color='0.6')
            ax.set_xticks([]); ax.set_yticks([])
        else:
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                q = o[o.stim == stim].sort_values(XKEY)
                ax.fill_between(q[XKEY], q.lo, q.hi, color=col, alpha=.20,
                                lw=0)
                ax.plot(q[XKEY], q.model, color=col, lw=1.3)
                ax.errorbar(q[XKEY], q.observed, yerr=q.observed_sem,
                            fmt='o', ms=3.6, color=col, lw=0, elinewidth=.9,
                            capsize=0, zorder=4)
            ax.axhline(.5, color='0.88', lw=.6, ls='--', zorder=0)
            ax.set_ylim(.40, .74)
            ax.set_yticks([.45, .55, .65])
            ax.set_xscale('log')
            ax.set_xticks(sorted(o[XKEY].unique()))
            ax.set_xticklabels([f'{v:.0f}' for v in sorted(o[XKEY].unique())])
            ax.minorticks_off()

        ax.set_xlabel(XLAB)
        if k == 'J':
            ax.set_ylabel('P(chose risky)')
            ax.text(.04, .95, 'IPS', color=IPS, transform=ax.transAxes,
                    va='top', fontsize=7)
            ax.text(.04, .82, 'Vertex', color=VERTEX, transform=ax.transAxes,
                    va='top', fontsize=7)
            ax.text(.04, .04, 'Dots: data ± SEM', transform=ax.transAxes,
                    fontsize=5.8, color='0.45')
        else:
            ax.set_yticklabels([])

    # -- I: model comparison ---------------------------------------------
    ax = AX['L']
    form = label.split('-')[1]
    place = label.split('-')[2]
    null = 'null' if place in ('null', 'perc', 'mem', 'percmem') else 'nullind'
    picks = [(label, 'cTBS changes payoff-\ndependent noise'),
             (f'log-weber-{place}', 'Noise same at every\npayoff (Weber)'),
             (f'log-{form}-{null}', 'cTBS changes nothing')]
    have = set(loo.label)
    picks = [(l, n) for l, n in picks if l in have]
    if len(picks) < 2:                       # e.g. split forms have no weber twin
        alt = [(f'log-weber-{place}', 'Noise same at every\npayoff (Weber)'),
               (f'log-power-{null}', 'cTBS changes nothing'),
               (f'log-{form}-{null}', 'cTBS changes nothing')]
        picks += [(l, n) for l, n in alt if l in have and l not in dict(picks)]
        picks = picks[:3]
    l = loo.set_index('label')
    base = l.elpd_loo.get(label, np.nan)
    y = np.arange(len(picks))[::-1]
    for yy, (lab, nm) in zip(y, picks):
        d = l.elpd_loo.get(lab, np.nan) - base
        col = '#2ca02c' if d >= -1e-9 else '#C44E52'
        ax.barh(yy, d, height=.5, color=col, alpha=.85, lw=0)
        ax.text(d / 2 if d < -8 else d - 2, yy, f'{d:.0f}' if d < -1e-9 else 'best',
                ha='center' if d < -8 else 'right', va='center', fontsize=6.5,
                color='white' if d < -8 else col, fontweight='bold')
        ax.text(2, yy, nm, fontsize=6.3, va='center', color='0.25')
    ax.axvline(0, color='0.4', lw=.8)
    ax.set_yticks([])
    ax.set_ylim(-.6, len(picks) - .4)
    ax.set_title('Model comparison', fontsize=8)
    ax.set_xlabel('ELPD relative to best (nats)')
    ax.set_xlim(min(-10, ax.get_xlim()[0]), abs(ax.get_xlim()[0]) * .75)

    for k in 'ABCDEHIJKL':
        AX[k].text(-.22, 1.12, k, transform=AX[k].transAxes, fontsize=8.5,
                   fontweight='bold', family='Arial', va='bottom')
    fig.suptitle('cTBS over parietal cortex adds noise to the second-presented '
                 'amount, and people gamble more\n'
                 f'35 subjects · within-subject IPS vs vertex · anchor '
                 f'parameterisation ({label}) · bands are 95% posterior intervals',
                 fontsize=8, y=1.045, linespacing=1.8)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-spl3-percmem')
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/story_anchor'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label)
