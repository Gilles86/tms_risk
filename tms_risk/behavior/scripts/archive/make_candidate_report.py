"""One PDF for every model still in contention: noise function, PPC, ELPD.

The full anchor grid is 190 fits and nobody can hold that in their head. This is
the shortlist -- the models that could actually be reported -- laid out so the
three things that decide between them are each on one page:

    p1  ELPD and convergence. Which fit better, and which can be trusted.
    p2  The fitted noise functions. What each model says the observer is like.
    p3  The cTBS effect on those noise functions.
    p4  The posterior predictive check on the paper's own cells:
        stake x order x stimulation.
    p5  The published Figure-3 quantities, model against data, in the one cell
        the effect lives in.

Candidates are named explicitly rather than globbed, because "within reason" is
a judgement and it should be visible in the code which judgement was made.

    python -m tms_risk.behavior.scripts.make_candidate_report
"""
import argparse
from glob import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
IPS, VERTEX = '#d62728', '#2ca02c'
N1, N2 = '0.15', '0.55'
MODEL, DATA = '#3B5BA5', '0.15'
PAGE = (11.69, 8.27)

#: the shortlist. Order is the reading order on every page.
CANDIDATES = [
    ('log-power-n1n2', 'Power · n1n2'),
    ('log-affine-n1n2', 'Affine · n1n2'),
    ('log-spl3-n1n2', 'Spline-3 · n1n2'),
    ('log-spl5-n1n2', 'Spline-5 · n1n2'),
    ('log-power-perc', 'Power · perc'),
    ('log-power-percmem', 'Power · percmem'),
    ('log-spl3-percmem', 'Spline-3 · percmem'),
    ('log-spl5-percmem', 'Spline-5 · percmem'),
    ('log-spl7-percmem.pathfinder', 'Spline-7 · percmem'),
    ('log-spl9-percmem.pathfinder', 'Spline-9 · percmem'),
    ('log-spl5+affine-percmem.pathfinder', 'Spl5+affine · percmem'),
    ('log-weber-percmem', 'Weber · percmem'),
    ('log-power-nullind', 'Null (independent)'),
    ('log-power-null', 'Null (shared)'),
]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 6.5, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'lines.linewidth': 1.0, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 130, 'savefig.dpi': 300,
})


def cat(pattern, dd, cols=None):
    fs = sorted(glob(str(dd / pattern)))
    if not fs:
        return pd.DataFrame(columns=cols or [])
    return pd.concat([pd.read_csv(f, **READ) for f in fs], ignore_index=True)


def header(fig, title, sub):
    fig.text(.035, .965, title, fontsize=11, fontweight='bold', family='Arial',
             va='top')
    fig.text(.035, .928, sub, fontsize=7.5, color='0.30', va='top',
             linespacing=1.5)


def grid_axes(fig, n, rows, cols, **kw):
    gs = fig.add_gridspec(rows, cols, left=.055, right=.985, top=.855,
                          bottom=.075, hspace=.55, wspace=.30, **kw)
    return [fig.add_subplot(gs[i // cols, i % cols]) for i in range(n)]


def logx(ax, ticks=(7, 14, 28, 56, 112), labels=True):
    ax.set_xscale('log')
    ax.set_xticks(list(ticks))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}') if labels
        else mpl.ticker.NullFormatter())


def blank(ax, msg='not fitted'):
    ax.text(.5, .5, msg, transform=ax.transAxes, ha='center', va='center',
            fontsize=6.5, color='0.65')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def page_elpd(pdf, loo, ess, present, stamp):
    fig = plt.figure(figsize=PAGE)
    header(fig, 'Candidate models: fit and convergence',
           'ΔELPD is relative to the best candidate. Open marker = failed the '
           'gate (r̂ ≤ 1.01 and ESS ≥ 400 on group-level parameters).\n'
           'A model that cannot be sampled cannot be compared, however good its '
           'score.')
    gs = fig.add_gridspec(1, 2, left=.20, right=.97, top=.86, bottom=.10,
                          width_ratios=[1.5, 1], wspace=.32)
    ax = fig.add_subplot(gs[0])
    rows = []
    for lbl, nm in CANDIDATES:
        if lbl not in present:
            continue
        r = loo[loo.label == lbl]
        e = ess[ess.label == lbl]
        rows.append((nm, lbl,
                     float(r.elpd_loo.iloc[0]) if len(r) else np.nan,
                     float(r.se.iloc[0]) if len(r) else np.nan,
                     float(e.min_ess_group.iloc[0]) if len(e) else np.nan,
                     float(e.max_rhat_group.iloc[0]) if len(e) else np.nan,
                     int(r.n_par.iloc[0]) if len(r) else 0))
    if rows:
        best = np.nanmax([r[2] for r in rows])
        y = np.arange(len(rows))[::-1]
        for yy, (nm, lbl, e_, se, ess_, rhat, npar) in zip(y, rows):
            ok = (ess_ >= 400) and (rhat <= 1.01)
            ax.errorbar(e_ - best, yy, xerr=se, fmt='o', ms=4.5,
                        color=MODEL, mfc=MODEL if ok else 'white', mew=1.0,
                        elinewidth=.8, capsize=0)
            ax.text(3, yy, f'{npar} par', fontsize=5.6, va='center',
                    color='0.5')
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=7)
        ax.axvline(0, color='0.75', lw=.7, ls='--', zorder=0)
        ax.set_xlabel('ΔELPD vs best candidate')
        ax.set_ylim(-.7, len(rows) - .3)

    ax = fig.add_subplot(gs[1])
    if rows:
        y = np.arange(len(rows))[::-1]
        ax.axvline(400, color='#d62728', lw=.8, ls='--', zorder=0)
        ax.scatter([r[4] for r in rows], y, s=20, color='0.2')
        ax.set_xscale('log')
        ax.set_yticks(y); ax.set_yticklabels([])
        ax.set_xlabel('Minimum group-level ESS')
        ax.set_ylim(-.7, len(rows) - .3)
        ax.text(.03, .02, 'Red line: the gate at 400', transform=ax.transAxes,
                fontsize=6, color='#d62728')
    fig.text(.985, .012, stamp, fontsize=5.6, color='0.6', ha='right')
    sns.despine(fig=fig, offset=3)
    pdf.savefig(fig); plt.close(fig)


def page_curves(pdf, curves, present, stamp, condition, title, sub):
    fig = plt.figure(figsize=PAGE)
    header(fig, title, sub)
    shown = [(l, n) for l, n in CANDIDATES if l in present]
    axes = grid_axes(fig, len(shown), 3, 5)
    if condition == 'delta':
        d = curves[curves.condition == 'delta']
        m = 1.1 * max(abs(d.lo.min()), abs(d.hi.max())) if len(d) else .05
    else:
        d = curves[curves.condition.isin(['ips', 'vertex'])]
        m = 1.06 * d[d.channel.isin(['n1', 'n2'])].hi.max() if len(d) else .4
    for ax, (lbl, nm) in zip(axes, shown):
        c = curves[curves.label == lbl]
        if not len(c):
            blank(ax); ax.set_title(nm, fontsize=6.8, pad=2); continue
        if condition == 'delta':
            ax.axhline(0, color='0.8', lw=.5, ls='--', zorder=0)
            for chan, col, ls in [('n1', N1, '-'), ('n2', N2, (0, (2.4, 1.3)))]:
                s = c[(c.channel == chan) & (c.condition == 'delta')].sort_values('x')
                ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=.18, lw=0)
                ax.plot(s.x, s['mid'], color=col, ls=ls, lw=1.0)
            ax.set_ylim(-m, m)
        else:
            for chan, ls in [('n1', '-'), ('n2', (0, (2.4, 1.3)))]:
                for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
                    s = c[(c.channel == chan) & (c.condition == cond)].sort_values('x')
                    ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=.15, lw=0)
                    ax.plot(s.x, s['mid'], color=col, ls=ls, lw=1.0)
            ax.set_ylim(0, m)
        logx(ax)
        ax.set_title(nm, fontsize=6.8, pad=2)
    a = axes[0]
    if condition != 'delta':
        a.text(.04, .96, 'IPS', color=IPS, transform=a.transAxes, va='top',
               fontsize=6.5)
        a.text(.04, .82, 'Vertex', color=VERTEX, transform=a.transAxes,
               va='top', fontsize=6.5)
    a.set_ylabel('σ (log CHF)' if condition != 'delta' else 'Δσ, IPS − vertex')
    fig.text(.5, .022, 'Payoff (CHF).   Solid σ$_{n1}$ (first-presented), '
                       'dashed σ$_{n2}$ (second).   Bands 95% CrI.',
             ha='center', fontsize=6.5, color='0.4')
    fig.text(.985, .012, stamp, fontsize=5.6, color='0.6', ha='right')
    sns.despine(fig=fig, offset=2)
    pdf.savefig(fig); plt.close(fig)


def page_ppc(pdf, stake, present, stamp):
    fig = plt.figure(figsize=PAGE)
    header(fig, 'Posterior predictive check: stake × order × cTBS',
           'The paper\'s own cells. Points are the observed choice proportions '
           '± SEM across subjects; bands are 95% posterior predictive intervals '
           'from simulated choices.\nThe effect to reproduce is a red-above-'
           'green gap in the risky-second row and none in the risky-first row.')
    shown = [(l, n) for l, n in CANDIDATES if l in present]
    gs = fig.add_gridspec(2, len(shown), left=.055, right=.985, top=.845,
                          bottom=.10, hspace=.30, wspace=.20)
    for j, (lbl, nm) in enumerate(shown):
        s = stake[stake.label == lbl]
        for r, order in enumerate(['Risky first', 'Risky second']):
            ax = fig.add_subplot(gs[r, j])
            o = s[s.order == order]
            if not len(o):
                blank(ax)
            else:
                for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                    q = o[o.stim == stim].sort_values('stake_chf')
                    ax.fill_between(q.stake_chf, q.lo, q.hi, color=col,
                                    alpha=.20, lw=0)
                    ax.plot(q.stake_chf, q.model, color=col, lw=1.0)
                    ax.errorbar(q.stake_chf, q.observed, yerr=q.observed_sem,
                                fmt='o', ms=2.6, color=col, lw=0, elinewidth=.7,
                                capsize=0, zorder=4)
                ax.axhline(.5, color='0.88', lw=.5, ls='--', zorder=0)
                ax.set_ylim(.40, .74)
                ax.set_yticks([.45, .55, .65])
                ax.set_xscale('log')
                ax.set_xticks(sorted(o.stake_chf.unique()))
                ax.set_xticklabels([f'{v:.0f}' for v in
                                    sorted(o.stake_chf.unique())], fontsize=5.4)
                ax.minorticks_off()
                if j:
                    ax.set_yticklabels([])
            if r == 0:
                ax.set_title(nm, fontsize=6.4, pad=2)
            if j == 0:
                ax.set_ylabel(f'P(risky)\n{order.lower()}', fontsize=6.5)
    fig.text(.5, .022, 'Stake (CHF, within-subject terciles)', ha='center',
             fontsize=7)
    fig.text(.985, .012, stamp, fontsize=5.6, color='0.6', ha='right')
    sns.despine(fig=fig, offset=2)
    pdf.savefig(fig); plt.close(fig)


def page_probit(pdf, dd, present, stamp, ri):
    fig = plt.figure(figsize=PAGE)
    src = '.ri' if ri else ''
    fig_sub = ('Published random-intercept structure' if ri else
               'Full random-effects structure')
    header(fig, 'The published Figure-3 quantities, model against data',
           f'Low stake, risky second — the cell the effect lives in. Model '
           f'values are DERIVED in closed form from each posterior, never '
           f'fitted to these quantities.\nObserved: {fig_sub} '
           f'(probit_stake_group_posterior{src}.tsv).')
    g = pd.read_csv(dd / f'probit_stake_group_posterior{src}.tsv', **READ)
    p = g.pivot_table(index=['parameter', 'order', 'stake', 'draw'],
                      columns='stimulation_condition', values='value')
    p['d'] = p['ips'] - p['vertex']
    obs = {par: p.xs((par, 'Risky second', 'Low stake'))['d'].values
           for par in ('slope', 'rnp')}

    def dr(s):
        return np.array([float(v) for v in s.split(',')])

    shown = [(l, n) for l, n in CANDIDATES if
             (dd / f'probit_derived/probit_derived.{l}.tsv').exists()]
    gs = fig.add_gridspec(1, 2, left=.22, right=.97, top=.84, bottom=.10,
                          wspace=.22)
    y = np.arange(len(shown))[::-1]
    for k, par in enumerate(['slope', 'rnp']):
        ax = fig.add_subplot(gs[k])
        lo, md, hi = np.quantile(obs[par], [.025, .5, .975])
        ax.axvspan(lo, hi, color='#f0f0f0', zorder=0)
        ax.axvline(md, color=DATA, lw=1.2, zorder=1)
        ax.axvline(0, color='0.8', lw=.7, ls='--', zorder=0)
        for yy, (lbl, nm) in zip(y, shown):
            d = pd.read_csv(dd / f'probit_derived/probit_derived.{lbl}.tsv',
                            **READ)
            s = d[(d.order == 'Risky second') & (d.stake2 == 0)]
            q = s[s.parameter == ('slope' if par == 'slope' else 'logfrac_star')]
            i = dr(q[q.stimulation_condition == 'ips'].draws.iloc[0])
            v = dr(q[q.stimulation_condition == 'vertex'].draws.iloc[0])
            n = min(len(i), len(v))
            val = (i[:n] - v[:n]) if par == 'slope' else (np.exp(-i[:n])
                                                          - np.exp(-v[:n]))
            a, b, c = np.quantile(val, [.025, .5, .975])
            ax.plot([a, c], [yy] * 2, color=MODEL, lw=1.3, solid_capstyle='butt')
            ax.plot(b, yy, 'o', ms=4, color=MODEL)
        ax.set_yticks(y)
        ax.set_yticklabels([n for _, n in shown] if k == 0 else [], fontsize=7)
        ax.set_ylim(-.7, len(shown) - .3)
        ax.set_xlabel('Δ probit slope (consistency)' if par == 'slope'
                      else 'Δ risk-neutral probability (risk attitude)')
        if k == 0:
            ax.text(.03, .03, 'Grey band: data 95% CrI', transform=ax.transAxes,
                    fontsize=6, color='0.4')
    fig.text(.985, .012, stamp, fontsize=5.6, color='0.6', ha='right')
    sns.despine(fig=fig, offset=3)
    pdf.savefig(fig); plt.close(fig)


def main(data_dir, out_pdf, ri):
    dd = Path(data_dir)
    curves = pd.read_csv(dd / 'anchor_curves.tsv', **READ)
    loo = cat('loo_anchor/loo.*.tsv', dd, ['label', 'elpd_loo', 'se', 'n_par'])
    ess = (pd.read_csv(dd / 'anchor_ess.tsv', **READ)
           if (dd / 'anchor_ess.tsv').exists()
           else pd.DataFrame(columns=['label', 'min_ess_group',
                                      'max_rhat_group']))
    stake = cat('ppc_anchor/ppc_anchor.stake.*.tsv', dd,
                ['label', 'order', 'stim', 'model', 'lo', 'hi', 'observed',
                 'observed_sem', 'stake_chf'])
    present = set(curves.label) | set(loo.label)
    have = [l for l, _ in CANDIDATES if l in present]
    stamp = (f'{len(have)}/{len(CANDIDATES)} candidates present · '
             f'PRIOR_SPEC v1-2026-08-28')
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        page_elpd(pdf, loo, ess, present, stamp)
        page_curves(pdf, curves[curves.label.isin(have)], present, stamp,
                    'levels', 'Fitted noise functions',
                    'What each candidate says the observer is like. σ is the SD '
                    'of the log-payoff percept, so a flat line is Weber\'s law.')
        page_curves(pdf, curves[curves.label.isin(have)], present, stamp,
                    'delta', 'The cTBS effect on the noise function',
                    'Δσ = IPS − vertex, differenced within draw. The null '
                    'models are flat at zero by construction and are the '
                    'reference.')
        page_ppc(pdf, stake, present, stamp)
        page_probit(pdf, dd, present, stamp, ri)
    print(f'wrote {out_pdf}  ({len(have)} candidates)')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--out_pdf',
                    default=str(REPO / 'notes/figures/candidate_report.pdf'))
    ap.add_argument('--ri', action='store_true',
                    help='compare against the published random-intercept probit')
    a = ap.parse_args()
    main(a.data_dir, a.out_pdf, a.ri)
