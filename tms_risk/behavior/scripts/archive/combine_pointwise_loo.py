"""Build the master ELPD ladder from per-trace pointwise LOO files.

Why this exists: the models the paper compares no longer live on one machine. The
log-space lfx2 grid is on sciencecluster (`derivatives/cogmodels.lfxgrid`), the
natural-space head refits and the power-law family are on the sciencecloud VM
(`cogmodels.{overnight,ladder,power}`), and the traces are ~1.2 GB each, so a single
`az.compare` over the set is impossible. Everything the table needs -- ELPD, SE,
ELPD difference and the *paired* dSE -- is a function of the pointwise elpd_i vectors
alone, and those are 67 KB. `pointwise_loo.py` writes one npz per trace wherever the
trace lives; this script combines them.

The paired dSE is the whole point. `se` is the standard error of a model's own ELPD and
is huge (~46 nats here) because it is dominated by between-subject variance that is
*common to every model*. dSE differences that variance away trial by trial:

    dSE(a, b) = sqrt(n * var_i(elpd_i[a] - elpd_i[b]))

which is why a 90-nat gap at se = 46 is nonetheless a 10-dSE result. Quoting `se`
where the paper means `dse` understates every comparison by roughly fivefold.

Gate: every trace must carry the identical observed-data vector (same trials, same
order). A mismatch makes the pairing meaningless, so it aborts rather than warns.

    python -m tms_risk.behavior.scripts.combine_pointwise_loo \\
        --ploo_dir notes/data/ploo --reference lfx2-bs3-m2-dp-bm \\
        --out_stem notes/data/ladder_v11
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(ploo_dirs):
    """Read every npz; return {label: (elpd_i, meta)} and the shared observed vector."""
    elpd, meta, obs_hashes, obs = {}, {}, {}, None
    for d in ploo_dirs:
        for p in sorted(Path(d).glob('*.npz')):
            z = np.load(p, allow_pickle=False)
            m = json.loads(str(z['meta']))
            lab = m['label']
            if lab in elpd:
                lab = f"{lab}@{m['dir']}"
                m['label'] = lab
            elpd[lab] = z['elpd_i']
            meta[lab] = m
            obs_hashes[lab] = m['obs_hash']
            if obs is None:
                obs = z['obs']
    if not elpd:
        raise SystemExit(f'no .npz files found in {ploo_dirs}')
    return elpd, meta, obs_hashes, obs


def check_comparable(elpd, obs_hashes):
    """Abort unless every trace scored the identical observations in the same order."""
    hashes = set(obs_hashes.values())
    if len(hashes) > 1:
        by = {}
        for lab, h in obs_hashes.items():
            by.setdefault(h, []).append(lab)
        lines = [f'  {h}  n={len(v):2d}  e.g. {v[0]}' for h, v in by.items()]
        raise SystemExit('observed data differ across traces -- the comparison would '
                         'be meaningless:\n' + '\n'.join(lines))
    n = {len(v) for v in elpd.values()}
    if len(n) > 1:
        raise SystemExit(f'pointwise ELPD lengths differ: {sorted(n)}')
    return n.pop()


def stacking_weights(mat):
    """Yao et al. (2018) stacking of predictive distributions.

    Maximizes sum_i log(sum_k w_k exp(elpd_ik)) over the simplex, via a softmax
    reparameterization so the optimizer stays unconstrained. Returns uniform weights
    if scipy is unavailable -- the weights are decoration here, the ELPD/dSE columns
    are what the paper quotes.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return np.full(mat.shape[0], 1 / mat.shape[0])

    # exp(elpd) underflows; shift by the per-observation max, which cancels in the ratio.
    shifted = np.exp(mat - mat.max(axis=0, keepdims=True))

    def neg_ll(free):
        w = np.exp(np.concatenate([[0.0], free]))
        w = w / w.sum()
        return -np.log(np.maximum(w @ shifted, 1e-300)).sum()

    res = minimize(neg_ll, np.zeros(mat.shape[0] - 1), method='BFGS')
    w = np.exp(np.concatenate([[0.0], res.x]))
    return w / w.sum()


def build(elpd, meta, reference, n):
    labels = list(elpd)
    mat = np.vstack([elpd[l] for l in labels])
    totals = mat.sum(axis=1)

    best = labels[int(np.argmax(totals))]
    if reference is None:
        reference = best
    if reference not in elpd:
        raise SystemExit(f'reference {reference!r} not among:\n  ' + '\n  '.join(labels))

    weights = dict(zip(labels, stacking_weights(mat)))
    rows = []
    for lab in labels:
        d_ref = elpd[lab] - elpd[reference]
        d_best = elpd[lab] - elpd[best]
        m = meta[lab]
        rows.append({
            'label': lab,
            'elpd_loo': elpd[lab].sum(),
            'se': np.sqrt(n * np.var(elpd[lab], ddof=1)),
            'p_loo': m['p_loo'],
            'elpd_diff_ref': d_ref.sum(),
            # var of an all-zero difference is 0; keep the self-comparison as 0, not NaN.
            'dse_ref': np.sqrt(n * np.var(d_ref, ddof=1)) if lab != reference else 0.0,
            'z_ref': (d_ref.sum() / np.sqrt(n * np.var(d_ref, ddof=1))
                      if lab != reference else np.nan),
            'elpd_diff_best': d_best.sum(),
            'dse_best': np.sqrt(n * np.var(d_best, ddof=1)) if lab != best else 0.0,
            'weight': weights[lab],
            'max_rhat': m['max_rhat'],
            'min_ess': m['min_ess'],
            'divergences': m['divergences'],
            'n_group_par': m['n_group_par'],
            'dir': m['dir'],
            'bauer_commit': m['bauer_commit'][:7],
        })
    tab = pd.DataFrame(rows).sort_values('elpd_loo', ascending=False)
    tab['converged'] = (tab.max_rhat <= 1.01) & (tab.min_ess >= 400)
    return tab, reference, best


def markdown(tab, reference, n):
    head = (f'Reference model: `{reference}`. n = {n} trials. '
            'ELPD differences are paired: dSE is the SE of the trial-by-trial '
            'difference, not of the two ELPDs separately.\n')
    lines = [head,
             '| Model | ELPD | SE | p_loo | ΔELPD vs ref | dSE | z | r̂ | ESS | div |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in tab.iterrows():
        flag = '' if r.converged else ' ⚠'
        z = '—' if not np.isfinite(r.z_ref) else f'{r.z_ref:+.1f}'
        d = '—' if r.label == reference else f'{r.elpd_diff_ref:+.1f}'
        dse = '—' if r.label == reference else f'{r.dse_ref:.1f}'
        lines.append(
            f'| `{r.label}`{flag} | {r.elpd_loo:.1f} | {r.se:.1f} | {r.p_loo:.1f} | '
            f'{d} | {dse} | {z} | {r.max_rhat:.3f} | {r.min_ess:.0f} | {r.divergences} |')
    bad = tab.loc[~tab.converged, 'label'].tolist()
    if bad:
        lines += ['', '⚠ failed the convergence gate (r̂ ≤ 1.01, ESS ≥ 400): '
                      + ', '.join(f'`{b}`' for b in bad) + '.']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ploo_dir', nargs='+', required=True)
    ap.add_argument('--reference', default=None,
                    help='label to report differences against (default: best ELPD)')
    ap.add_argument('--out_stem', required=True)
    args = ap.parse_args()

    elpd, meta, obs_hashes, _ = load(args.ploo_dir)
    n = check_comparable(elpd, obs_hashes)
    print(f'{len(elpd)} traces, all scoring the identical {n} observations '
          f'(hash {next(iter(obs_hashes.values()))})\n')

    tab, reference, best = build(elpd, meta, args.reference, n)
    Path(args.out_stem).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(f'{args.out_stem}.tsv', sep='\t', index=False)
    md = markdown(tab, reference, n)
    Path(f'{args.out_stem}.md').write_text(md + '\n')
    print(md)
    print(f'\nbest ELPD: {best}')
    print(f'wrote {args.out_stem}.tsv and {args.out_stem}.md')


if __name__ == '__main__':
    main()
