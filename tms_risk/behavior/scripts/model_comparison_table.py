"""Rebuild the paper's ELPD model-comparison table (Table 1).

There is already a notebook for this — `behavior/notebooks/comprehensive_model_comparison.ipynb` —
but it cannot currently produce correct numbers, for two reasons:

  * It has no saved outputs, so there is no record of what it actually produced.
  * It calls `build_model()` against whatever bauer happens to be importable and then
    `pm.compute_log_likelihood`. Because bauer's likelihood has drifted (see CLAUDE.md),
    that silently evaluates the stored posteriors under a *different* choice rule than
    the one they were fitted with — no error, wrong ELPD.

This script fixes both: bauer is pinned explicitly, and every model must pass a
posterior-predictive grand-mean check before its log-likelihood is trusted. A model
that fails the gate is reported as such rather than quietly contributing a wrong row.

    # published traces (5-spline flexible family + Weber family)
    python -m tms_risk.behavior.scripts.model_comparison_table \\
        --bauer_path /tmp/bauer_ecc6454 --thin 2

    # the corrected refits, which already carry log_likelihood
    python -m tms_risk.behavior.scripts.model_comparison_table \\
        --cogmodels_dir derivatives/cogmodels.noisefix --labels-from-dir
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
_bauer_path = None
for _i, _a in enumerate(sys.argv):
    if _a == '--bauer_path':
        _bauer_path = sys.argv[_i + 1]
sys.path.insert(0, _bauer_path or str(REPO / 'libs' / 'bauer'))
sys.path.insert(0, str(REPO / 'tms_risk' / 'behavior'))

import arviz as az        # noqa: E402
import pymc as pm         # noqa: E402
import bauer              # noqa: E402
import bauer.models as bm  # noqa: E402
from fit_model import get_data   # noqa: E402
import fit_model                 # noqa: E402

# Row names for the comparison. VERIFIED 2026-07-30 against the traces themselves
# (counting stimulation regressors per noise term), which is authoritative and needs
# no git archaeology:
#     11a -> TMS on memory AND perceptual      11b -> memory only
#     11c -> perceptual only                   11_null -> nothing
# comprehensive_model_comparison.ipynb maps 11a->"memory only", 11b->"perception only",
# 11c->"both", i.e. a cyclic permutation, so the three Weber rows of the published
# Table 1 carry the wrong names. The ELPD values are fine; the labels are swapped.
PAPER_ROWS = {
    'flexible2': 'Flexible PMC model (TMS affects both perception and working memory)',
    'flexible2a': 'Flexible PMC model (TMS affects working memory only)',
    'flexible2b': 'Flexible PMC model (TMS affects perception only)',
    'flexible2_null': 'Flexible PMC null model',
    '11a': 'Weber PMC model (TMS affects both perception and memory)',
    '11b': 'Weber PMC model (TMS affects memory only)',
    '11c': 'Weber PMC model (TMS affects perception only)',
    '11_null': 'Weber PMC null model',
}
PUBLISHED_ROW = {   # what Table 1 currently calls each label
    '11a': 'Weber PMC model (TMS affects memory only)',
    '11b': 'Weber PMC model (TMS affects perception only)',
    '11c': 'Weber PMC model (TMS affects both perception and memory)',
}
DISPATCH_MEANING = {
    '11a': 'both noise terms', '11b': 'memory only', '11c': 'perception only',
    '11_null': 'no TMS regressor',
    'flexible2': 'both noise terms', 'flexible2a': 'memory only',
    'flexible2b': 'perception only', 'flexible2_null': 'no TMS regressor',
}
DEFAULT_LABELS = list(PAPER_ROWS)


def build(label, df):
    """Build a model for `label`, working around the spline_order/polynomial_order rename."""
    m = re.fullmatch(r'flexible2(\.\d)?(_noisefix)?(_null|a|b|_memory|_perception)?', label)
    if m and m.group(2) is None:
        spline_order = 5 if m.group(1) is None else int(m.group(1)[1:])
        suffix = m.group(3) or ''
        names = {'': ['memory_noise_sd', 'perceptual_noise_sd'], '_null': [],
                 'a': ['memory_noise_sd'], 'b': ['perceptual_noise_sd']}[suffix]
        import inspect
        cls = bm.FlexibleNoiseRiskRegressionModel
        key = ('spline_order' if 'spline_order' in inspect.signature(cls).parameters
               else 'polynomial_order')
        return cls(df, regressors={n: 'stimulation_condition' for n in names},
                   memory_model='shared_perceptual_noise', prior_estimate='full',
                   **{key: spline_order})
    return fit_model.build_model(label, df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--cogmodels_dir', default='derivatives/cogmodels',
                        help='relative to --bids_folder')
    parser.add_argument('--bauer_path', default=None)
    parser.add_argument('--labels', nargs='*', default=None)
    parser.add_argument('--thin', default=2, type=int,
                        help='keep every Nth draw (the notebook used 2)')
    parser.add_argument('--gate', default=0.02, type=float,
                        help='max |model - observed| grand-mean gap to trust a model')
    parser.add_argument('--out_stem',
                        default='/Users/gdehol/git/tms_risk/notes/data/model_comparison')
    args = parser.parse_args()

    print(f'bauer {Path(bauer.__file__).parent}')
    cog = Path(args.bids_folder) / args.cogmodels_dir
    labels = args.labels or DEFAULT_LABELS
    df = get_data(args.bids_folder)
    observed = df['choice'].astype(float).mean()
    print(f'data  {len(df)} trials, observed P(choose 2) = {observed:.4f}')

    idatas, rows = {}, []
    for label in labels:
        hits = sorted(cog.glob(f'model-{label}_trace.netcdf')) + \
               sorted(cog.glob(f'model-{label}.*_trace.netcdf'))
        if not hits:
            print(f'  {label:24s} MISSING'); continue
        idata = az.from_netcdf(str(hits[0]))
        if args.thin > 1:
            idata = idata.sel(draw=slice(None, None, args.thin))

        model = build(label, df)
        model.build_estimation_model(save_p_choice=True)
        det = pm.compute_deterministics(idata.posterior, model=model.estimation_model,
                                        var_names=['p'], merge_dataset=False,
                                        progressbar=False)
        pred = float(np.nanmean(det['p'].values))
        gap = abs(pred - observed)
        ok = gap <= args.gate
        print(f'  {label:24s} draws={idata.posterior.sizes["draw"]:5d}  '
              f'model P={pred:.4f}  gap={gap:.4f}  {"OK" if ok else "GATE FAILED"}')
        if not ok:
            rows.append({'label': label, 'status': f'gate failed (gap {gap:.3f})'})
            continue
        if 'log_likelihood' not in idata.groups():
            with model.estimation_model:
                pm.compute_log_likelihood(idata, progressbar=False)
        idatas[label] = idata
        rows.append({'label': label, 'status': 'ok', 'ppc_gap': gap})

    if len(idatas) < 2:
        raise SystemExit('need at least two models past the gate to compare')

    comparison = az.compare(idatas, ic='loo')
    comparison.index.name = 'label'
    out = comparison.reset_index()
    out['paper_row'] = out['label'].map(PAPER_ROWS)
    out['dispatch_meaning'] = out['label'].map(DISPATCH_MEANING)
    out['published_row'] = out['label'].map(PUBLISHED_ROW)
    out = out.merge(pd.DataFrame(rows)[['label', 'ppc_gap']], on='label', how='left')

    stem = Path(args.out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(f'{stem}.tsv', sep='\t', index=False)

    # Table 1's own columns and ordering
    tbl = out.rename(columns={'elpd_loo': 'ELPD (LOO)', 'p_loo': 'Effective n params',
                              'elpd_diff': 'Difference in ELPD', 'se': 'SE', 'dse': 'dSE'})
    cols = ['paper_row', 'label', 'ELPD (LOO)', 'Effective n params',
            'Difference in ELPD', 'SE', 'dSE', 'dispatch_meaning']
    tbl = tbl[cols].round(2)
    print('\n=== Table 1, rebuilt ===')
    print(tbl.to_string(index=False))
    Path(f'{stem}.md').write_text(tbl.to_markdown(index=False))
    print(f'\nwrote {stem}.tsv and {stem}.md')


if __name__ == '__main__':
    main()
