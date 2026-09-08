"""Per-participant cTBS contrast for EVERY free parameter of an anchor fit.

The group-level panels of Figure 5 are hierarchical means, and a hierarchical
mean can look tidy while individual participants disagree wildly. This dumps the
IPS - vertex contrast one participant at a time, for every parameter the model
lets vary, so the spread behind each group estimate is visible.

Every regressor in these models is the cTBS contrast, coded as
`stimulation_condition[T.vertex]`: index 0 of the regressor dimension is the IPS
intercept and index 1 the vertex offset. So the IPS - vertex contrast is simply
MINUS the second coefficient -- no curve machinery, no design matrix, and the
credible interval is on the contrast itself rather than on a difference of two
summaries.

Parameters live on the log scale, so the contrast is reported both in log units
(additive, what the model samples) and as a percentage change (readable).

    python -m tms_risk.behavior.scripts.extract_anchor_subject_params \\
        log-power-n2psd --trace_dir .../cogmodels.anchor --out_dir notes/data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

def ctbs_contrast(coef, colnames):
    """IPS - vertex from a regressor axis, whatever the contrast coding.

    Treatment coding (`stimulation_condition`) puts the IPS cell in the
    intercept and vertex - IPS in column 1, so the contrast is -coef[1]. Sum
    coding (`C(stimulation_condition, Sum)`) puts the grand mean in the
    intercept and (IPS - grand mean) in column 1, with vertex = -that, so the
    contrast is 2*coef[1]. Getting this wrong is silent -- same shape, same
    sign in some models -- so it is read off the column NAME, never assumed.
    """
    name = str(colnames[1])
    if 'Sum)' in name or '[S.' in name:
        return 2.0 * coef[..., 1]
    if 'T.vertex' in name:
        return -coef[..., 1]
    raise ValueError(f'unrecognised cTBS regressor column {name!r}')



def main(label, trace_dir, out_dir):
    ds = xr.open_dataset(Path(trace_dir) / f'model-{label}_trace.netcdf',
                         group='posterior')
    names = ds.attrs['tms_risk_parameters'].split(',')
    subj = [str(s) for s in ds['subject'].values]
    rows = []
    for p in names:
        rdim = f'{p}_regressors'
        if p not in ds or ds.sizes.get(rdim, 1) < 2:
            continue                      # no cTBS term on this parameter
        coef = (ds[p].stack(sample=('chain', 'draw'))
                .transpose('subject', 'sample', rdim).values)
        # IPS is the intercept; the second column is the vertex OFFSET, so the
        # IPS - vertex contrast is its negation
        delta = ctbs_contrast(coef, ds[rdim].values)   # (subject, sample)
        grp = delta.mean(axis=0)                    # the group mean, per draw
        for i, sub in enumerate(subj):
            d = delta[i]
            rows.append(dict(label=label, parameter=p, subject=sub,
                             mid=float(np.median(d)),
                             lo=float(np.quantile(d, .025)),
                             hi=float(np.quantile(d, .975)),
                             pct=float(100 * np.expm1(np.median(d))),
                             p_gt0=float((d > 0).mean())))
        rows.append(dict(label=label, parameter=p, subject='GROUP',
                         mid=float(np.median(grp)),
                         lo=float(np.quantile(grp, .025)),
                         hi=float(np.quantile(grp, .975)),
                         pct=float(100 * np.expm1(np.median(grp))),
                         p_gt0=float((grp > 0).mean())))
    ds.close()
    out_dir = Path(out_dir)
    (out_dir / 'subject_params').mkdir(parents=True, exist_ok=True)
    f = out_dir / 'subject_params' / f'subject_params.{label}.tsv'
    pd.DataFrame(rows).to_csv(f, sep='\t', index=False)
    print(f'wrote {f} ({len(rows)} rows, '
          f'{len(set(r["parameter"] for r in rows))} parameters)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_dir', default='notes/data')
    a = ap.parse_args()
    main(a.label, a.trace_dir, a.out_dir)
