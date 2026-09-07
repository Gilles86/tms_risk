"""Per-subject, per-condition prior mean and width.

Needed because the mechanism panels of Figure 5 are a NONLINEAR function of the
parameters -- perceived value depends on w = sd^2 / (sd^2 + nu^2) -- so the
average over participants of that function is not the function at the average
parameters. Computing it from group-level parameters gives "a hypothetical
average participant", which is a different object from "the mean effect across
participants", and the two diverge in proportion to the between-subject
variance. For `psd` models the per-subject prior-width contrast has SD 0.33,
comparable to its own intercept, so the gap is not small.

`extract_anchor_curves` already writes per-subject noise (sigma_ips /
sigma_vertex); this is the missing half.

    python -m tms_risk.behavior.scripts.extract_anchor_priors_subject \
        --trace_dir .../cogmodels.anchor --out_dir priors_subject
"""
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main(trace_dir, out_dir, labels):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(glob(str(Path(trace_dir) / 'model-*_trace.netcdf'))):
        ds = xr.open_dataset(path, group='posterior')
        a = ds.attrs
        label = a.get('tms_risk_label')
        if label is None or (labels and label not in labels):
            ds.close()
            continue
        space = a['tms_risk_space']
        subs = list(ds['subject'].values) if 'subject' in ds.dims else []
        for which in ('risky', 'safe'):
            for kind in ('mu', 'sd'):
                v = f'{space}_{which}_prior_{kind}'
                if v not in ds:
                    continue
                rdim = [d for d in ds[v].dims if 'regressors' in d]
                if not rdim:
                    continue
                regs = [str(r) for r in ds[rdim[0]].values]
                x = ds[v].stack(s=('chain', 'draw')).values   # (subj, reg, draw)
                has = len(regs) > 1
                for i, sub in enumerate(subs):
                    # bambi treatment coding: reference level is IPS
                    icpt = x[i, 0]
                    for cond, val in [('ips', icpt),
                                      ('vertex', icpt + (x[i, 1] if has else 0))]:
                        val = np.exp(val)          # both mu and sd are on a log scale
                        rows.append(dict(label=label, subject=int(sub),
                                         which=which, kind=kind, condition=cond,
                                         value=float(np.median(val)),
                                         varies=int(has)))
        ds.close()
    d = pd.DataFrame(rows)
    f = out_dir / 'anchor_priors_subject.tsv'
    d.to_csv(f, sep='\t', index=False)
    print(f'wrote {f}  ({d.label.nunique()} models, '
          f'{d.subject.nunique() if len(d) else 0} subjects)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_dir', default='priors_subject')
    ap.add_argument('--labels', nargs='*', default=None)
    a = ap.parse_args()
    main(a.trace_dir, a.out_dir, set(a.labels) if a.labels else None)
