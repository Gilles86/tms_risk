"""Where the anchor grid's low ESS actually lives, and whether draws would fix it.

For every trace: ESS split into GROUP-level parameters (the ones anybody reports)
and SUBJECT-level ones, plus the posterior correlation between the anchor
intercepts within a channel. That correlation is the thing to look at before
buying more draws -- two anchors that trade off produce a ridge, and a ridge
costs ESS at a rate no number of draws fixes cheaply.

It is also predictable BEFORE fitting: `AnchorNoiseMixin.anchor_correlation`
returns the off-diagonal of the design Gram matrix at the payoffs actually
presented. Reported here next to the posterior correlation so the two can be
compared.

    python -m tms_risk.behavior.scripts.diagnose_anchor_ess \\
        --trace_dir .../cogmodels.anchor --out_tsv /home/gdehol/anchor_ess.tsv
"""
import argparse
import re
import warnings
from glob import glob
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')

NAME_RE = re.compile(r'^(log|chf)_(perc|mem|n1|n2)_'
                     r'(weber|affine|power|genweber|spl3|spl5|spl7|spl9|cspl3|cspl5|cspl7)_sd(\d*)$')


def label_from_path(path, attrs):
    """Label a trace by its FILENAME, not by its stamp.

    A variant refit (`...pathfinder_trace.netcdf`) carries the base label in
    `tms_risk_label`, so trusting the stamp collapses the variant and the
    original into one row.
    """
    stem = Path(path).name.replace('model-', '').replace('_trace.netcdf', '')
    return stem or attrs.get('tms_risk_label')


def main(trace_dir, out_tsv, bids_folder):
    design = {}
    if bids_folder:
        from tms_risk.behavior.fit_model import get_data
        from bauer.models.anchor_noise import NOISE_FORMS
        from tms_risk.behavior.fit_anchor import build_model
        df = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm')
        for form in NOISE_FORMS:
            for space in ('log', 'chf'):
                try:
                    m = build_model(f'{space}-{form}-null', df.copy())
                    C = m.anchor_correlation()
                    design[(space, form)] = float(np.abs(
                        C[np.triu_indices_from(C, 1)]).max()) if C.size > 1 else 0.0
                except Exception as e:                     # noqa: BLE001
                    design[(space, form)] = np.nan
                    print(f'  design corr {space}-{form}: {e}')

    rows = []
    for path in sorted(glob(str(Path(trace_dir) / 'model-*_trace.netcdf'))):
        idata = az.from_netcdf(path)
        post = idata.posterior
        a = post.attrs
        label = label_from_path(path, a)
        if label is None:
            continue
        grp = [v for v in post.data_vars if 'subject' not in post[v].dims]
        sub = [v for v in post.data_vars
               if 'subject' in post[v].dims and not v.endswith('_offset')]
        sg = az.summary(post, var_names=grp, kind='diagnostics')
        ss = az.summary(post, var_names=sub, kind='diagnostics')

        # posterior correlation between anchor INTERCEPTS within each channel
        worst_corr, worst_pair = 0.0, ''
        anchors = [v for v in post.data_vars if NAME_RE.match(v)]
        by_chan = {}
        for v in anchors:
            by_chan.setdefault(NAME_RE.match(v).group(2), []).append(v)
        for chan, vs in by_chan.items():
            if len(vs) < 2:
                continue
            vs = sorted(vs, key=lambda n: float(NAME_RE.match(n).group(4) or 0))
            # group mean of the intercept, flattened over chain x draw
            M = np.stack([post[f'{v}_mu'].isel({f'{v}_regressors': 0})
                          .stack(s=('chain', 'draw')).values for v in vs])
            C = np.corrcoef(M)
            iu = np.triu_indices_from(C, 1)
            k = int(np.argmax(np.abs(C[iu])))
            if abs(C[iu][k]) > abs(worst_corr):
                worst_corr = float(C[iu][k])
                worst_pair = f'{vs[iu[0][k]]} ~ {vs[iu[1][k]]}'

        # The flat direction Rule A is supposed to have: with a SHARED prior and
        # shared noise, w cancels from the psychometric slope and (mu_R, sd_R,
        # mu_S, sd_S) are only identified through the threshold. Whether that
        # survives here is an empirical question, because w = sd_p^2/(sd_p^2 +
        # nu^2) and nu differs between the first- and second-presented option --
        # and presentation order is randomised, so the SAME prior parameter is
        # paired with both noise levels across trials. Measure it rather than
        # assume it.
        # GROUP-level prior parameters only. The subject-level variable is
        # `log_risky_prior_mu`; its group mean is `log_risky_prior_mu_mu`. A
        # bare `endswith('_mu')` catches both, and the subject-level one has no
        # `<name>_regressors` dim under that truncation.
        pri = [v for v in post.data_vars
               if re.search(r'_prior_(mu|sd)_mu$', v)
               and 'subject' not in post[v].dims]
        prior_corr, prior_pair = 0.0, ''
        if len(pri) > 1:
            M = []
            for v in pri:
                rdim = f'{v[:-3]}_regressors'
                M.append(post[v].isel({rdim: 0}).stack(s=('chain', 'draw')).values)
            C = np.corrcoef(np.stack(M))
            iu = np.triu_indices_from(C, 1)
            k = int(np.argmax(np.abs(C[iu])))
            prior_corr = float(C[iu][k])
            prior_pair = f'{pri[iu[0][k]]} ~ {pri[iu[1][k]]}'

        row = dict(
            label=label,
            choice_noise=a.get('tms_risk_choice_noise', 'raw_evidence_sd'),
            prior_corr=prior_corr, prior_pair=prior_pair, space=a['tms_risk_space'], form=a['tms_risk_noise_form'],
            placement=a['tms_risk_placement'],
            draws=int(post.sizes['draw']), chains=int(post.sizes['chain']),
            n_anchors=len(a['tms_risk_anchors'].split(',')),
            divergences=int(idata.sample_stats['diverging'].values.sum()),
            max_rhat_group=float(sg['r_hat'].max()),
            min_ess_group=float(sg['ess_bulk'].min()),
            worst_group_par=str(sg['ess_bulk'].idxmin()),
            max_rhat_subject=float(ss['r_hat'].max()),
            min_ess_subject=float(ss['ess_bulk'].min()),
            worst_subject_par=str(ss['ess_bulk'].idxmin()),
            anchor_corr_posterior=worst_corr, anchor_corr_pair=worst_pair,
            anchor_corr_design=design.get((a['tms_risk_space'],
                                           a['tms_risk_noise_form']), np.nan),
        )
        row['ok_group'] = bool(row['max_rhat_group'] <= 1.01
                               and row['min_ess_group'] >= 400)
        rows.append(row)
        print(f"  {label:24s} group ESS {row['min_ess_group']:6.0f} "
              f"({row['worst_group_par']:28s})  subj ESS "
              f"{row['min_ess_subject']:6.0f}  anchor-rho {worst_corr:+.2f}  "
              f"prior-rho {prior_corr:+.2f}")
        idata.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv} ({len(df)} traces)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace_dir', required=True)
    ap.add_argument('--out_tsv', default='anchor_ess.tsv')
    ap.add_argument('--bids_folder', default='/shares/zne.uzh/gdehol/ds-tmsrisk')
    args = ap.parse_args()
    main(args.trace_dir, args.out_tsv, args.bids_folder)
