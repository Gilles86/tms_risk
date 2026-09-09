"""Is the decoded magnitude of an option shifted by cTBS, within payoff level?

The naive version of this test does not work. `decode.py` evaluates the
likelihood on a bounded grid with no prior, which is a FLAT prior on an
interval, and a flat prior on an interval pulls the posterior mean toward the
grid's centre. On the log-space grid that produced a regressive bias of +0.4 log
units on small payoffs and −0.4 on large ones, with a slope of 0.92 toward the
centre — nothing to do with the brain.

The fix is to compare WITHIN PAYOFF LEVEL. For a given true payoff the grid's
pull is identical under both stimulation conditions, so it cancels in the
IPS − vertex contrast, and what is left is interpretable.

Reads the derivative the paper's decoding analyses use
(`decoded_pdfs.volume.cv_voxel_selection.denoise.natural_space`,
cross-validated voxel selection), decodes the FIRST-presented option — which is
the safe option on risky-second trials and the risky option on risky-first
trials — and asks whether the cTBS contrast differs by that role.

    python -m tms_risk.behavior.scripts.decoded_bias_by_payoff --mask NPC12r
"""
import argparse
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DEC = ('derivatives/decoded_pdfs.volume.cv_voxel_selection.denoise'
       '.natural_space')


def load(bids_folder, mask):
    rows = []
    pat = f'{bids_folder}/{DEC}/sub-*/func/*mask-{mask}_space-T1w_pars.tsv'
    for f in sorted(glob(pat)):
        m = re.search(r'sub-(\d+)_ses-(\d)', os.path.basename(f))
        if not m:
            continue
        d = pd.read_csv(f, sep='\t')
        grid = np.array([float(c) for c in d.columns[2:]])
        p = d.iloc[:, 2:].values.astype(float)
        s = p.sum(1, keepdims=True)
        ok = (s[:, 0] > 0) & np.isfinite(s[:, 0])
        q = d[['stimulation_condition', 'trial_nr']].copy()
        q['E'] = np.where(ok, (p / np.where(s > 0, s, 1) * grid).sum(1), np.nan)
        q['subject'] = int(m.group(1))
        q['session'] = int(m.group(2))
        rows.append(q)
    return pd.concat(rows, ignore_index=True)


def main(bids_folder, mask, out_tsv):
    from tms_risk.utils.data import get_all_behavior
    dec = load(bids_folder, mask)
    beh = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True,
                           exclude_outliers=True).reset_index()
    beh = beh[beh.session.isin([2, 3])]
    # trial_nr in the decoded files runs 1..120 across the session; the
    # behaviour table carries run and trial_nr separately
    beh = beh.sort_values(['subject', 'session', 'run', 'trial_nr'])
    beh['tn'] = beh.groupby(['subject', 'session']).cumcount() + 1
    j = dec.merge(beh[['subject', 'session', 'tn', 'n1', 'risky_first']],
                  left_on=['subject', 'session', 'trial_nr'],
                  right_on=['subject', 'session', 'tn'], how='inner').dropna(
                      subset=['E'])
    j['role'] = np.where(j.risky_first.astype(bool), 'risky', 'safe')
    print(f'{len(j)} trials, {j.subject.nunique()} participants, mask {mask}')

    # WITHIN PAYOFF LEVEL: the grid's pull is the same in both conditions, so
    # it drops out of the contrast
    cell = (j.groupby(['subject', 'role', 'n1', 'stimulation_condition'])['E']
            .mean().unstack('stimulation_condition'))
    cell = cell.dropna()
    cell['d'] = cell['ips'] - cell['vertex']
    # average over payoff levels within participant, then treat participants as
    # the sampling unit
    per_sub = cell.groupby(level=['subject', 'role'])['d'].mean().unstack('role')
    rows = []
    for role in [c for c in ('safe', 'risky') if c in per_sub]:
        v = per_sub[role].dropna()
        rows.append(dict(mask=mask, role=role, n=len(v), mean=v.mean(),
                         sem=v.sem(), t=v.mean() / v.sem()))
    if {'safe', 'risky'} <= set(per_sub.columns):
        v = (per_sub['safe'] - per_sub['risky']).dropna()
        rows.append(dict(mask=mask, role='safe − risky', n=len(v),
                         mean=v.mean(), sem=v.sem(), t=v.mean() / v.sem()))
    out = pd.DataFrame(rows)
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep='\t', index=False)
    print('\ncTBS effect on the decoded magnitude of the FIRST-presented '
          'option (CHF), within payoff level:')
    print(out.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print(f'\nwrote {out_tsv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--mask', default='NPC12r')
    ap.add_argument('--out_tsv', default=None)
    a = ap.parse_args()
    main(a.bids_folder, a.mask,
         a.out_tsv or str(REPO / f'notes/data/decoded_bias_{a.mask}.tsv'))
