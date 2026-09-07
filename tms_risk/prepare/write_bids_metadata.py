"""Make the BIDS dataset describe itself, instead of relying on this repository.

Two things about `/data/ds-tmsrisk` only lived in the code until now: the cTBS
stimulation site of each session (`tms_risk/data/tms_keys.yml`) and the meaning of
the `_events.tsv` columns. This script writes both into the dataset, in the
BIDS-standard places:

    <bids>/README                              study, design and known quirks
    <bids>/sessions.json                       sidecar for the session tables
    <bids>/participants.json                   sidecar for participants.tsv
    <bids>/task-task_events.json               sidecar for every _events.tsv
    <bids>/sub-XX/sub-XX_sessions.tsv          session_id, stimulation

The four root files are static and live in `tms_risk/data/bids_metadata/`; the
session tables are generated from `tms_keys.yml`, which stays the authoritative
source. Session 1 is the pre-TMS baseline / nPRF-mapping session and is written
as ``baseline``. Sessions with no data on disk (empty leftover directories) are
skipped.

Usage
-----
    python -m tms_risk.prepare.write_bids_metadata --bids_folder /data/ds-tmsrisk
    python -m tms_risk.prepare.write_bids_metadata --check        # verify only
"""

import argparse
import sys
from importlib.resources import files
from pathlib import Path

import pandas as pd

from tms_risk.utils.data import get_tms_conditions

STATIC_FILES = ['README', 'sessions.json', 'participants.json', 'task-task_events.json']


def _static_source(name):
    return files('tms_risk').joinpath('data/bids_metadata', name)


def _session_has_data(session_dir):
    """True if the session directory holds actual data (ignoring dotfiles like .DS_Store)."""
    return any((p.is_file() or p.is_symlink()) and not p.name.startswith('.')
               for p in session_dir.rglob('*'))


def build_sessions_table(subject, bids_folder):
    """Return the sessions.tsv contents for one subject, or None if it has no data."""
    subject = f'{int(subject):02d}'
    subject_dir = Path(bids_folder) / f'sub-{subject}'
    tms_conditions = get_tms_conditions().get(subject, {})

    rows = []
    for session_dir in sorted(subject_dir.glob('ses-*')):
        session = int(session_dir.name.split('-')[1])

        if not _session_has_data(session_dir):
            continue

        if session == 1:
            stimulation = 'baseline'
        else:
            stimulation = tms_conditions.get(session, 'n/a')

        rows.append({'session_id': session_dir.name, 'stimulation': stimulation})

    if len(rows) == 0:
        return None

    return pd.DataFrame(rows)


def main(bids_folder='/data/ds-tmsrisk', check=False, dry_run=False):
    bids_folder = Path(bids_folder)
    subject_dirs = sorted(bids_folder.glob('sub-*'))

    problems = []
    n_written = 0

    for name in STATIC_FILES:
        expected = _static_source(name).read_text()
        target = bids_folder / name

        if check:
            if not target.exists():
                problems.append(f'{target} is missing')
            elif target.read_text() != expected:
                problems.append(f'{target} differs from tms_risk/data/bids_metadata/{name}')
        else:
            print(f'{target}: written')
            if not dry_run:
                target.write_text(expected)
                n_written += 1

    for subject_dir in subject_dirs:
        subject = subject_dir.name.split('-')[1]
        table = build_sessions_table(subject, bids_folder)

        if table is None:
            continue

        missing = table[table['stimulation'] == 'n/a']
        if len(missing) > 0:
            problems.append(f'{subject_dir.name}: no tms_keys.yml entry for '
                            f'{list(missing["session_id"])}')

        target = subject_dir / f'{subject_dir.name}_sessions.tsv'

        if check:
            if not target.exists():
                problems.append(f'{target} is missing')
            else:
                on_disk = pd.read_csv(target, sep='\t', keep_default_na=False)
                if not on_disk.equals(table):
                    problems.append(f'{target} disagrees with tms_keys.yml:\n'
                                    f'  on disk:  {on_disk.to_dict("records")}\n'
                                    f'  expected: {table.to_dict("records")}')
        else:
            print(f'{target}: ' + ', '.join(f'{r.session_id}={r.stimulation}'
                                            for r in table.itertuples()))
            if not dry_run:
                table.to_csv(target, sep='\t', index=False, na_rep='n/a')
                n_written += 1

    if problems:
        print('\nPROBLEMS:', file=sys.stderr)
        for problem in problems:
            print(f' - {problem}', file=sys.stderr)
        return 1

    if check:
        print(f'{bids_folder} is in sync with tms_keys.yml and '
              'tms_risk/data/bids_metadata/ ({} subject folders).'.format(len(subject_dirs)))
    else:
        print(f'\nWrote {n_written} files.\n'
              'The dataset is git-annex/datalad managed, so if you track it, save just '
              'these files:\n'
              '  datalad save -m "Describe sessions and stimulation site in the dataset" \\\n'
              '    README sessions.json participants.json task-task_events.json '
              'sub-*/sub-*_sessions.tsv')

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--check', action='store_true',
                        help='Verify the dataset against this repo; write nothing.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be written without touching the dataset.')
    args = parser.parse_args()

    sys.exit(main(args.bids_folder, check=args.check, dry_run=args.dry_run))
