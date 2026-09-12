"""Import the autoflatten flatmaps into the pycortex subjects ``tms.sub-XX``.

The patches are made on the cluster by ``surface/slurm_jobs/autoflatten.sh`` from
the local FreeSurfer recon; pull them back first:

    for s in 01 02 ...; do rsync -a sciencecluster:/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/autoflatten/subjects/sub-$s/surf/{lh,rh}.autoflatten.flat.patch.3d \\
        /data/ds-tmsrisk/derivatives/freesurfer/sub-$s/surf/; done

    ~/mambaforge/envs/pycortex2/bin/python -m tms_risk.visualize.import_flatmaps

``import_flat`` appends ``.flat`` to `patch` itself, so it is passed as
``'autoflatten'`` (the pycortex skill). A subject whose flat surfaces already
exist is skipped; ``--overwrite`` re-imports. The subject's ``.ctm`` cache is
cleared so the next viewer build picks the flat surface up.
"""
import argparse
from pathlib import Path

import cortex
from cortex import freesurfer

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31, 34, 35, 36,
            37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
FS_DIR = Path('/data/ds-tmsrisk/derivatives/freesurfer')


def main(subjects, overwrite=False):
    store = Path(cortex.database.default_filestore)
    for s in subjects:
        sid, cx = f'sub-{s:02d}', f'tms.sub-{s:02d}'
        surf = FS_DIR / sid / 'surf'
        patches = [surf / f'{h}.autoflatten.flat.patch.3d' for h in ('lh', 'rh')]
        if not all(p.exists() for p in patches):
            print(f'{cx}: no flat patches yet -- skipped')
            continue
        if (store / cx / 'surfaces' / 'flat_rh.gii').exists() and not overwrite:
            print(f'{cx}: flat surfaces already imported')
            continue
        freesurfer.import_flat(sid, patch='autoflatten', hemis=['lh', 'rh'], cx_subject=cx,
                               freesurfer_subject_dir=str(FS_DIR), auto_overwrite=True)
        for f in (store / cx / 'cache').glob('*'):
            if f.is_file():
                f.unlink()
        print(f'{cx}: flat surfaces imported')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('subjects', nargs='*', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    main(a.subjects or SUBJECTS, a.overwrite)
