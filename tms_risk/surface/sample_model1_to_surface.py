"""Sample the canonical m1 nPRF maps onto each participant's cortical surface.

Feeds the paper's interactive viewers (``tms_risk/visualize/make_static_viewers.py``).
For every TMS participant it writes two NPZ files of per-vertex maps, one on the
participant's own FreeSurfer surface (fsnative) and one resampled to fsaverage:

    cvr2        leave-one-run-out cvR² of m1 (session-agnostic)
    cvr2_null   cvR² of the null predictor -- the training-fold mean -- under the
                same folds and formula (``reproduce_figure2_stats.null_cvr2``). A
                vertex carries nPRF signal where cvr2 > cvr2_null, which is the
                Figure-2b signal definition.
    r2          full-fit R²
    mu, sd      log preferred numerosity and log tuning width (shared across
                sessions in m1, so one map each)
    amp_ips, amp_vertex
                the per-session response amplitude, relabelled by stimulation arm
    roi2cm      fraction of the sample falling in the paper's individualised ROI
                (``NPCr2cm-cluster``: within 2 cm of the session-1 targeting cluster,
                inside NPC1+NPC2)
    stim_dist   geodesic distance (mm) from the cTBS target on the right pial
                surface; NaN on the left hemisphere

Surfaces are FreeSurfer's own white and pial meshes, moved from tkr-RAS to scanner
RAS with the ``orig.mgz`` headers. fMRIPrep's T1w and FreeSurfer's ``rawavg.mgz``
share one affine in this dataset, so scanner RAS *is* the T1w space the parameter
maps live in. The script checks this per participant: the stimulation coordinate
from neuronavigation must land on the pial surface (it prints the distance).

Volumes are sampled along the normal between white and pial (nilearn ``depth``
sampling) with linear interpolation, ignoring voxels outside the fitted brain
mask. fsnative -> fsaverage is nearest neighbour on the ``sphere.reg``
registration, which is what ``mri_surf2surf --mapmethod nnf`` does.

The local ``encoding_model2.model-1.smoothed.cv`` still holds the November-2025
10-iteration cvR² maps; the converged 2026-08 maps were copied from the cluster
to ``encoding_model2.model-1.smoothed.cv.20260806``, which is the default here.

    python -m tms_risk.surface.sample_model1_to_surface            # all 35
    python -m tms_risk.surface.sample_model1_to_surface 02 --group # one + group
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import surface
from nilearn.surface import InMemoryMesh
from scipy import stats
from scipy.spatial import cKDTree

from tms_risk.utils.data import Subject, get_tms_conditions

SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31, 34, 35, 36,
            37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
HEMIS = ('lh', 'rh')


def fs_meshes(fs_dir, sid):
    """White and pial meshes per hemisphere, in scanner RAS (= fMRIPrep T1w)."""
    orig = nib.load(str(fs_dir / f'sub-{sid}' / 'mri' / 'orig.mgz'))
    tkr2scanner = orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    meshes = {}
    for hemi in HEMIS:
        m = {}
        for kind in ('white', 'pial'):
            coords, faces = nib.freesurfer.read_geometry(
                str(fs_dir / f'sub-{sid}' / 'surf' / f'{hemi}.{kind}'))
            m[kind] = InMemoryMesh(nib.affines.apply_affine(tkr2scanner, coords), faces)
        meshes[hemi] = m
    return meshes


def sample(img, meshes, mask_img):
    """Depth-sample a volume between white and pial, L then R (pycortex order)."""
    return np.concatenate([
        surface.vol_to_surf(img, meshes[h]['pial'], inner_mesh=meshes[h]['white'],
                            mask_img=mask_img, interpolation='linear')
        for h in HEMIS]).astype(np.float32)


def null_cvr2_volume(bids_folder, sid, mask):
    """cvR² of the training-fold mean, every voxel in `mask`, as a 3D array.

    Identical folds and formula to ``reproduce_figure2_stats.null_cvr2`` (run r of
    BOTH TMS sessions held out; the denominator uses the held-out run's own mean)
    -- only the voxel set is the whole fitted mask rather than one ROI.
    """
    deriv = Path(bids_folder) / 'derivatives'
    runs = (Subject(sid, bids_folder=bids_folder).get_paradigm()
            .reset_index('session').index.get_level_values('run'))
    data = []
    for ses in (2, 3):
        im = nib.load(str(deriv / 'glm_stim1.denoise.smoothed' / f'sub-{sid}' / f'ses-{ses}'
                          / 'func' / f'sub-{sid}_ses-{ses}_task-task_space-T1w_desc-stims1_pe.nii.gz'))
        assert im.shape[:3] == mask.shape, (im.shape, mask.shape)
        data.append(np.asarray(im.dataobj, dtype=np.float32)[mask].T)
    data = np.vstack(data)
    assert len(data) == len(runs), (len(data), len(runs))
    folds = []
    for r in sorted(np.unique(runs)):
        te, tr = runs == r, runs != r
        ssr = ((data[te] - data[tr].mean(0)) ** 2).sum(0)
        sst = ((data[te] - data[te].mean(0)) ** 2).sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            f = 1 - ssr / sst
        f[sst == 0] = np.nan
        folds.append(f)
    out = np.zeros(mask.shape, dtype=np.float32)
    out[mask] = np.nanmean(np.stack(folds), axis=0)
    return out


def surface_distance(mesh, point):
    """Shortest-path distance (mm) along mesh edges from the vertex nearest `point`.

    Only a fallback for a participant whose stored geodesic map was made on another
    reconstruction; within the 5 mm used for display it agrees with the heat-method
    geodesic of ``sample_geodesic_stimulation_mask.py`` to a fraction of a mm.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    v, f = mesh.coordinates, mesh.faces
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]), 1), axis=0)
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    g = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(v), len(v))).tocsr()
    src = int(np.argmin(np.linalg.norm(v - point, axis=1)))
    return dijkstra(g, directed=False, indices=src).astype(np.float32)


def fsaverage_index(fs_dir, sid):
    """For every fsaverage vertex (L then R), the nearest fsnative vertex index."""
    idx, offset = [], 0
    for hemi in HEMIS:
        subj = nib.freesurfer.read_geometry(str(fs_dir / f'sub-{sid}' / 'surf' / f'{hemi}.sphere.reg'))[0]
        avg = nib.freesurfer.read_geometry(str(fs_dir / 'fsaverage' / 'surf' / f'{hemi}.sphere.reg'))[0]
        idx.append(cKDTree(subj).query(avg, workers=-1)[1] + offset)
        offset += len(subj)
    return np.concatenate(idx)


def process_subject(subject, bids_folder, cvr2_dir, out_root):
    sid = f'{int(subject):02d}'
    deriv = Path(bids_folder) / 'derivatives'
    fs_dir = deriv / 'freesurfer'
    m1 = deriv / 'encoding_model2.model-1.smoothed' / f'sub-{sid}'
    arms = get_tms_conditions()[sid]                    # {2: 'vertex', 3: 'ips'}
    ses_of = {arm: ses for ses, arm in arms.items()}

    def vol(par, ses=None):
        if par == 'cvr2':
            return Path(cvr2_dir) / f'sub-{sid}' / f'sub-{sid}_desc-cvr2.optim_space-T1w_pars.nii.gz'
        if par == 'r2':
            return m1 / f'sub-{sid}_desc-r2.optim_space-T1w_pars.nii.gz'
        return m1 / f'ses-{ses}' / f'sub-{sid}_ses-{ses}_desc-{par}.optim_space-T1w_pars.nii.gz'

    mu_img = nib.load(str(vol('mu', 2)))
    mask = mu_img.get_fdata() != 0
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), mu_img.affine)
    meshes = fs_meshes(fs_dir, sid)

    maps = {}
    maps['cvr2'] = sample(str(vol('cvr2')), meshes, mask_img)
    null = nib.Nifti1Image(null_cvr2_volume(bids_folder, sid, mask), mu_img.affine)
    maps['cvr2_null'] = sample(null, meshes, mask_img)
    maps['r2'] = sample(str(vol('r2')), meshes, mask_img)
    maps['mu'] = sample(str(vol('mu', 2)), meshes, mask_img)
    maps['sd'] = sample(str(vol('sd', 2)), meshes, mask_img)
    for arm in ('ips', 'vertex'):
        maps[f'amp_{arm}'] = sample(str(vol('amplitude', ses_of[arm])), meshes, mask_img)

    roi_fn = deriv / 'ips_masks' / f'sub-{sid}' / 'func' / 'ses-1' / f'sub-{sid}_space-T1w_desc-NPCr2cm-cluster_mask.nii.gz'
    maps['roi2cm'] = (sample(nib.load(str(roi_fn)), meshes, None) if roi_fn.exists()
                      else np.full_like(maps['cvr2'], np.nan))

    n_lh = len(meshes['lh']['pial'].coordinates)
    coords = np.loadtxt(deriv / 'stim_coordinates' / f'sub-{sid}' / f'sub-{sid}_coords_warped.txt')
    dist_fn = deriv / 'ips_masks' / f'sub-{sid}' / 'anat' / f'sub-{sid}_space-fsnative_desc-NPCr_geodesic_distance_hemi-R.anat.gii'
    rh_dist = surface.load_surf_data(str(dist_fn)).astype(np.float32)
    if len(rh_dist) != len(meshes['rh']['pial'].coordinates):
        # sub-45's stored map was computed on a different reconstruction (144,267
        # vs 143,898 vertices); recompute on the one the viewer shows
        print(f'sub-{sid}: stored geodesic map does not match rh.pial -- recomputing')
        rh_dist = surface_distance(meshes['rh']['pial'], coords[0])
    maps['stim_dist'] = np.concatenate([np.full(n_lh, np.nan, np.float32), rh_dist])

    # The neuronavigation coordinate must sit on the pial surface we sampled on --
    # this is the check that tkr -> scanner put the meshes in the maps' space.
    target = meshes['rh']['pial'].coordinates[np.argmin(rh_dist)]
    off = float(np.linalg.norm(target - coords[0]))
    print(f'sub-{sid}: target vertex {off:.2f} mm from the navigation coordinate; '
          f'{np.mean(maps["cvr2"] > maps["cvr2_null"]):.1%} of vertices beat the null',
          flush=True)

    out = Path(out_root) / f'sub-{sid}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'sub-{sid}_space-fsnative_desc-model1_maps.npz',
                        n_lh=n_lh, target_offset_mm=off, **maps)
    ix = fsaverage_index(fs_dir, sid)
    np.savez_compressed(out / f'sub-{sid}_space-fsaverage_desc-model1_maps.npz',
                        **{k: v[ix] for k, v in maps.items()})


def group_maps(out_root, subjects):
    """Stack every participant's fsaverage maps and reduce to group maps."""
    stacks = {}
    for s in subjects:
        fn = Path(out_root) / f'sub-{s:02d}' / f'sub-{s:02d}_space-fsaverage_desc-model1_maps.npz'
        with np.load(fn) as d:
            for k in d.files:
                stacks.setdefault(k, []).append(d[k])
    S = {k: np.stack(v) for k, v in stacks.items()}          # (n_subjects, n_vertices)
    signal = S['cvr2'] > S['cvr2_null']
    out = dict(subjects=np.array(subjects),
               prevalence=signal.mean(0).astype(np.float32),
               cvr2_minus_null=np.nanmean(S['cvr2'] - S['cvr2_null'], 0).astype(np.float32))
    # preferred numerosity: mean of log mu over the participants who have signal there
    with np.errstate(invalid='ignore'):
        mu = np.where(signal, S['mu'], np.nan)
        out['mu_mean'] = np.nanmean(mu, 0).astype(np.float32)
        out['n_signal'] = signal.sum(0).astype(np.int16)
        delta = S['amp_ips'] - S['amp_vertex']
        out['amp_delta_mean'] = np.nanmean(delta, 0).astype(np.float32)
        t, p = stats.ttest_1samp(delta, 0.0, axis=0, nan_policy='omit')
        out['amp_delta_t'] = np.asarray(t, dtype=np.float32)
        out['amp_mean'] = np.nanmean((S['amp_ips'] + S['amp_vertex']) / 2, 0).astype(np.float32)
        # the same contrast restricted to each participant's signal vertices
        dsig = np.where(signal, delta, np.nan)
        out['amp_delta_mean_signal'] = np.nanmean(dsig, 0).astype(np.float32)
        out['target_count'] = (S['stim_dist'] <= 5.0).sum(0).astype(np.int16)
    out['stim_dist'] = S['stim_dist'].astype(np.float32)
    out['signal'] = signal
    fn = Path(out_root) / 'group_space-fsaverage_desc-model1_maps.npz'
    np.savez_compressed(fn, **out)
    print(f'wrote {fn}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('subject', nargs='*', default=None)
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--cvr2_dir', default=None,
                   help='default: <bids>/derivatives/encoding_model2.model-1.smoothed.cv.20260806')
    p.add_argument('--out_root', default=None,
                   help='default: <bids>/derivatives/surface_viewer')
    p.add_argument('--group', action='store_true', help='also (re)build the group maps')
    a = p.parse_args()
    deriv = Path(a.bids_folder) / 'derivatives'
    cvr2_dir = a.cvr2_dir or deriv / 'encoding_model2.model-1.smoothed.cv.20260806'
    out_root = a.out_root or deriv / 'surface_viewer'
    subs = [int(s) for s in a.subject] if a.subject else SUBJECTS
    for s in subs:
        process_subject(s, a.bids_folder, cvr2_dir, out_root)
    if a.group or not a.subject:
        group_maps(out_root, SUBJECTS)
