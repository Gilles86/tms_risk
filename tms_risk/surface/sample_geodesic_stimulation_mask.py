import argparse
import numpy as np
from tms_risk.utils.data import Subject
from nilearn import surface
import os.path as op
from cortex.polyutils import Surface
import nibabel as nb
from nipype.interfaces.freesurfer import SurfaceTransform
from tms_risk.utils.data import get_tms_subjects

def transform_fsaverage(in_file, fs_hemi, source_subject, bids_folder):

        subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')

        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = in_file.replace('fsnative', 'fsaverage')
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.target_subject = 'fsaverage'
        sxfm.inputs.hemi = fs_hemi

        r = sxfm.run()
        return r

# Standard stuff...
def main(subject, bids_folder):
    print(f'Processing subject {subject} in folder {bids_folder}')

    target_dir = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat')

    coords = np.loadtxt(op.join(bids_folder, 'derivatives', 'stim_coordinates', f'sub-{subject}', f'sub-{subject}_coords_warped.txt'))


    coords_surface = coords[0]

    sub = Subject(subject, bids_folder=bids_folder)
    surfinfo = sub.get_surf_info()
    pts, polys = surface.load_surf_data(surfinfo['R']['outer'])
    # Find the index of the closest vertex to coords_surface
    dist = np.linalg.norm(pts - coords_surface, axis=1)
    closest_vertex = np.argmin(dist)

    surf = Surface(pts, polys)

    # vertex_mask = surf.get_geodesic_patch(closest_vertex, 5)['vertex_mask']
    vertex_distance = surf.geodesic_distance(closest_vertex)
    im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(vertex_distance.astype(np.float32))])
    target_fn =  op.join(target_dir, f'sub-{subject}_space-fsnative_desc-NPCr_geodesic_distance_hemi-R.anat.gii')
    nb.save(im, target_fn)
    transform_fsaverage(target_fn, 'rh', f'sub-{subject}', bids_folder)

    fsaverage_fn = target_fn.replace('fsnative', 'fsaverage')
    distances_fsaverage = surface.load_surf_data(fsaverage_fn)
    pts, polys = surface.load_surf_mesh(op.join(bids_folder, 'derivatives', 'freesurfer', 'fsaverage', 'surf', 'rh.inflated'))
    surf = Surface(pts, polys)
    peak_vertex = np.argmin(distances_fsaverage)
    vertex_mask2 = surf.get_geodesic_patch(peak_vertex, 5)['vertex_mask']
    im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(vertex_mask2.astype(np.float32))])
    target_fn =  op.join(target_dir, f'sub-{subject}_space-fsaverage_desc-NPCr5mm_geodesic_hemi-R.anat.gii')
    nb.save(im, target_fn)


def transform_group(bids_folder):
    subjects = [f'{subject:02d}' for subject in get_tms_subjects(bids_folder)]

    for subject in subjects:
        try:
            main(subject, bids_folder)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('subject', default=None)
    argument_parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    argument_parser.add_argument('--fit_group', action='store_true')
    args = argument_parser.parse_args()

    if args.fit_group:
        transform_group(args.bids_folder)
    else:
        main(args.subject, args.bids_folder)