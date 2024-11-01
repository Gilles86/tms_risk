import argparse
import os.path as op
import nipype.pipeline.engine as pe
from fmriprep.workflows.bold import init_bold_surf_wf
import nipype.interfaces.utility as niu
from nipype.interfaces.io import ExportFile
from itertools import product
from nipype.utils.misc import flatten
from tms_risk.encoding_model.fit_nprf import get_key_target_dir


def main(subject, session, bids_folder='/data', smoothed=False):
    spaces = ['fsnative', 'fsaverage']


    parameters = ['r2', 'mu', 'sd', 'cvr2', 'amplitude', 'baseline']

    key = get_key_target_dir(f'{int(subject):02d}', session, bids_folder, smoothed, denoise=True, pca_confounds=False, retroicor=False, natural_space=True, only_target_key=True)    
    key_cv = key.replace('encoding_model', 'encoding_model.cv')


    wf = pe.Workflow(name=f'resample_{subject}_{session}_{key.replace(".", "_")}', base_dir='/tmp')


    for par in parameters:
        surf_wf = init_bold_surf_wf(mem_gb=4, surface_spaces=spaces, name=f'sample_{par}', medial_surface_nan=True)
        
        inputnode = pe.Node(niu.IdentityInterface(fields=['source_file', 'subjects_dir', 'subject_id', 't1w2fsnative_xfm']),
                       name=f'inputnode_{par}')

        if par == 'cvr2':
            inputnode.inputs.source_file = op.join(bids_folder, f'derivatives/{key_cv}/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_desc-{par}.optim_space-T1w_pars.nii.gz')
        else:
            inputnode.inputs.source_file = op.join(bids_folder, f'derivatives/{key}/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_desc-{par}.optim_space-T1w_pars.nii.gz')

        inputnode.inputs.subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')
        inputnode.inputs.subject_id = f'sub-{subject}'
        inputnode.inputs.t1w2fsnative_xfm = op.join(bids_folder, f'derivatives/fmriprep/sub-{subject}/ses-1/anat/sub-{subject}_ses-1_from-T1w_to-fsnative_mode-image_xfm.txt')

        wf.connect([(inputnode, surf_wf, [('source_file', 'inputnode.source_file'),
                                       ('subjects_dir', 'inputnode.subjects_dir'), 
                                                                ('subject_id', 'inputnode.subject_id'),
                                                                ('t1w2fsnative_xfm', 'inputnode.t1w2fsnative_xfm')])])

        export_file = pe.MapNode(ExportFile(clobber=True), iterfield=['in_file', 'out_file'], 
                                 name=f'exporter_{par}')

        export_file.inputs.out_file = [op.join(bids_folder, f'derivatives/{key}/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_desc-{par}.volume.optim_space-{space}_hemi-{hemi}.func.gii') for space, hemi in product(spaces, ['L', 'R'])]

        wf.connect(surf_wf, ('outputnode.surfaces', flatten), export_file, 'in_file')

    wf.run(plugin='MultiProc', plugin_args={'n_procs' : 4})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument(
        '--bids_folder', default='/data')
    parser.add_argument(
        '--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder,smoothed=args.smoothed )
