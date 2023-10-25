import os
import os.path as op
import argparse
import pandas as pd
import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nipype.interfaces import fsl


def main(subject, bids_folder):


    def prepare_coords(in_file):
        import pandas as pd
        import os.path as op
        coords = pd.read_csv(in_file, sep='\t', comment='#',
                         index_col=0, names=['x', 'y', 'z', 'm0n0', 'm0n1',	'm0n2', 'm1n0',	'm1n1', 'm1n2',	'm2n0', 'm2n1', 'm2n2'])
        fn = op.abspath('coords.txt')
        coords.iloc[:, :3].to_csv(fn, sep=' ', header=False, index=False)

        return fn


    coords = op.join(bids_folder, 'derivatives', 'stim_coordinates', f'sub-{subject}_coords.txt')

    workflow = pe.Workflow(name=f'sub-{subject}_coords_to_T1w', base_dir='/tmp')

    inputnode = pe.Node(niu.IdentityInterface(fields=['coords', 'T1w', 'T1w_marius']), name='inputnode')

    inputnode.inputs.coords = coords
    inputnode.inputs.T1w_marius = op.join(bids_folder, f'derivatives/tms_navigation/sub-{subject}/sub-{subject}_T1w.nii.gz')
    inputnode.inputs.T1w = op.join(bids_folder, f'derivatives/fmriprep/sub-{subject}/ses-1/anat/sub-{subject}_ses-1_desc-preproc_T1w.nii.gz')

    flirt = pe.Node(fsl.FLIRT(dof=6), name='flirt')
    workflow.connect(inputnode, 'T1w_marius', flirt, 'in_file')
    workflow.connect(inputnode, 'T1w', flirt, 'reference')

    prepare_coords_node  = pe.Node(niu.Function(function=prepare_coords, input_names=['in_file'], output_names=['out_file']), name='prepare_coords_node')
    workflow.connect(inputnode, 'coords', prepare_coords_node, 'in_file')

    applier = pe.Node(fsl.WarpPoints(coord_mm=True), name='applier')
    workflow.connect(prepare_coords_node, 'out_file', applier, 'in_coords')
    workflow.connect(flirt, 'out_matrix_file', applier, 'xfm_file')
    workflow.connect(inputnode, 'T1w', applier, 'dest_file')
    workflow.connect(inputnode, 'T1w_marius', applier, 'src_file')

    ds = pe.Node(nio.DataSink(infields=[f'sub-{subject}']), name='datasink')
    ds.inputs.base_directory = op.join(bids_folder, 'derivatives', 'stim_coordinates')
    ds.inputs.substitutions = [('coords_warped.txt', f'sub-{subject}_coords_warped.txt')]

    workflow.connect(applier, 'out_file', ds, f'sub-{subject}')

    workflow.run(plugin='MultiProc', plugin_args={'n_procs': 4})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.subject, args.bids_folder)
