import os.path as op

class Subject(object):

    def __init__(self, subject, bids_folder='/data/ds-tmsrisk'):

        self.subject = '%02d' % int(subject)
        self.bids_folder = bids_folder


    def get_volume_mask(self, roi='NPC12r'):

        if roi.startswith('NPC'):
            return op.join(self.derivatives_dir
            ,'ips_masks',
            f'sub-{self.subject}',
            'anat',
            f'sub-{self.subject}_space-T1w_desc-{roi}_mask.nii.gz'
            )

        else:
            raise NotImplementedError

    @property
    def derivatives_dir(self):
        return op.join(self.bids_folder, 'derivatives')

    @property
    def fmriprep_dir(self):
        return op.join(self.derivatives_dir, 'fmriprep', f'sub-{self.subject}')

    @property
    def t1w(self):
        t1w = op.join(self.fmriprep_dir,
        'anat',
        'sub-{self.subject}_desc-preproc_T1w.nii.gz')

        if not op.exists(t1w):
            t1w = op.join(self.fmriprep_dir,
            'ses-1', 'anat',
            f'sub-{self.subject}_ses-1_desc-preproc_T1w.nii.gz')
        
        if not op.exists(t1w):
            raise Exception(f'T1w can not be found for subject {self.subject}')

        return t1w

    def get_nprf_pars(self, session=1, model='encoding_model.smoothed', parameter='r2',
    volume=True):

        if not volume:
            raise NotImplementedError

        im = op.join(self.derivatives_dir, model, f'sub-{self.subject}',
        f'ses-{session}', 'func', 
        f'sub-{self.subject}_ses-{session}_desc-{parameter}.optim_space-T1w_pars.nii.gz')

        return im
