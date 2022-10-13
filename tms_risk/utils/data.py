import os
import os.path as op
from re import I
import pandas as pd
from itertools import product
import numpy as np
import pkg_resources
import yaml
from sklearn.decomposition import PCA
from nilearn import image
from nilearn.maskers import NiftiMasker
from collections.abc import Iterable

def get_subjects(bids_folder='/data/ds-tmsrisk', correct_behavior=True, correct_npc=False):
    subjects = list(range(1, 100))

    # Does not exist
    subjects.pop(subjects.index(14))

    if correct_behavior:
       # Pure random behavior
       subjects.pop(subjects.index(17))

    subjects = [Subject(subject, bids_folder) for subject in subjects]

    return subjects

def get_all_behavior(bids_folder='/data/ds-tmsrisk', correct_behavior=True, correct_npc=False, drop_no_responses=True):

    subjects = get_subjects(bids_folder, correct_behavior, correct_npc)
    behavior = [s.get_behavior(drop_no_responses=drop_no_responses) for s in subjects]
    return pd.concat(behavior)

def get_tms_conditions():
    with pkg_resources.resource_stream('tms_risk', '/data/tms_keys.yml') as stream:
        return yaml.safe_load(stream)


class Subject(object):

    def __init__(self, subject, bids_folder='/data/ds-tmsrisk'):

        self.subject = '%02d' % int(subject)
        self.bids_folder = bids_folder

        self.tms_conditions = {1:'baseline', 2:None, 3:None}


        if self.subject in get_tms_conditions():
            tc = get_tms_conditions()[self.subject]
            for session in [2, 3]:
                if session in tc:
                    self.tms_conditions[session] = tc[session]

    def get_stimulation_condition(self, session):
        if session == 1:
            return 'baseline'
        else:
            tms_conditions = get_tms_conditions()
            if self.subject in tms_conditions:
                if session in tms_conditions[self.subject]:
                    return tms_conditions[self.subject][session]
            return None

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

    def get_preprocessed_bold(self, session=1, runs=None, space='T1w'):
        if runs is None:
            runs = range(1, 7)

        images = [op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject}',
         f'ses-{session}', 'func', f'sub-{self.subject}_ses-{session}_task-task_run-{run}_space-{space}_desc-preproc_bold.nii.gz') for run in runs]

        return images

    def get_nprf_pars(self, session=1, model='encoding_model.smoothed', parameter='r2',
    volume=True):

        if not volume:
            raise NotImplementedError

        im = op.join(self.derivatives_dir, model, f'sub-{self.subject}',
        f'ses-{session}', 'func', 
        f'sub-{self.subject}_ses-{session}_desc-{parameter}.optim_space-T1w_pars.nii.gz')

        return im

    def get_behavior(self, sessions=None, drop_no_responses=True):
        if sessions is None:
            sessions = [1, 2, 3]

        if not isinstance(sessions, Iterable):
            sessions = [sessions]

        runs = range(1, 7)
        df = []
        for session, run in product(sessions, runs):

            tms_condition = self.tms_conditions[session]

            fn = op.join(self.bids_folder, f'sub-{self.subject}/ses-{session}/func/sub-{self.subject}_ses-{session}_task-task_run-{run}_events.tsv')

            if op.exists(fn):
                d = pd.read_csv(fn, sep='\t',
                            index_col=['trial_nr', 'trial_type'])
                d['subject'], d['session'], d['run'], d['stimulation_condition'] = int(self.subject), session, run, tms_condition
                df.append(d)

        if len(df) > 0:
            df = pd.concat(df)
            df = df.reset_index().set_index(['subject', 'session', 'stimulation_condition', 'run', 'trial_nr', 'trial_type']) 
            df = df.unstack('trial_type')
            return self._cleanup_behavior(df, drop_no_responses=drop_no_responses)
        else:
            return pd.DataFrame([])

    @staticmethod
    def _cleanup_behavior(df_, drop_no_responses=True):
        df = df_[[]].copy()
        df['rt'] = df_.loc[:, ('onset', 'choice')] - df_.loc[:, ('onset', 'stimulus 2')]
        df['n1'], df['n2'] = df_['n1']['stimulus 1'], df_['n2']['stimulus 1']
        df['prob1'], df['prob2'] = df_['prob1']['stimulus 1'], df_['prob2']['stimulus 1']

        df['choice'] = df_[('choice', 'choice')]
        df['risky_first'] = df['prob1'] == 0.55
        df['chose_risky'] = (df['risky_first'] & (df['choice'] == 1.0)) | (~df['risky_first'] & (df['choice'] == 2.0))
        df.loc[df.choice.isnull(), 'chose_risky'] = np.nan


        df['n_risky'] = df['n1'].where(df['risky_first'], df['n2'])
        df['n_safe'] = df['n2'].where(df['risky_first'], df['n1'])
        df['frac'] = df['n_risky'] / df['n_safe']
        df['log(risky/safe)'] = np.log(df['frac'])

        df['log(n1)'] = np.log(df['n1'])

        if drop_no_responses:
            df = df[~df.chose_risky.isnull()]
            df['chose_risky'] = df['chose_risky'].astype(bool)

        def get_risk_bin(d):
            labels = [f'{int(e)}%' for e in np.linspace(20, 80, 6)]
            try: 
                # return pd.qcut(d, 6, range(1, 7))
                return pd.qcut(d, 6, labels=labels)
            except Exception as e:
                n = len(d)
                ix = np.linspace(0, 6, n, False)

                d[d.sort_values().index] = [labels[e] for e in np.floor(ix).astype(int)]
                
                return d
        df['bin(risky/safe)'] = df.groupby(['subject'], group_keys=False)['frac'].apply(get_risk_bin)

        return df.droplevel(-1, 1)

    def get_fmriprep_confounds(self, session, include=None):

        if include is None:
            include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                        'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                        'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02', 
                                        'non_steady_state_outlier00', 'non_steady_state_outlier01', 'non_steady_state_outlier02']


        runs = range(1, 7)

        fmriprep_confounds = [
            op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject}/ses-{session}/func/sub-{self.subject}_ses-{session}_task-task_run-{run}_desc-confounds_timeseries.tsv') for run in runs]
        fmriprep_confounds = [pd.read_table(
            cf)[include] for cf in fmriprep_confounds]

        return fmriprep_confounds

    def get_retroicor_confounds(self, session, n_cardiac=3, n_respiratory=4, n_interaction=2):

        runs = range(1, 7)

        columns = []
        for n, modality in zip([3, 4, 2], ['cardiac', 'respiratory', 'interaction']):
            for order in range(1, n+1):
                columns += [(modality, order, 'sin'), (modality, order, 'cos')]
        columns = pd.MultiIndex.from_tuples(
            columns, names=['modality', 'order', 'type'])                        

        retroicor_confounds = [
            op.join(self.bids_folder, f'derivatives/physiotoolbox/sub-{self.subject}/ses-{session}/func/sub-{self.subject}_ses-{session}_task-task_run-{run}_desc-retroicor_timeseries.tsv') for run in runs]
        retroicor_confounds = [pd.read_table(
            cf, header=None, usecols=np.arange(18), names=columns) if op.exists(cf) else pd.DataFrame(np.zeros((135, 0))) for cf in retroicor_confounds]

        retroicor_confounds = pd.concat(retroicor_confounds, 0, keys=runs,
                            names=['run']).sort_index(axis=1)

        retroicor_confounds = pd.concat((retroicor_confounds.loc[:, ('cardiac', slice(n_cardiac))],
                            retroicor_confounds.loc[:, ('respiratory',
                                                slice(n_respiratory))],
                            retroicor_confounds .loc[:, ('interaction', slice(n_interaction))]), axis=1)

        retroicor_confounds = [cf.droplevel('run') for _, cf in retroicor_confounds.groupby(['run'])]


        for cf in retroicor_confounds:
            cf.columns = [f'retroicor_{i}' for i in range(cf.shape[1])]

        return retroicor_confounds 

    def get_confounds(self, session, include_fmriprep=None, include_retroicor=None, pca=False, pca_n_components=.95):
        
        fmriprep_confounds = self.get_fmriprep_confounds(session, include=include_fmriprep)
        retroicor_confounds = self.get_retroicor_confounds(session)
        print(retroicor_confounds)
        confounds = [pd.concat((rcf, fcf), axis=1) for rcf, fcf in zip(retroicor_confounds, fmriprep_confounds)]
        confounds = [c.fillna(method='bfill') for c in confounds]

        if pca:
            def map_cf(cf, n_components=pca_n_components):
                pca = PCA(n_components=n_components)
                cf -= cf.mean(0)
                cf /= cf.std(0)
                cf = pd.DataFrame(pca.fit_transform(cf))
                cf.columns = [f'pca_{i}' for i in range(1, cf.shape[1]+1)]
                return cf
            confounds = [map_cf(cf) for cf in confounds]

        else:
            # remove column names
            confounds = [cf.T.reset_index(drop=True).T for cf in confounds]

        return confounds

    def get_single_trial_volume(self, session, roi=None, 
            smoothed=False,
            pca_confounds=False):

        key= 'glm_stim1'

        if smoothed:
            key += '.smoothed'

        if pca_confounds:
            key += '.pca_confounds'

        fn = op.join(self.bids_folder, 'derivatives', key, f'sub-{self.subject}', f'ses-{session}', 'func', 
                f'sub-{self.subject}_ses-{session}_task-task_space-T1w_desc-stims1_pe.nii.gz')

        im = image.load_img(fn)
        
        mask = self.get_volume_mask(roi=roi, session=session, epi_space=True)
        masker = NiftiMasker(mask_img=mask)

        data = pd.DataFrame(masker.fit_transform(im))

        return data

    def get_volume_mask(self, roi=None, session=None, epi_space=False):

        if roi is None:
            if epi_space:
                base_mask = op.join(self.bids_folder, 'derivatives', f'fmriprep/sub-{self.subject}/ses-{session}/func/sub-{self.subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')
                return image.load_img(base_mask)
            else:
                raise NotImplementedError
        elif roi.startswith('NPC'):
            mask = op.join(self.derivatives_dir
            ,'ips_masks',
            f'sub-{self.subject}',
            'anat',
            f'sub-{self.subject}_space-T1w_desc-{roi}_mask.nii.gz'
            )
        else:
            raise NotImplementedError

        if epi_space:
            base_mask = op.join(self.bids_folder, 'derivatives', f'fmriprep/sub-{self.subject}/ses-{session}/func/sub-{self.subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')

            mask = image.resample_to_img(mask, base_mask, interpolation='nearest')

        return mask
    
    def get_prf_parameters_volume(self, session, 
            run=None,
            smoothed=False,
            pca_confounds=False,
            denoise=False,
            cross_validated=True,
            hemi=None,
            roi=None,
            space='fsnative'):

        dir = 'encoding_model'
        if cross_validated:
            if run is None:
                raise Exception('Give run')

            dir += '.cv'

        if denoise:
            dir += '.denoise'
            
        if smoothed:
            dir += '.smoothed'

        if pca_confounds:
            dir += '.pca_confounds'

        parameters = []

        keys = ['mu', 'sd', 'amplitude', 'baseline']

        mask = self.get_volume_mask(session=session, roi=roi, epi_space=True)
        masker = NiftiMasker(mask)

        for parameter_key in keys:
            if cross_validated:
                fn = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject}', f'ses-{session}', 
                        'func', f'sub-{self.subject}_ses-{session}_run-{run}_desc-{parameter_key}.optim_space-T1w_pars.nii.gz')
            else:
                fn = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject}', f'ses-{session}', 
                        'func', f'sub-{self.subject}_ses-{session}_desc-{parameter_key}.optim_space-T1w_pars.nii.gz')
            
            pars = pd.Series(masker.fit_transform(fn).ravel())
            parameters.append(pars)

        return pd.concat(parameters, axis=1, keys=keys, names=['parameter'])

    def get_fmri_events(self, session, runs=None):

        if runs is None:
            runs = range(1,7)

        behavior = []
        for run in runs:
            behavior.append(pd.read_table(op.join(
                self.bids_folder, f'sub-{self.subject}/ses-{session}/func/sub-{self.subject}_ses-{session}_task-task_run-{run}_events.tsv')))

        behavior = pd.concat(behavior, keys=runs, names=['run'])
        behavior = behavior.reset_index().set_index(
            ['run', 'trial_type'])


        stimulus1 = behavior.xs('stimulus 1', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_nr', 'trial_type']]
        stimulus1['duration'] = 0.6
        stimulus1['trial_type'] = stimulus1.trial_nr.map(lambda trial: f'trial_{trial:03d}_n1')

        
        stimulus2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_nr', 'trial_type']]
        stimulus2['duration'] = 0.6
        stimulus2['trial_type'] = stimulus1.trial_nr.map(lambda trial: f'trial_{trial:03d}_n2')

        events = pd.concat((stimulus1, stimulus2)).sort_index()

        return events

    def get_target_dir(subject, session, sourcedata, base, modality='func'):
        target_dir = op.join(sourcedata, 'derivatives', base, f'sub-{subject}', f'ses-{session}',
                            modality)

        if not op.exists(target_dir):
            os.makedirs(target_dir)

        return target_dir