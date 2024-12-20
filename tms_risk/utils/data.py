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
import warnings
from nilearn import surface
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


def get_tms_subjects(bids_folder='/data/ds-tmsrisk', exclude_outliers=True):
    subjects = [int(e) for e in get_tms_conditions().keys()]

    outliers = [22, 49] # see tms_risk/behavior/outliers.ipynb
    if exclude_outliers:
        for outlier in outliers:
            if outlier in subjects:
                subjects.pop(subjects.index(outlier))
    return subjects

def get_all_subject_ids(bids_folder='/data/ds-tmsrisk', exclude_outliers=True):

    with pkg_resources.resource_stream('tms_risk', '/data/all_subjects.yml') as stream:
        subjects = yaml.safe_load(stream)

    outliers = [22, 49] # see tms_risk/behavior/outliers.ipynb
    if exclude_outliers:
        for outlier in outliers:
            if outlier in subjects:
                subjects.pop(subjects.index(outlier))

    return subjects

def get_subjects(bids_folder='/data/ds-tmsrisk', all_tms_conditions=False, exclude_outliers=True):


    if all_tms_conditions:
        subjects = get_tms_subjects(bids_folder, exclude_outliers)
    else:
        subjects = get_all_subject_ids(bids_folder=bids_folder, exclude_outliers=exclude_outliers)

    subjects = [Subject(subject, bids_folder) for subject in subjects]

    return subjects


def get_all_behavior(bids_folder='/data/ds-tmsrisk', drop_no_responses=True, all_tms_conditions=False, exclude_outliers=True):

    subjects = get_subjects(bids_folder, all_tms_conditions=all_tms_conditions, exclude_outliers=exclude_outliers)
    behavior = [s.get_behavior(drop_no_responses=drop_no_responses) for s in subjects]
    return pd.concat(behavior)

def get_tms_conditions():
    with pkg_resources.resource_stream('tms_risk', '/data/tms_keys.yml') as stream:
        return yaml.safe_load(stream)


def get_participant_info(bids_folder='/data/ds-tmsrisk'):
    return pd.read_csv(op.join(bids_folder, 'participants.tsv'), sep='\t', index_col='participant_id')

def get_all_apriori_roi_labels():
    return ['NPC1l', 'NPC1r', 'NPC2l', 'NPC2r', 'NPC3l', 'NPC3r', 'NTOl', 'NTOr', 'NF1l', 'NF1r', 'NF2l', 'NF2r']



def get_pdf(subject, session, pca_confounds=False, denoise=False, smoothed=False, bids_folder='/data/ds-tmsrisk', mask='NPC12r', n_voxels=100, natural_space=False,
            new_parameterisation=False):

    if n_voxels == 1:
        key = 'decoded_pdfs.volume.cv_voxel_selection'
    else:
        key = 'decoded_pdfs.volume'

    subject = f'{subject:02d}'

    if denoise:
        key += '.denoise'

    if smoothed:
        key += '.smoothed'

    if pca_confounds and not denoise:
        key += '.pca_confounds'

    if natural_space:
        key += '.natural_space'

    if new_parameterisation:
        key += '.new_parameterisation'

    if n_voxels == 1:
        pdf = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func', f'sub-{subject}_ses-{session}_mask-{mask}_space-T1w_pars.tsv')
    else:
        pdf = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func', f'sub-{subject}_ses-{session}_mask-{mask}_nvoxels-{n_voxels}_space-T1w_pars.tsv')

    if op.exists(pdf):
        pdf = pd.read_csv(pdf, sep='\t', index_col=[0,1])
        pdf.columns = pdf.columns.astype(float)

        if natural_space:
            pdf = pdf.loc[:, 5:112]
        else:
            pdf = pdf.loc[:, np.log(5):np.log(112)]
    else:
        print(pdf)
        pdf = pd.DataFrame(np.zeros((0, 0)))
    
    pdf /= np.trapz(pdf, pdf.columns, axis=1)[:, np.newaxis]

    return pdf

def get_decoding_info(subject, session, pca_confounds=False, denoise=False, smoothed=False, bids_folder='/data/ds-tmsrisk', mask='NPC12r', n_voxels=100, natural_space=False,
                      new_parameterisation=False):

    pdf = get_pdf(subject, session, pca_confounds=pca_confounds, denoise=denoise, smoothed=smoothed, bids_folder=bids_folder, mask=mask, n_voxels=n_voxels, natural_space=natural_space,
                  new_parameterisation=new_parameterisation)

    E = pd.Series(np.trapz(pdf*pdf.columns.values[np.newaxis,:], pdf.columns, axis=1), index=pdf.index)

    E = pd.concat((E,), keys=[(int(subject), int(session), 'pca_confounds' if pca_confounds else 'no pca', 'GLMstim' if denoise else "glm", 'smoothed' if smoothed else 'not smoothed', mask, n_voxels,
                                'natural' if natural_space else 'log')],
    names=['subject', 'session', 'pca', 'glm', 'smoothed', 'mask', 'n_voxels', 'space']).to_frame('E')

    
    E['sd'] = np.trapz(np.abs(E.values - pdf.columns.astype(float).values[np.newaxis, :]) * pdf, pdf.columns, axis=1)

    return E


class Subject(object):

    def __init__(self, subject, bids_folder='/data/ds-tmsrisk'):

        self.subject = '%02d' % int(subject)
        self.bids_folder = bids_folder

        self.tms_conditions = {1:'baseline', 2:None, 3:None}

        for key, value in get_participant_info(bids_folder).loc[f'sub-{self.subject}'].items():
            setattr(self, key, value)

        if self.subject in get_tms_conditions():
            tc = get_tms_conditions()[self.subject]
            for session in [2, 3]:
                if session in tc:
                    self.tms_conditions[session] = tc[session]


    def get_runs(self, session):
        if (self.subject == '10') & (int(session) == 1):
            logging.info('Subject 10/session 1 has only 5 runs!!')
            return range(1, 6)
        else:
            return range(1, 7)


    def get_stimulation_condition(self, session):
        if session == 1:
            return 'baseline'
        else:
            tms_conditions = get_tms_conditions()
            if self.subject in tms_conditions:
                if session in tms_conditions[self.subject]:
                    return tms_conditions[self.subject][session]
            return None

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
            runs = self.get_runs(session)

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

        df = []
        for session in sessions:
            runs = self.get_runs(session)
            tms_condition = self.tms_conditions[session]
            for run in runs:

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
        df['p1'], df['p2'] = df_['prob1']['stimulus 1'], df_['prob2']['stimulus 1']

        df['choice'] = df_[('choice', 'choice')]
        df['risky_first'] = df['p1'] == 0.55
        df['chose_risky'] = (df['risky_first'] & (df['choice'] == 1.0)) | (~df['risky_first'] & (df['choice'] == 2.0))
        df['chose_risky'] = df['chose_risky'].where(df['choice'].notnull(), np.nan)


        df['n_risky'] = df['n1'].where(df['risky_first'], df['n2'])
        df['n_safe'] = df['n2'].where(df['risky_first'], df['n1'])
        df['frac'] = df['n_risky'] / df['n_safe']
        df['log(risky/safe)'] = np.log(df['frac'])

        df['log(n1)'] = np.log(df['n1'])

        if drop_no_responses:
            df = df[~df.chose_risky.isnull()]
            df['chose_risky'] = df['chose_risky'].astype(bool)

        def get_risk_bin(d, n_bins=6):
            labels = [f'{int(e)}%' for e in np.linspace(20, 80, n_bins)]
            try:
                # Use qcut to assign bins with the specified labels
                return pd.qcut(d, n_bins, labels=labels)
            except Exception as e:
                # Explicitly cast to object to avoid dtype issues
                d = d.astype('object')
                n = len(d)
                ix = np.linspace(0, n_bins, n, False)
                
                # Assign the labels to sorted values, avoiding dtype issues
                d[d.sort_values().index] = [labels[e] for e in np.floor(ix).astype(int)]
                
                return d

        # Apply the function
        df['bin(risky/safe)'] = df.groupby(['subject'], group_keys=False)['frac'].apply(get_risk_bin)
        return df.droplevel(-1, 1)

    def get_fmriprep_confounds(self, session, include=None):

        if include is None:
            include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                        'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                        'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02', 
                                        'non_steady_state_outlier00', 'non_steady_state_outlier01', 'non_steady_state_outlier02']


        runs = self.get_runs(session)

        fmriprep_confounds = [
            op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject}/ses-{session}/func/sub-{self.subject}_ses-{session}_task-task_run-{run}_desc-confounds_timeseries.tsv') for run in runs]
        fmriprep_confounds = [pd.read_table(
            cf)[include] for cf in fmriprep_confounds]

        return fmriprep_confounds

    def get_retroicor_confounds(self, session, n_cardiac=3, n_respiratory=4, n_interaction=2):

        runs = self.get_runs(session)

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

        ix = ~retroicor_confounds.groupby(['run']).apply(lambda d: (d == 0.0).all(0)).any(0)
        retroicor_confounds = retroicor_confounds.loc[:, ix]
        retroicor_confounds = [cf.droplevel('run') for _, cf in retroicor_confounds.groupby(['run'])]


        for cf in retroicor_confounds:
            cf.columns = [f'retroicor_{i}' for i in range(cf.shape[1])]

        return retroicor_confounds 

    def get_confounds(self, session, include_fmriprep=None, include_retroicor=None, pca=False, pca_n_components=.95):
        
        fmriprep_confounds = self.get_fmriprep_confounds(session, include=include_fmriprep)
        retroicor_confounds = self.get_retroicor_confounds(session)
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
            denoise=False,
            smoothed=False,
            pca_confounds=False,
            retroicor=False):

        key= 'glm_stim1'

        if denoise:
            key += '.denoise'

        if (retroicor) and (not denoise):
            raise Exception("When not using GLMSingle RETROICOR is *always* used!")

        if retroicor:
            key += '.retroicor'

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

        base_mask = op.join(self.bids_folder, 'derivatives', f'fmriprep/sub-{self.subject}/ses-{session}/func/sub-{self.subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')
        base_mask = image.load_img(base_mask, dtype='int32') # To prevent weird nilearn warning

        first_run = self.get_preprocessed_bold(session=session, runs=[1])[0]
        base_mask = image.resample_to_img(base_mask, first_run, interpolation='nearest')

        if roi is None:
            if epi_space:
                return base_mask
            else:
                raise NotImplementedError

        elif roi.startswith('NPC') or roi.startswith('NF') or roi.startswith('NTO'):
            
            anat_mask = op.join(self.derivatives_dir
            ,'ips_masks',
            f'sub-{self.subject}',
            'anat',
            f'sub-{self.subject}_space-T1w_desc-{roi}_mask.nii.gz'
            )

            if epi_space:
                epi_mask = op.join(self.derivatives_dir
                                    ,'ips_masks',
                                    f'sub-{self.subject}',
                                    'func',
                                    f'ses-{session}',
                                    f'sub-{self.subject}_space-T1w_desc-{roi}_mask.nii.gz')

                if not op.exists(epi_mask):
                    if not op.exists(op.dirname(epi_mask)):
                        os.makedirs(op.dirname(epi_mask))


                    im = image.resample_to_img(image.load_img(anat_mask, dtype='int32'), image.load_img(base_mask, dtype='int32'), interpolation='nearest')
                    im.to_filename(epi_mask)

                mask = epi_mask

            else: 
                mask = anat_mask

        else:
            raise NotImplementedError

        return image.load_img(mask, dtype='int32')
    
    def get_prf_parameters_volume(self, session, 
            run=None,
            smoothed=False,
            pca_confounds=False,
            denoise=False,
            retroicor=False,
            cross_validated=True,
            natural_space=False,
            keys=None,
            roi=None,
            new_parameterisation=False,
            return_image=False):

        dir = 'encoding_model'

        if cross_validated:
            if run is None:
                raise Exception('Give run')

            dir += '.cv'

        if denoise:
            dir += '.denoise'
            
        if (retroicor) and (not denoise):
            raise Exception("When not using GLMSingle RETROICOR is *always* used!")

        if retroicor:
            dir += '.retroicor'

        if smoothed:
            dir += '.smoothed'

        if pca_confounds:
            dir += '.pca_confounds'

        if natural_space:
            dir += '.natural_space'

        if new_parameterisation:
            dir += '.new_parameterisation'

        parameters = []

        if keys is None:

            if new_parameterisation:
                keys = ['mode', 'fwhm', 'amplitude', 'baseline', 'r2', 'cvr2']
            else:
                keys = ['mu', 'sd', 'amplitude', 'baseline', 'r2', 'cvr2']

        mask = self.get_volume_mask(session=session, roi=roi, epi_space=True)
        masker = NiftiMasker(mask)

        for parameter_key in keys:
            if cross_validated:
                fn = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject}', f'ses-{session}', 
                        'func', f'sub-{self.subject}_ses-{session}_run-{run}_desc-{parameter_key}.optim_space-T1w_pars.nii.gz')
            else:
                if parameter_key == 'cvr2':
                    fn = op.join(self.bids_folder, 'derivatives', dir.replace('encoding_model', 'encoding_model.cv'), f'sub-{self.subject}', f'ses-{session}', 
                            'func', f'sub-{self.subject}_ses-{session}_desc-{parameter_key}.optim_space-T1w_pars.nii.gz')
                else:
                    fn = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject}', f'ses-{session}', 
                            'func', f'sub-{self.subject}_ses-{session}_desc-{parameter_key}.optim_space-T1w_pars.nii.gz')
            
            pars = pd.Series(masker.fit_transform(fn).ravel())
            parameters.append(pars)

        parameters =  pd.concat(parameters, axis=1, keys=keys, names=['parameter'])

        if return_image:
            return masker.inverse_transform(parameters.T)

        return parameters

    def get_prf_parameters_surf(self, session, run=None, smoothed=False, cross_validated=False, hemi=None, mask=None, space='fsnative',
    parameters=None, key=None, nilearn=False):

        if mask is not None:
            raise NotImplementedError

        if parameters is None:
            parameter_keys = ['mu', 'sd', 'cvr2', 'r2']
        else:
            parameter_keys = parameters

        if hemi is None:
            prf_l = self.get_prf_parameters_surf(session, 
                    run, smoothed, cross_validated, hemi='L',
                    mask=mask, space=space, key=key, parameters=parameters, nilearn=nilearn)
            prf_r = self.get_prf_parameters_surf(session, 
                    run, smoothed, cross_validated, hemi='R',
                    mask=mask, space=space, key=key, parameters=parameters, nilearn=nilearn)
            
            return pd.concat((prf_l, prf_r), axis=0, 
                    keys=pd.Index(['L', 'R'], name='hemi'))


        if key is None:
            if cross_validated:
                dir = 'encoding_model.cv.denoise'
            else:
                dir = 'encoding_model.denoise'

            if smoothed:
                dir += '.smoothed'

            dir += '.natural_space'
        else:
            dir = key

        parameters = []

        for parameter_key in parameter_keys:
            if cross_validated:
                if nilearn:
                    fn = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject}', f'ses-{session}', 
                            'func', f'sub-{self.subject}_ses-{session}_run-{run}_desc-{parameter_key}.optim.nilearn_space-{space}_hemi-{hemi}.func.gii')
                else:
                    fn = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject}', f'ses-{session}', 
                            'func', f'sub-{self.subject}_ses-{session}_run-{run}_desc-{parameter_key}.volume.optim_space-{space}_hemi-{hemi}.func.gii')
            else:
                if nilearn:
                    fn = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject}', f'ses-{session}', 
                            'func', f'sub-{self.subject}_ses-{session}_desc-{parameter_key}.optim.nilearn_space-{space}_hemi-{hemi}.func.gii')
                else:
                    fn = op.join(self.bids_folder, 'derivatives', dir, f'sub-{self.subject}', f'ses-{session}', 
                            'func', f'sub-{self.subject}_ses-{session}_desc-{parameter_key}.volume.optim_space-{space}_hemi-{hemi}.func.gii')

            pars = pd.series(surface.load_surf_data(fn))
            pars.index.name = 'vertex'

            parameters.append(pars)

        return pd.concat(parameters, axis=1, keys=parameter_keys, names=['parameter'])

    def get_surf_info(self):
        info = {'L':{}, 'R':{}}

        for hemi in ['L', 'R']:

            fs_hemi = {'L':'lh', 'R':'rh'}[hemi]

            info[hemi]['inner'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject}', 'ses-1', 'anat', f'sub-{self.subject}_ses-1_hemi-{hemi}_smoothwm.surf.gii')
            info[hemi]['mid'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject}', 'ses-1', 'anat', f'sub-{self.subject}_ses-1_hemi-{hemi}_midthickness.surf.gii')
            info[hemi]['outer'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject}', 'ses-1', 'anat', f'sub-{self.subject}_ses-1_hemi-{hemi}_pial.surf.gii')
            info[hemi]['inflated'] = op.join(self.bids_folder, 'derivatives', 'fmriprep', f'sub-{self.subject}', 'ses-1', 'anat', f'sub-{self.subject}_ses-1_hemi-{hemi}_inflated.surf.gii')
            info[hemi]['curvature'] = op.join(self.bids_folder, 'derivatives', 'freesurfer', f'sub-{self.subject}', 'surf', f'{fs_hemi}.curv')

            for key in info[hemi]:
                assert(os.path.exists(info[hemi][key])), f'{info[hemi][key]} does not exist'

        return info

    def get_fmri_events(self, session, runs=None):

        if runs is None:
            runs = self.get_runs(session)

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

        
        stimulus2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_nr', 'trial_type', 'n2']]
        stimulus2['duration'] = 0.6
        stimulus2['trial_type'] = stimulus2.n2.map(lambda n2: f'n2_{int(n2)}')

        events = pd.concat((stimulus1, stimulus2)).sort_index()

        return events

    def get_target_dir(subject, session, sourcedata, base, modality='func'):
        target_dir = op.join(sourcedata, 'derivatives', base, f'sub-{subject}', f'ses-{session}',
                            modality)

        if not op.exists(target_dir):
            os.makedirs(target_dir)

        return target_dir