import pymc as pm
import numpy as np
import pandas as pd
import aesara.tensor as tt
from patsy import dmatrix
from aesara.tensor.nnet.basic import softplus
from ipywidgets import interact
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from collections.abc import Iterable 

def softplus_np(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


class EvidenceModel(object):

    free_parameters = {'prior_mu':3.,
                       'prior_sd':1.,
                       'evidence_sd':[.25, .25]}

    def __init__(self, data=None, n_subjects=20):

        self.base_numbers = [7, 10, 14, 20, 28]

        if data is not None:
            self.data = data
            self.unique_subjects = data.index.get_level_values('subject').unique()
        else:
            self.unique_subjects = np.arange(1, n_subjects+1)
            self.subject_ix = self.unique_subjects - 1
            data = self.create_data()

        self.n_subjects = len(self.unique_subjects)

        self.coords = {
            "subject": self.unique_subjects,
            "presentation": ['first', 'second'],
        }


    def build_fixed_parameters(self, model):
        with model:
            risky_prior_mu = pm.Deterministic('risky_prior_mu', model['prior_mu'])
            risky_prior_sd = pm.Deterministic('risky_prior_sd', model['prior_sd'])

            safe_prior_mu = pm.Deterministic('safe_prior_mu', model['prior_mu'])
            sae_prior_sd = pm.Deterministic('safe_prior_sd', model['prior_sd'])


    def build_priors(self, model):
        with model:
            # Hyperpriors for group nodes
            prior_mu_mu = pm.HalfNormal("prior_mu_mu", sigma=np.log(20.))
            prior_mu_sd = pm.HalfCauchy('prior_mu_sd', .5)
            prior_mu_offset = pm.Normal('prior_mu_offset', mu=0, sigma=1, dims='subject')  # shape=n_subjects)
            prior_mu = pm.Deterministic('prior_mu', prior_mu_mu + prior_mu_sd * prior_mu_offset,
                                              dims='subject')

            prior_sd_mu = pm.HalfNormal("prior_sd_mu", sigma=1.25)
            prior_sd_sd = pm.HalfCauchy('prior_sd_sd', .5)

            prior_sd = pm.TruncatedNormal('prior_sd',
                                                mu=prior_sd_mu,
                                                sigma=prior_sd_sd,
                                                lower=0,
                                                dims='subject')

            # ix0 = first presented, ix1=later presented
            evidence_sd_mu = pm.HalfNormal(
                "evidence_sd_mu", sigma=1., dims=('presentation'))
            evidence_sd_sd = pm.HalfCauchy(
                "evidence_sd_sd", 1., dims=('presentation'))
            evidence_sd = pm.TruncatedNormal('evidence_sd',
                                             mu=evidence_sd_mu,
                                             sigma=evidence_sd_sd,
                                             lower=0,
                                             dims=('subject', 'presentation'))

            self.build_fixed_parameters(model)

    def build_likelihood(self, paradigm, model, get_rnp=False, get_precision=False):

        risky_prior_mu = model['risky_prior_mu']
        risky_prior_sd = model['risky_prior_sd']
        safe_prior_mu = model['safe_prior_mu']
        safe_prior_sd = model['safe_prior_sd']
        evidence_sd = model['evidence_sd']

        with model:
            post_risky_mu, post_risky_sd = get_posterior(risky_prior_mu[paradigm['subject_ix']],
                                                         risky_prior_sd[paradigm['subject_ix']],
                                                         tt.log(paradigm['n_risky']),
                                                         evidence_sd[paradigm['subject_ix'], paradigm['risky_ix']])

            post_safe_mu, post_safe_sd = get_posterior(safe_prior_mu[paradigm['subject_ix']],
                                                       safe_prior_sd[paradigm['subject_ix']],
                                                       tt.log(paradigm['n_safe']),
                                                       evidence_sd[paradigm['subject_ix'], paradigm['safe_ix']])

            diff_mu, diff_sd = get_diff_dist(post_risky_mu, post_risky_sd, post_safe_mu, post_safe_sd)

            post_risky_mu = pm.Deterministic('post_risky_mu', post_risky_mu)
            p = pm.Deterministic('p', cumulative_normal( tt.log(.55), diff_mu, diff_sd))

            ll = pm.Bernoulli('ll_bernoulli', p=p,
                              observed=paradigm['choices'])

            self._get_rnp_precision(get_rnp, get_precision, paradigm, diff_mu, diff_sd)

    def _get_rnp_precision(self, get_rnp, get_precision, paradigm, diff_mu, diff_sd):
            if get_rnp:
                # Ratio of est(risky/safe) vs actual ratio
                distortion =  tt.exp(((tt.log(paradigm['n_safe']) - tt.log(paradigm['n_risky'])) - diff_mu))
                rnp = pm.Deterministic('rnp', distortion*.55)

            if get_precision:
                precision = pm.Deterministic('precision', 1./diff_sd)

    def predict(self, parameters):

        if ('subject' not in parameters.columns) & (parameters.index.name != 'subject'):
            parameters['subject'] = range(1, len(parameters)+1)

        if not hasattr(self, 'pred_model'):
            self.build_prediction_model(parameters)

        for key in self.free_parameters.keys():

            if isinstance(self.free_parameters[key], Iterable):
                key_ = [f'{key}{i+1}' for i in range(len(self.free_parameters[key]))]
            else:
                key_ = key

            self.pred_model.set_data(key, parameters[key_].values)

        self.data['predicted_p'] = self.pred_model['p'].eval()
        return self.data


    def get_example_parameters(self):
        parameters = pd.DataFrame({'prior_mu': [3.0, 3.0, 4., 4.],
                             'prior_sd':[.5, .5, .5, .5],
                             'evidence_sd1':[.15, .25, .15, .25],
                             'evidence_sd2': [.1, .15, .15, .15]})
        parameters['subject'] = range(1, len(parameters)+1)

        return parameters

    def build_prediction_model(self, parameters):
        parameters = parameters.reset_index()
        self.n_subjects = len(parameters)
        self.unique_subjects = parameters['subject'].values
        coords = { "subject": self.unique_subjects, "presentation": ['first', 'second']}

        self.data = self.create_data()

        with pm.Model(coords=coords) as self.pred_model:

            paradigm = self._get_paradigm(self.data)

            for key in self.free_parameters.keys():
                if isinstance(self.free_parameters[key], Iterable):
                    key_ = [f'{key}{i+1}' for i in range(len(self.free_parameters[key]))]
                    dims = ('subject', 'presentation')
                else:
                    key_ = key
                    dims = 'subject'
                
                pm.MutableData(key, parameters[key_], dims=dims)

        self.build_fixed_parameters(self.pred_model)
        self.build_likelihood(paradigm, self.pred_model, get_rnp=True, get_precision=True)

    def build_estimation_model(self, data=None, get_rnp=False, get_precision=False):

        with pm.Model(coords=self.coords) as self.model:
            paradigm = self._get_paradigm(data=data)

        self.build_priors(self.model)
        self.build_likelihood(paradigm, self.model, get_rnp=get_rnp, get_precision=get_precision)

    def _get_paradigm(self, data=None):

        if data is None:
            data = self.data

        paradigm = {}
        paradigm['n_safe'] = data['n_safe'].values
        paradigm['n_risky'] = data['n_risky'].values

        paradigm['risky_ix'] = (~data['risky_first']).astype(int).values
        paradigm['safe_ix'] = (data['risky_first']).astype(int).values

        paradigm['subject_ix'], _ = pd.factorize(data.index.get_level_values('subject'))

        for key, value in paradigm.items():
            paradigm[key] = pm.MutableData(key, value)

        if 'chose_risky' in data.columns:
            paradigm['choices'] = pm.MutableData('choice', data.chose_risky.values)
        else:
            paradigm['choices'] = None

        return paradigm

    def create_data(self):

        data = pd.MultiIndex.from_product([self.unique_subjects,
                                                np.linspace(
                                                    1, 4, 12), self.base_numbers,
                                                [True, False]],
                                                names=['subject', 'frac', 'n_safe', 'risky_first']).to_frame().reset_index(drop=True)

        data['n_risky'] = (data['frac'] * data['n_safe']).round().values
        data['n_safe'] = data['n_safe'].values
        data['risky/safe'] = data['frac']
        data['log(risky/safe)'] = np.log(data['frac'])

        data['trial_nr'] = data.groupby('subject').cumcount() + 1

        return data.set_index(['subject', 'trial_nr'])

    def sample(self, draws=1000, tune=1000, target_accept=0.9):

        if not hasattr(self, 'model'):
            self.build_estimation_model()

        with self.model:
            self.trace = pm.sample(
                draws, tune=tune, target_accept=target_accept, return_inferencedata=True)

        return self.trace

    def ppc(self, data=None, trace=None, var_names=['p', 'll_bernoulli', 'rnp']):

        if data is None:
            data = self.create_data()

        if trace is None:
            trace = self.trace

        if hasattr(self, 'model'):
            del self.model

        self.build_estimation_model(data=data,
        get_rnp='rnp' in var_names,
        get_precision='precision' in var_names)

        with self.model:
            idata = pm.sample_posterior_predictive(
                
                trace, var_names=var_names)

        pred = [idata['posterior_predictive'][key].to_dataframe() for key in var_names]
        pred = pd.concat(pred, axis=1, keys=var_names, names=['variable'])
        pred = pred.unstack(['chain', 'draw']).droplevel(1, axis=1)
        pred.index = data.index
        pred = pred.set_index(pd.MultiIndex.from_frame(data), append=True)
        pred = pred.stack('variable')
        pred = pred.reorder_levels(np.roll(pred.index.names, 1)).sort_index()

        return pred

    def make_example_data(self, data=None, n=100):

        if data is None:
            data = data

        subjects = data.index.unique(level='subject')

        min_frac = data['frac'].min()
        max_frac = data['frac'].max()

        frac = np.linspace(min_frac, max_frac, n)
        risky_first = [False, True]

        safe_n = data['n_safe'].unique()

        perm = pd.MultiIndex.from_product([subjects, frac, safe_n, risky_first], names=[
                                          'subject', 'frac', 'n_safe', 'risky_first']).to_frame().reset_index(drop=True)
        perm['n_risky'] = perm['n_safe'] * perm['frac']

        perm['n1'] = perm['n_risky'].where(perm['risky_first'], perm['n_safe'])
        perm['n2'] = perm['n_safe'].where(perm['risky_first'], perm['n_risky'])
        perm = perm.set_index('subject')

        perm['log(risky/safe)'] = np.log(perm['frac'])

        return perm


    def get_widget(self):
        pars = self.free_parameters.copy()

        for key in self.free_parameters:
            if isinstance(self.free_parameters[key], Iterable):
                values = pars.pop(key)
                for i in range(len(self.free_parameters[key])):
                    pars[f'{key}{i+1}'] = values[i]

        def make_plot(**parameters):
            parameters = pd.DataFrame([parameters])
            pred = self.predict(parameters).reset_index()

            pred['Safe offer'] = pred['n_safe']
            pred['Predicted prop. accept risky'] = pred['predicted_p']
            pred['Order'] = pred['risky_first'].map({True:'Risky first', False:'Safe first'})

            fig, (ax1, ax2) = plt.subplots(1, 2) 
            fig.set_size_inches(12, 5)
            sns.lineplot(x='Safe offer', y='Predicted prop. accept risky', hue='Order', data=pred.reset_index(), ax=ax1, hue_order=['Safe first', 'Risky first'])
            sns.lineplot(x='log(risky/safe)', y='Predicted prop. accept risky', hue='Order', data=pred.reset_index(), ax=ax2, hue_order=['Safe first', 'Risky first'])
            ax1.set_ylim(0, 1)
            ax2.set_ylim(0, 1)
            ax1.axhline(.5, c='k', ls='--')
            ax2.axhline(.5, c='k', ls='--')
            ax2.axvline(np.log(1./.55), c='k', ls='--')

        kwargs = {}
        for key, value in pars.items():
            kwargs[key] = (0.0, value*2, value/5.)

        return display(interact(make_plot, **kwargs))


class EvidenceModelTwoPriors(EvidenceModel):

    free_parameters = {'risky_prior_mu':3.,
                       'risky_prior_sd':1.,
                       'evidence_sd':[.25, .25]}

    def build_fixed_parameters(self, model):
        with model:
            safe_prior_mu = np.tile(np.mean(np.log(self.base_numbers)), self.n_subjects)
            safe_prior_mu = pm.ConstantData('safe_prior_mu', safe_prior_mu, dims='subject')

            safe_prior_sd = np.tile(np.std(np.log(self.base_numbers)), self.n_subjects)
            safe_prior_sd = pm.ConstantData('safe_prior_sd', safe_prior_sd, dims='subject')

    def build_priors(self, model):
        with model:
            # Hyperpriors for group nodes
            risky_prior_mu_mu = pm.HalfNormal(
                "risky_prior_mu_mu", sigma=np.log(20.))
            risky_prior_mu_sd = pm.HalfCauchy('risky_prior_mu_sd', .5)
            risky_prior_mu_offset = pm.Normal(
                'risky_prior_mu_offset', mu=0, sigma=1, dims='subject')  # shape=n_subjects)
            risky_prior_mu = pm.Deterministic('risky_prior_mu', risky_prior_mu_mu + risky_prior_mu_sd * risky_prior_mu_offset,
                                              dims='subject')

            risky_prior_sd_mu = pm.HalfNormal("risky_prior_sd_mu", sigma=1.25)
            risky_prior_sd_sd = pm.HalfCauchy('risky_prior_sd_sd', .5)

            risky_prior_sd = pm.TruncatedNormal('risky_prior_sd',
                                                mu=risky_prior_sd_mu,
                                                sigma=risky_prior_sd_sd,
                                                lower=0,
                                                dims='subject')

            # ix0 = first presented, ix1=later presented
            evidence_sd_mu = pm.HalfNormal(
                "evidence_sd_mu", sigma=1., dims=('presentation'))
            evidence_sd_sd = pm.HalfCauchy(
                "evidence_sd_sd", 1., dims=('presentation'))
            evidence_sd = pm.TruncatedNormal('evidence_sd',
                                             mu=evidence_sd_mu,
                                             sigma=evidence_sd_sd,
                                             lower=0,
                                             dims=('subject', 'presentation'))

            self.build_fixed_parameters(model)

    def get_example_parameters(self):
        parameters = pd.DataFrame({'risky_prior_mu': [3.0, 3.0, 4., 4.],
                             'risky_prior_sd':[.5, .5, .5, .5],
                             'evidence_sd1':[.15, .25, .15, .25],
                             'evidence_sd2': [.1, .15, .15, .15]})
        parameters['subject'] = range(1, len(parameters)+1)

        return parameters



class EvidenceModelGauss(EvidenceModel):

    free_parameters = {'prior_mu':28.,
                       'prior_sd':10.,
                       'evidence_sd':[2., 2.]}

    def build_likelihood(self, paradigm, model, get_rnp=False, get_precision=False):

        risky_prior_mu = model['risky_prior_mu']
        risky_prior_sd = model['risky_prior_sd']
        safe_prior_mu = model['safe_prior_mu']
        safe_prior_sd = model['safe_prior_sd']
        evidence_sd = model['evidence_sd']

        with model:
            post_risky_mu, post_risky_sd = get_posterior(.55 * risky_prior_mu[paradigm['subject_ix']],
                                                         risky_prior_sd[paradigm['subject_ix']],
                                                         .55 * paradigm['n_risky'],
                                                         evidence_sd[paradigm['subject_ix'], paradigm['risky_ix']])

            post_safe_mu, post_safe_sd = get_posterior(safe_prior_mu[paradigm['subject_ix']],
                                                       safe_prior_sd[paradigm['subject_ix']],
                                                       paradigm['n_safe'],
                                                       evidence_sd[paradigm['subject_ix'], paradigm['safe_ix']])

            diff_mu, diff_sd = get_diff_dist(post_risky_mu, post_risky_sd, post_safe_mu, post_safe_sd)

            post_risky_mu = pm.Deterministic('post_risky_mu', post_risky_mu)
            p = pm.Deterministic('p', cumulative_normal(0.0, diff_mu, diff_sd))

            ll = pm.Bernoulli('ll_bernoulli', p=p,
                              observed=paradigm['choices'])

            self._get_rnp_precision(get_rnp, get_precision, paradigm, diff_mu, diff_sd)

class EvidenceModelDiminishingUtility(EvidenceModel):

    free_parameters = {'prior_mu':3.,
                       'prior_sd':2.,
                       'evidence_sd':[.75, .5],
                       'alpha':1.}

    def build_priors(self, model):

        with model:
            # Hyperpriors for group nodes
            alpha_mu = pm.Normal("alpha_mu", mu=1.0, sigma=0.1)
            alpha_sd = pm.HalfCauchy('alpha_sd', .25)
            alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, dims='subject')  # shape=n_subjects)
            alpha = pm.Deterministic('alpha', alpha_mu + alpha_sd * alpha_offset,
                                              dims='subject')

        super(EvidenceModelDiminishingUtility, self).build_priors(model)

    def build_likelihood(self, paradigm, model, get_rnp=False, get_precision=False):

        risky_prior_mu = model['risky_prior_mu']
        risky_prior_sd = model['risky_prior_sd']
        safe_prior_mu = model['safe_prior_mu']
        safe_prior_sd = model['safe_prior_sd']
        evidence_sd = model['evidence_sd']
        alpha = model['alpha']

        with model:
            post_risky_mu, post_risky_sd = get_posterior(risky_prior_mu[paradigm['subject_ix']],
                                                         risky_prior_sd[paradigm['subject_ix']],
                                                         tt.log(paradigm['n_risky']),
                                                         evidence_sd[paradigm['subject_ix'], paradigm['risky_ix']])

            post_safe_mu, post_safe_sd = get_posterior(safe_prior_mu[paradigm['subject_ix']],
                                                       safe_prior_sd[paradigm['subject_ix']],
                                                       tt.log(paradigm['n_safe']),
                                                       evidence_sd[paradigm['subject_ix'], paradigm['safe_ix']])

            diff_mu, diff_sd = get_diff_dist(alpha[paradigm['subject_ix']]*post_risky_mu,
            alpha[paradigm['subject_ix']]*post_risky_sd,
            alpha[paradigm['subject_ix']]*post_safe_mu,
            alpha[paradigm['subject_ix']]*post_safe_sd)

            p = pm.Deterministic('p', cumulative_normal( tt.log(.55), diff_mu, diff_sd))

            ll = pm.Bernoulli('ll_bernoulli', p=p,
                              observed=paradigm['choices'])

            self._get_rnp_precision(get_rnp, get_precision, paradigm, diff_mu, diff_sd)

class EvidenceModelTwoPriorsDiminishingUtility(EvidenceModelTwoPriors, EvidenceModelDiminishingUtility):

    free_parameters = {'risky_prior_mu':3.,
                       'risky_prior_sd':2.,
                       'evidence_sd':[.75, .5],
                       'alpha':1.}

    def build_priors(self, model):

        with model:
            # Hyperpriors for group nodes
            alpha_mu = pm.Normal("alpha_mu", mu=1.0, sigma=0.5)
            alpha_sd = pm.HalfCauchy('alpha_sd', .25)
            alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, dims='subject')  # shape=n_subjects)
            alpha = pm.Deterministic('alpha', alpha_mu + alpha_sd * alpha_offset,
                                              dims='subject')

        EvidenceModelTwoPriors.build_priors(self, model)

    def build_likelihood(self, paradigm, model, get_rnp, get_precision):
        EvidenceModelDiminishingUtility.build_likelihood(self, paradigm, model, get_rnp, get_precision)

class EvidenceModelDifferentEvidence(EvidenceModel):
    free_parameters = {'prior_mu':3.,
                    'prior_sd':2.,
                    'evidence_sd_safe':[.75, .5],
                    'evidence_sd_risky':[.75, .5]}

    def build_priors(self, model):
        with model:
            # Hyperpriors for group nodes
            prior_mu_mu = pm.HalfNormal("prior_mu_mu", sigma=np.log(20.))
            prior_mu_sd = pm.HalfCauchy('prior_mu_sd', .5)
            prior_mu_offset = pm.Normal('prior_mu_offset', mu=0, sigma=1, dims='subject')  # shape=n_subjects)
            prior_mu = pm.Deterministic('prior_mu', prior_mu_mu + prior_mu_sd * prior_mu_offset,
                                            dims='subject')

            prior_sd_mu = pm.HalfNormal("prior_sd_mu", sigma=1.25)
            prior_sd_sd = pm.HalfCauchy('prior_sd_sd', .5)

            prior_sd = pm.TruncatedNormal('prior_sd',
                                                mu=prior_sd_mu,
                                                sigma=prior_sd_sd,
                                                lower=0,
                                                dims='subject')

            # ix0 = first presented, ix1=later presented
            evidence_sd_safe_mu = pm.HalfNormal(
                "evidence_sd_safe_mu", sigma=1., dims=('presentation'))
            evidence_sd_safe_sd = pm.HalfCauchy(
                "evidence_sd_safe_sd", 1., dims=('presentation'))
            evidence_sd_safe = pm.TruncatedNormal('evidence_sd_safe',
                                            mu=evidence_sd_safe_mu,
                                            sigma=evidence_sd_safe_sd,
                                            lower=0,
                                            dims=('subject', 'presentation'))

            evidence_sd_risky_mu = pm.HalfNormal(
                "evidence_sd_risky_mu", sigma=1., dims=('presentation'))
            evidence_sd_risky_sd = pm.HalfCauchy(
                "evidence_sd_risky_sd", 1., dims=('presentation'))
            evidence_sd_risky = pm.TruncatedNormal('evidence_sd_risky',
                                            mu=evidence_sd_risky_mu,
                                            sigma=evidence_sd_risky_sd,
                                            lower=0,
                                            dims=('subject', 'presentation'))

    def build_likelihood(self, paradigm, model, get_rnp=False, get_precision=False):

        risky_prior_mu = model['prior_mu']
        risky_prior_sd = model['prior_sd']
        safe_prior_mu = model['prior_mu']
        safe_prior_sd = model['prior_sd']
        evidence_sd_safe = model['evidence_sd_safe']
        evidence_sd_risky = model['evidence_sd_risky']

        with model:
            post_risky_mu, post_risky_sd = get_posterior(.55 * risky_prior_mu[paradigm['subject_ix']],
                                                        risky_prior_sd[paradigm['subject_ix']],
                                                        .55 * paradigm['n_risky'],
                                                        evidence_sd_risky[paradigm['subject_ix'], paradigm['risky_ix']])

            post_safe_mu, post_safe_sd = get_posterior(safe_prior_mu[paradigm['subject_ix']],
                                                    safe_prior_sd[paradigm['subject_ix']],
                                                    paradigm['n_safe'],
                                                    evidence_sd_safe[paradigm['subject_ix'], paradigm['safe_ix']])

            diff_mu, diff_sd = get_diff_dist(post_risky_mu, post_risky_sd, post_safe_mu, post_safe_sd)

            post_risky_mu = pm.Deterministic('post_risky_mu', post_risky_mu)
            p = pm.Deterministic('p', cumulative_normal(0.0, diff_mu, diff_sd))

            ll = pm.Bernoulli('ll_bernoulli', p=p,
                            observed=paradigm['choices'])
 
            self._get_rnp_precision(get_rnp, get_precision, paradigm, diff_mu, diff_sd)

class EvidenceModelDifferentEvidenceTwoPriors(EvidenceModelDifferentEvidence, EvidenceModelTwoPriors):
    free_parameters = {'prior_risky_mu':3.,
                    'prior_risky_sd':2.,
                    'evidence_sd_safe':[.75, .5],
                    'evidence_sd_risky':[.75, .5]}

    def build_priors(self, model):
        with model:
            # Hyperpriors for group nodes
            risky_prior_mu_mu = pm.HalfNormal("risky_prior_mu_mu", sigma=np.log(20.))
            risky_prior_mu_sd = pm.HalfCauchy('risky_prior_mu_sd', .5)
            risky_prior_mu_offset = pm.Normal('risky_prior_mu_offset', mu=0, sigma=1, dims='subject')  # shape=n_subjects)
            risky_prior_mu = pm.Deterministic('risky_prior_mu', risky_prior_mu_mu + risky_prior_mu_sd * risky_prior_mu_offset,
                                            dims='subject')

            risky_prior_sd_mu = pm.HalfNormal("risky_prior_sd_mu", sigma=1.25)
            risky_prior_sd_sd = pm.HalfCauchy('risky_prior_sd_sd', .5)

            risky_prior_sd = pm.TruncatedNormal('risky_prior_sd',
                                                mu=risky_prior_sd_mu,
                                                sigma=risky_prior_sd_sd,
                                                lower=0,
                                                dims='subject')

            # ix0 = first presented, ix1=later presented
            evidence_sd_safe_mu = pm.HalfNormal(
                "evidence_sd_safe_mu", sigma=1., dims=('presentation'))
            evidence_sd_safe_sd = pm.HalfCauchy(
                "evidence_sd_safe_sd", 1., dims=('presentation'))
            evidence_sd_safe = pm.TruncatedNormal('evidence_sd_safe',
                                            mu=evidence_sd_safe_mu,
                                            sigma=evidence_sd_safe_sd,
                                            lower=0,
                                            dims=('subject', 'presentation'))

            evidence_sd_risky_mu = pm.HalfNormal(
                "evidence_sd_risky_mu", sigma=1., dims=('presentation'))
            evidence_sd_risky_sd = pm.HalfCauchy(
                "evidence_sd_risky_sd", 1., dims=('presentation'))
            evidence_sd_risky = pm.TruncatedNormal('evidence_sd_risky',
                                            mu=evidence_sd_risky_mu,
                                            sigma=evidence_sd_risky_sd,
                                            lower=0,
                                            dims=('subject', 'presentation'))

            EvidenceModelTwoPriors.build_fixed_parameters(self, model)

    def build_likelihood(self, paradigm, model, get_rnp=False, get_precision=False):

        risky_prior_mu = model['risky_prior_mu']
        risky_prior_sd = model['risky_prior_sd']
        safe_prior_mu = model['safe_prior_mu']
        safe_prior_sd = model['safe_prior_sd']
        evidence_sd_safe = model['evidence_sd_safe']
        evidence_sd_risky = model['evidence_sd_risky']

        with model:
            post_risky_mu, post_risky_sd = get_posterior(.55 * risky_prior_mu[paradigm['subject_ix']],
                                                        risky_prior_sd[paradigm['subject_ix']],
                                                        .55 * paradigm['n_risky'],
                                                        evidence_sd_risky[paradigm['subject_ix'], paradigm['risky_ix']])

            post_safe_mu, post_safe_sd = get_posterior(safe_prior_mu[paradigm['subject_ix']],
                                                    safe_prior_sd[paradigm['subject_ix']],
                                                    paradigm['n_safe'],
                                                    evidence_sd_safe[paradigm['subject_ix'], paradigm['safe_ix']])

            diff_mu, diff_sd = get_diff_dist(post_risky_mu, post_risky_sd, post_safe_mu, post_safe_sd)

            post_risky_mu = pm.Deterministic('post_risky_mu', post_risky_mu)
            p = pm.Deterministic('p', cumulative_normal(0.0, diff_mu, diff_sd))

            ll = pm.Bernoulli('ll_bernoulli', p=p,
                            observed=paradigm['choices'])
 
            self._get_rnp_precision(get_rnp, get_precision, paradigm, diff_mu, diff_sd)

def get_posterior(mu1, sd1, mu2, sd2):

    var1, var2 = sd1**2, sd2**2

    return mu1 + (var1/(var1+var2))*(mu2 - mu1), tt.sqrt((var1*var2)/(var1+var2))


def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, tt.sqrt(sd1**2+sd2**2)


def cumulative_normal(x, mu, sd, s=np.sqrt(2.)):
    #     Cumulative distribution function for the standard normal distribution
    return tt.clip(0.5 + 0.5 *
                   tt.erf((x - mu) / (sd*s)), 1e-9, 1-1e-9)
