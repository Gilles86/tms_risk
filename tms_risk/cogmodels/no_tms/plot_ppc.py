import argparse
from tms_risk.cogmodels.evidence_model import (EvidenceModel,
                                               EvidenceModelTwoPriors, EvidenceModelGauss, EvidenceModelDiminishingUtility,
                                               EvidenceModelTwoPriorsDiminishingUtility,
                                               EvidenceModelDifferentEvidence,
                                               EvidenceModelDifferentEvidenceTwoPriors,
                                               WoodfordModel)
from tms_risk.cogmodels.utils import format_bambi_ppc
from tms_risk.utils.data import get_all_behavior
import os
import os.path as op
import arviz as az
from utils import plot_ppc, cluster_offers, format_bambi_ppc
from fit_probit import build_model


def main(model_label, bids_folder, group_only=False, subject=None, col_wrap=5):

    df = get_all_behavior(bids_folder=bids_folder)
    df = df.xs(1, 0, 'session')

    plot_ppc_(df, model_label, bids_folder, group_only, subject, col_wrap)

    trace_folder = op.join(bids_folder, 'derivatives', 'cogmodels')
    trace = az.from_netcdf(
        op.join(trace_folder, f'model-{model_label}_trace.netcdf'))


def plot_ppc_(df, trace, model_label, bids_folder, group_only=False, subject=None, col_wrap=5, thin=None):

    df['log(risky/safe)'] = df.groupby(['subject'],
                                       group_keys=False).apply(cluster_offers)

    if thin is not None: 
        trace = trace.sel(draw=slice(None, None, thin))


    if group_only:
        levels = ['group']
    else:
        levels = ['group', 'subject']

    if model_label == '1':
        model = EvidenceModel(df)
    elif model_label == '2':
        model = EvidenceModelTwoPriors(df)
    elif model_label == '3':
        model = EvidenceModelGauss(df)
    elif model_label == '4':
        model = EvidenceModelDiminishingUtility(df)
    elif model_label == '5':
        model = EvidenceModelTwoPriorsDiminishingUtility(df)
    elif model_label == '6':
        model = EvidenceModelDifferentEvidence(df)
    elif model_label == '7':
        model = EvidenceModelDifferentEvidenceTwoPriors(df)
    elif model_label == '8':
        model = WoodfordModel(df)
    elif model_label.startswith('probit'):
        model = build_model(model_label[-1], df)
    else:
        raise Exception(f'Do not know model label {model_label}')

    if model_label.startswith('probit'):
        ppc = format_bambi_ppc(trace, model, df)
    else:
        ppc = model.ppc(trace=trace, data=df)

    if subject is not None:
        df = df.xs(int(subject), 0, 'subject', drop_level=False)
        ppc = ppc.xs(int(subject), 0, 'subject', drop_level=False)
        levels = ['subject']
        col_wrap = 1

    for plot_type in [1,2,3,5]:
        for var_name in ['p', 'll_bernoulli']:
            for level in levels:
                target_folder = op.join(bids_folder, 'derivatives', 'cogmodels', 'plots', level, var_name)

                if not op.exists(target_folder):
                    os.makedirs(target_folder)

                if subject is None:
                    fn = f'plot-{plot_type}_model-{model_label}_pred.pdf'
                else:
                    fn = f'plot-{plot_type}_model-{model_label}_subject-{subject}_pred.pdf'

                plot_ppc(df, ppc, level=level, plot_type=plot_type, var_name=var_name, col_wrap=col_wrap).savefig(
                    op.join(target_folder, fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--group_only', action='store_true')
    parser.add_argument('--subject', default=None)
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder, group_only=args.group_only, subject=args.subject)
