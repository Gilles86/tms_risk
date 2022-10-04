import argparse
from tms_risk.cogmodels.evidence_model import (EvidenceModel,
                                               EvidenceModelTwoPriors, EvidenceModelGauss, EvidenceModelDiminishingUtility,
                                               EvidenceModelTwoPriorsDiminishingUtility,
                                               EvidenceModelDifferentEvidence,
                                               EvidenceModelDifferentEvidenceTwoPriors)
from tms_risk.utils.data import get_all_behavior
import os
import os.path as op
import arviz as az
from utils import plot_ppc, cluster_offers


def main(model_label, bids_folder):

    df = get_all_behavior(bids_folder=bids_folder)
    df = df.xs(1, 0, 'session')
    df['log(risky/safe)'] = df.groupby(['subject'],
                                       group_keys=False).apply(cluster_offers)

    trace_folder = op.join(bids_folder, 'derivatives', 'cogmodels')
    trace = az.from_netcdf(
        op.join(trace_folder, f'model-{model_label}_trace.netcdf'))


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
    else:
        raise Exception(f'Do not know model label {model_label}')

    trace = trace.sel(draw=slice(None, None, 5))
    ppc = model.ppc(trace=trace, data=df)

    for plot_type in [1,2,3]:
        for var_name in ['p', 'll_bernoulli']:
            for level in ['group', 'subject']:
                target_folder = op.join(bids_folder, 'derivatives', 'cogmodels', 'plots', level, var_name)

                if not op.exists(target_folder):
                    os.makedirs(target_folder)

                plot_ppc(df, ppc, level=level, plot_type=plot_type, var_name=var_name).savefig(
                    op.join(target_folder, f'plot-{plot_type}_model-{model_label}_pred.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)
