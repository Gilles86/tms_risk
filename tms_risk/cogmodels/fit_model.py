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


def main(model_label, burnin=500, samples=500, bids_folder='/data/ds-tmsrisk'):

    df = get_all_behavior(bids_folder=bids_folder)
    print(df)
    df = df.xs(1, 0, 'session')

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    target_accept = 0.9

    if model_label == '1':
        model = EvidenceModel(df)
    elif model_label == '2':
        model = EvidenceModelTwoPriors(df)
    elif model_label == '3':
        model = EvidenceModelGauss(df)
    elif model_label == '4':
        model = EvidenceModelDiminishingUtility(df)
        target_accept = 0.95
    elif model_label == '5':
        model = EvidenceModelTwoPriorsDiminishingUtility(df)
        target_accept = 0.9
    elif model_label == '6':
        model = EvidenceModelDifferentEvidence(df)
    elif model_label == '7':
        model = EvidenceModelDifferentEvidenceTwoPriors(df)
        target_accept = 0.9
    else:
        raise Exception(f'Do not know model label {model_label}')

    trace = model.sample(burnin, samples, target_accept=target_accept)
    az.to_netcdf(trace,
                 op.join(target_folder, f'model-{model_label}_trace.netcdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)
