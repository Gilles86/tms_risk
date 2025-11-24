import pandas as pd
import argparse
from bauer.models import PsychometricModel
from pathlib import Path
from tms_risk.utils.data import get_all_behavior
import numpy as np
import pingouin as pg
from tqdm import tqdm

bids_folder = '/data/ds-tmsrisk'



def main(model_label, bids_folder):

    target_dir = Path(bids_folder) / 'derivatives' / 'map_models'

    target_dir.mkdir(parents=True, exist_ok=True)

    assert model_label in ['psychometric_simple', 'psychometric_order'], 'model_label must be one of: psychometric_order'

    # Load data
    df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True, exclude_outliers=True)
    df = df.drop('baseline', level='stimulation_condition')

    df['x1'] = np.log(df['n_safe'])
    df['x2'] = np.log(df['n_risky'])
    df['choice'] = df['chose_risky']

    model = PsychometricModel()

    pars = []
    keys = []

    if model_label == 'psychometric_simple':
        groupby = ['subject', 'stimulation_condition']
    elif model_label == 'psychometric_order':
        groupby = ['subject', 'stimulation_condition', 'risky_first']

    for key, d in tqdm(list(df.groupby(groupby))):
        model.build_estimation_model(data=d, hierarchical=False)
        pars.append(model.fit_map(progressbar=False))
        keys.append(key)
    
    pars = pd.DataFrame(pars, index=pd.MultiIndex.from_tuples(keys, names=groupby))

    pars.to_csv(target_dir / f'{model_label}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit MAP model to all subjects')
    parser.add_argument('model_label', type=str, help='Model label')
    parser.add_argument('--bids_folder', type=str, help='BIDS folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.model_label, args.bids_folder)