#!/bin/bash
#SBATCH --job-name=base_forms
#SBATCH --account=zne.uzh
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/gdehol/logs/base_forms_%A_%a.txt
#SBATCH --array=0-5
FORMS=(weber affine power spl3 cspl3 cspl5)
F=${FORMS[$SLURM_ARRAY_TASK_ID]}
cd $HOME/git/tms_risk
PYTHONPATH=/scratch/gdehol/bauer_multi \
  $HOME/data/conda/envs/tms_risk_cpu/bin/python -m tms_risk.behavior.fit_anchor \
  "log-${F}-nullind" --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk \
  --out_folder cogmodels.baseline --data_label baseline_all \
  --chains 8 --tune 3000 --draws 3000
