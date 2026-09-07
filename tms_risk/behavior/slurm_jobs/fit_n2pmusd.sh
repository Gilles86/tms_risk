#!/bin/bash
#SBATCH --job-name=n2pmusd
#SBATCH --account=zne.uzh
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/home/gdehol/logs/n2pmusd_%A_%a.txt
#SBATCH --array=0-1
LABELS=(log-power-n2pmusd log-power-n1n2pmusd)
L=${LABELS[$SLURM_ARRAY_TASK_ID]}
cd $HOME/git/tms_risk
PYTHONPATH=/scratch/gdehol/bauer_multi \
  $HOME/data/conda/envs/tms_risk_cpu/bin/python -m tms_risk.behavior.fit_anchor "$L" \
  --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk --out_folder cogmodels.anchor \
  --chains 8 --tune 3000 --draws 3000
