#!/bin/bash
#SBATCH --job-name=probit_ppc_ml
#SBATCH --account=zne.uzh
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/gdehol/logs/probit_ppc_ml_%A_%a.txt
#SBATCH --array=0-3
LABELS=(log-power-n1n2 log-power-n1n2psd log-power-n2psd log-power-n1psd)
L=${LABELS[$SLURM_ARRAY_TASK_ID]}
cd $HOME/git/tms_risk
PYTHONPATH=/scratch/gdehol/bauer_multi \
  $HOME/data/conda/envs/tms_risk_cpu/bin/python -m tms_risk.behavior.scripts.probit_ppc_ml \
  "$L" --trace_dir /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor \
  --out_dir $HOME/probit_ppc_ml --n_draws 200
