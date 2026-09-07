#!/bin/bash
#SBATCH --job-name=probit_ppc
#SBATCH --account=zne.uzh
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/gdehol/logs/probit_ppc_%A_%a.txt
#SBATCH --array=0-1
LABELS=(log-power-n1n2 log-power-n1n2psd)
L=${LABELS[$SLURM_ARRAY_TASK_ID]}
cd $HOME/git/tms_risk
PYTHONPATH=/scratch/gdehol/bauer_multi \
  $HOME/data/conda/envs/tms_risk_cpu/bin/python -m tms_risk.behavior.scripts.extract_anchor_probit_ppc \
  "$L" --trace_dir /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor \
  --out_dir $HOME/probit_ppc --n_draws 200
