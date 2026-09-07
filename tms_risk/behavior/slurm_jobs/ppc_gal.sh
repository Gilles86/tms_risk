#!/bin/bash
#SBATCH --job-name=ppc_gal
#SBATCH --account=zne.uzh
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/gdehol/logs/ppc_gal_%j.txt
cd $HOME/git/tms_risk
PYTHONPATH=/scratch/gdehol/bauer_multi \
  $HOME/data/conda/envs/tms_risk_cpu/bin/python -m tms_risk.behavior.scripts.extract_anchor_ppc \
  log-power-psd log-power-n2 log-weber-psd log-power-nullind log-power-pmusd \
  --trace_dir /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor \
  --out_dir $HOME/ppc_gal
