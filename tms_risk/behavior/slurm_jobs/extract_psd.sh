#!/bin/bash
#SBATCH --job-name=extract_psd
#SBATCH --account=zne.uzh
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/gdehol/logs/extract_psd_%j.txt
cd $HOME/git/tms_risk
export PYTHONPATH=/scratch/gdehol/bauer_multi
PY=$HOME/data/conda/envs/tms_risk_cpu/bin/python
TD=/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor
$PY -m tms_risk.behavior.scripts.extract_anchor_loo --trace_dir $TD --out_dir $HOME/loo_psd
$PY -m tms_risk.behavior.scripts.extract_anchor_curves --trace_dir $TD \
    --out_tsv $HOME/curves_psd.tsv --out_subject_tsv $HOME/curves_psd_subject.tsv
$PY -m tms_risk.behavior.scripts.extract_anchor_ppc log-power-n2psd log-power-n1psd \
    --trace_dir $TD --out_dir $HOME/ppc_psd2
