#!/bin/bash
#SBATCH --job-name=extract_n2psd
#SBATCH --account=zne.uzh
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/gdehol/logs/extract_n2psd_%j.txt
set -x
cd $HOME/git/tms_risk
export PYTHONPATH=/scratch/gdehol/bauer_multi
PY=$HOME/data/conda/envs/tms_risk_cpu/bin/python
TD=/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor
# a directory holding only the new traces, so the whole-directory extractors
# do not re-walk all ~130 fits
NEW=$HOME/anchor_new
rm -rf $NEW && mkdir -p $NEW
for L in log-power-n2psd log-power-n1psd; do
  ln -s $TD/model-${L}_trace.netcdf $NEW/
done
OUT=$HOME/psd_extract
mkdir -p $OUT
$PY -m tms_risk.behavior.scripts.extract_anchor_curves --trace_dir $NEW \
    --out_tsv $OUT/anchor_curves.tsv --out_subject_tsv $OUT/anchor_curves_subject.tsv
$PY -m tms_risk.behavior.scripts.extract_anchor_priors --trace_dir $NEW --out_dir $OUT
for L in log-power-n2psd log-power-n1psd; do
  $PY -m tms_risk.behavior.scripts.extract_anchor_loo $L --trace_dir $TD --out_dir $OUT/loo_anchor
  $PY -m tms_risk.behavior.scripts.extract_anchor_decision_function $L --trace_dir $TD --out_dir $OUT/decision_function
  $PY -m tms_risk.behavior.scripts.extract_anchor_subject_shifts $L --trace_dir $TD --out_dir $OUT/subject_shifts
done
$PY -m tms_risk.behavior.scripts.extract_anchor_ppc log-power-n2psd log-power-n1psd \
    --trace_dir $TD --out_dir $OUT/ppc_anchor
