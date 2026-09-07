#!/bin/bash
# Second attempt at log-weber+affine-n1n2.  The first (refit_weber_affine.sh)
# showed the failure is NOT an initialisation artefact: over 32 chains x 4
# inits, ~1 in 8 lands on a ridge that trades prior LOCATION and WIDTH against
# second-option noise, at wildly different points along it (safe prior mu 11,
# 15, 31, 33 CHF).  mapjitter passing 8/8 was luck, not a fix.
#
# So close the ridge instead of re-rolling the dice.  PRIOR_SPEC puts
# sigma_intercept = 1.0 on *_prior_mu, which in log space lets the group prior
# mean sit a factor of e from the payoff geometric mean -- a safe-option prior
# centred at 31 CHF when no safe payoff exceeds 28 is inside one SD.  Tightening
# it is a modelling statement, not a sampler tweak, so it is stamped into the
# trace (`tms_risk_prior_spec` gains `+spm<sigma>`) and the filename.
#
# Two widths, because changing a prior to fix sampling demands a sensitivity
# check: 0.4 puts the ridge at 2.0 SD, 0.6 at 1.3 SD.  If the two posteriors
# agree, the tightening is doing nothing but removing the ridge.  Both use the
# DEFAULT init, which was the worst performer (2-3 rogue chains of 8), plus one
# pathfinder arm at 0.4 -- pathfinder is what found the far end of the ridge.
#SBATCH --job-name=wa_tight
#SBATCH --account=zne.uzh
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/gdehol/logs/wa_tight_%A_%a.txt
#SBATCH --array=0-2
SPM=(0.4 0.6 0.4)
INIT=("" "" "--find_init pathfinder")
cd $HOME/git/tms_risk
PYTHONPATH=/scratch/gdehol/bauer_multi \
  $HOME/data/conda/envs/tms_risk_cpu/bin/python -m tms_risk.behavior.fit_anchor \
  log-weber+affine-n1n2 \
  --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk --out_folder cogmodels.anchor \
  --chains 8 --tune 5000 --draws 3000 --target_accept 0.95 \
  --sigma_prior_mu ${SPM[$SLURM_ARRAY_TASK_ID]} ${INIT[$SLURM_ARRAY_TASK_ID]}
