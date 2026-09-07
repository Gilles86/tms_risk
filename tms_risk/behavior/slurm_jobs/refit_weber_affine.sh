#!/bin/bash
# Refit log-weber+affine-n1n2, which failed convergence on ONE chain out of
# eight (max r_hat 1.29, min ESS 20; drop chain 0 and it is 1.002 / 3960).
# Chain 0 sat in a second mode with both priors pushed far above the payoff
# range (safe prior mu 33.5 CHF against 11.0 for the other seven), paid for
# with more second-option noise -- the w = sd^2/(sd^2 + nu^2) ridge. No
# divergences, so this is genuine multimodality, not a funnel, and the lever
# is starting-point diversity rather than target_accept.
#
# Three arms, identical except for the initialisation, so the answer is
# attributable:
#   0  default (pymc jitter+adapt_diag) -- was pathfinder to blame?
#   1  mapjitter                        -- a different deterministic seed
#   2  priorjitter                      -- widest starts; if every chain from
#                                          a broad prior draw lands in the
#                                          same mode, the second one is
#                                          negligible rather than merely
#                                          unvisited
# Sampler settings otherwise match the original stamp exactly
# (chains=8 tune=5000 draws=3000 ta=0.95) so only the init differs.
#SBATCH --job-name=wa_refit
#SBATCH --account=zne.uzh
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/gdehol/logs/wa_refit_%A_%a.txt
#SBATCH --array=0-2
INITS=("" "--find_init mapjitter" "--find_init priorjitter")
INIT=${INITS[$SLURM_ARRAY_TASK_ID]}
cd $HOME/git/tms_risk
PYTHONPATH=/scratch/gdehol/bauer_multi \
  $HOME/data/conda/envs/tms_risk_cpu/bin/python -m tms_risk.behavior.fit_anchor \
  log-weber+affine-n1n2 \
  --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk --out_folder cogmodels.anchor \
  --chains 8 --tune 5000 --draws 3000 --target_accept 0.95 $INIT
