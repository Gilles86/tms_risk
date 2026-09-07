# Handoff: cross-validated R² for the full nPRF encoding-model set (m0–m5)

Written 2026-08-19. Self-contained — read this before touching anything in
`derivatives/encoding_model2.model-*.smoothed.cv/`.

Companion notes (background, not required): `notes/encoding_model_set_2026-08.md`
(what the six models are, and the voxel-selection finding),
`notes/encoding_model_choice.md` (m1-vs-m2 on NPC12r, the convergence problem, and the
first computation of the null), `notes/analyses/cvr2_model_comparison.md` (the original
per-voxel win-count framing, superseded by the mean-cvR² analysis).

---

## 1. TL;DR

On the stimulation-site mask (`NPCr2cm-cluster`), with all six models now
cross-validated for the first time:

- **m1 — amplitude free per session, everything else pooled — wins outright.** Best mean
  cvR², best in 26/35 subjects, beats all five other models pairwise, and is the only
  model whose margin over a properly computed null is unambiguous (p = 0.011).
- **m4 (tuning free) and m5 (magnitude free) do not separate out of sample**
  (Δ = +0.0010 for m5, 95 % CI [−0.0009, +0.0027], p = 0.27). Cross-validation therefore
  **cannot** arbitrate the tuning-vs-magnitude question. The specificity claim still
  rests on the parameter-level result (amplitude moves in every model that frees it;
  mu/sd never move in any model that frees them).
- The ordering across the set is a clean complexity penalty: **one** free per-session
  parameter (m1) > **two** (m3, m5, m4) > **four** (m2) > **none** (m0). On this mask CV
  is mostly measuring the cost of extra per-session parameters.
- **The m0/m1/m2 numbers changed materially versus the 2026-08-06 run**, because the CV
  trees were rewritten on the cluster 2026-08-13→15. m1's margin over the null went from
  +0.0076 (p = 0.073) to **+0.0131 (p = 0.011)**. Anything quoting
  `notes/data/cvr2_vs_null.tsv` must be requoted from
  `notes/data/cvr2_vs_null_m0-5.tsv`.

Nothing here changes the paper's canonical model: m1 is what every downstream
decode / Fisher / mc_decode script uses, and m1 is what CV selects.

---

## 2. The model set

Each voxel: `amplitude · exp(−½((log n − mu)/sd)²) + baseline`. The models differ only in
which parameters are free to differ between the two TMS sessions — and "per session"
**is** the IPS-vs-vertex contrast, since sessions 2 and 3 are the two arms.

| | amplitude | mu | sd | baseline | question it asks |
|---|---|---|---|---|---|
| m0 | — | — | — | — | nothing changes (reference) |
| **m1** | ✓ | — | — | — | pure gain change — **the paper's canonical model** |
| m3 | ✓ | — | ✓ | — | gain + width |
| **m4** | — | ✓ | ✓ | — | **tuning**: what the population is tuned to |
| **m5** | ✓ | — | — | ✓ | **response magnitude**: how strongly it responds |
| m2 | ✓ | ✓ | ✓ | ✓ | everything |

Six labels, five with any free per-session parameter. All six grid specs were verified
parameter-by-parameter (see `notes/encoding_model_set_2026-08.md` §1); a wrong grid
ordering yields plausible-looking but meaningless fits, so do not skip that check if you
add a model.

---

## 3. What was run (2026-08-19)

SLURM job **5103983** on sciencecluster, 2 cores / 24 GB, completed in **2 min 45 s**,
exit 0, all 35 subjects.

```bash
ssh sciencecluster 'cd ~/git/tms_risk && sbatch --account=zne.uzh --job-name=cvr2_m0-5 \
  -c 2 --mem=24G --time=3:00:00 \
  --output=/home/gdehol/logs/cvr2_vs_null_m0-5_%j.txt \
  --wrap="$HOME/data/conda/envs/tms_risk_cpu/bin/python -m tms_risk.modeling.scripts.cvr2_vs_null \
    --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk --roi NPCr2cm-cluster \
    --models 0,1,2,3,4,5 --out_tsv notes/data/cvr2_vs_null_m0-5.tsv"'
```

| What | Where |
|---|---|
| Script | `/Users/gdehol/git/tms_risk/tms_risk/modeling/scripts/cvr2_vs_null.py` |
| Output TSV (local) | `/Users/gdehol/git/tms_risk/notes/data/cvr2_vs_null_m0-5.tsv` |
| Output TSV (cluster) | `/home/gdehol/git/tms_risk/notes/data/cvr2_vs_null_m0-5.tsv` |
| SLURM log | `/home/gdehol/logs/cvr2_vs_null_m0-5_5103983.txt` |
| Inputs | `/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/encoding_model2.model-{0..5}.smoothed.cv/sub-XX/sub-XX_desc-cvr2.optim_space-T1w_pars.nii.gz` |
| Also read | `derivatives/glm_stim1.denoise.smoothed/` (to recompute the null), `derivatives/ips_masks/` (ROI + EPI brain mask) |
| Superseded | `/Users/gdehol/git/tms_risk/notes/data/cvr2_vs_null.tsv` (2026-08-06, m0/m1/m2 only) — kept, not overwritten |

Subject set: the 35 hardcoded in `cvr2_vs_null.py` (`tms_keys.yml` ∩ encoding_model2 on
disk; excludes sub-22 and sub-49, who have no PRF fits). Mask `NPCr2cm-cluster` = the
2 cm stimulation cluster, 11 022 voxels summed over subjects (~315/subject).

**The null** is *not* cvR² = 0. braincoder's `get_rsq` puts the held-out fold's **own**
mean in the denominator, so cvR² = 0 means "as good as already knowing the test fold's
mean" — which a real null predictor does not know. `cvr2_vs_null.py` recomputes, per
voxel and under the identical leave-one-run-out folds, the cvR² of predicting the
**training** mean. That null sits at **−0.01776** here. Every number below is to be read
against that offset, never against zero.

---

## 4. Results

35 subjects, 11 022 voxels, null = **−0.01776**. "Mean cvR²" = mean over ROI voxels
within subject, then over subjects; tests are paired across subjects (n = 35, df = 34).

| model | free per session | mean cvR² | Δ vs null | t, p | % voxels beating null | % voxels > 0 | subjects > null |
|---|---|---|---|---|---|---|---|
| **m1** | amplitude | **−0.00466** | **+0.01311** | 2.67, **0.011** | **48.9 %** | **36.1 %** | 23/35 |
| m3 | amplitude + sd | −0.00816 | +0.00960 | 2.08, 0.045 | 45.6 % | 33.9 % | 21/35 |
| m5 | amplitude + baseline | −0.00915 | +0.00861 | 1.76, 0.088 | 45.8 % | 34.0 % | 21/35 |
| m4 | mu + sd | −0.01018 | +0.00758 | 1.59, 0.12 | 43.9 % | 32.7 % | 20/35 |
| m2 | all four | −0.01753 | +0.00023 | 0.05, 0.96 | 40.3 % | 30.4 % | 15/35 |
| m0 | nothing | −0.02840 | −0.01063 | −4.62, 0.0001 | 27.6 % | 14.9 % | 6/35 |

**Pairwise, m1 against each other model** (Δ = m1 − other, 95 % CI = 10 k bootstrap over
subjects, seed 0):

| contrast | Δ cvR² | 95 % CI | t(34) | p | subjects favouring m1 |
|---|---|---|---|---|---|
| m1 − m0 | +0.02374 | [+0.01661, +0.03160] | +6.09 | 7e−7 | 32/35 |
| m1 − m2 | +0.01287 | [+0.01098, +0.01461] | +13.70 | 2e−15 | 34/35 |
| m1 − m4 | +0.00552 | [+0.00313, +0.00745] | +4.91 | 2e−5 | 31/35 |
| m1 − m5 | +0.00450 | [+0.00267, +0.00614] | +5.11 | 1e−5 | 33/35 |
| m1 − m3 | +0.00351 | [+0.00122, +0.00537] | +3.19 | 0.0030 | 29/35 |

**The two-parameter models do not separate from each other:**

| contrast | Δ cvR² | 95 % CI | t(34) | p | subjects favouring first |
|---|---|---|---|---|---|
| **m5 − m4** (magnitude vs tuning) | +0.00103 | [−0.00085, +0.00265] | +1.13 | **0.27** | 24/35 |
| m3 − m5 | +0.00099 | [−0.00064, +0.00300] | +1.07 | **0.29** | 16/35 (mean is outlier-driven — the median subject favours m5) |

**Per-subject winner** (highest mean cvR² of the six): m1 **26**, m3 5, m0 2, m5 1, m4 1.
Mean rank (1 = best): m1 **1.46**, m3 2.74, m5 2.89, m4 3.51, m0 5.20, m2 5.20.

### 4b. The refreshed m0/m1/m2 versus 2026-08-06

The cluster CV trees were rewritten after the Aug-6 run (m0 entirely on 08-13→15; m1/m2
partially on 08-13). Same 35 subjects, same mask, paired:

| model | Aug 6 | 2026-08-19 | Δ | p |
|---|---|---|---|---|
| m0 | −0.02432 | −0.02840 | −0.00407 | <0.0001 |
| **m1** | −0.01014 | **−0.00466** | **+0.00549** | 0.0096 |
| m2 | −0.01452 | −0.01753 | −0.00301 | 0.016 |

Consequence worth carrying forward: **m1 now beats the null significantly (p = 0.011) on
the stimulation-site mask, where on the Aug-6 numbers it was borderline (p = 0.073).**

---

## 5. How to reproduce the statistics from the TSV

The TSV has one row per subject with `null_mean`, `m{0..5}_mean`, `m{0..5}_beats_null`,
`m{0..5}_gt0`, `n_voxels`. Everything in §4 comes from this snippet — no cluster, no
NIfTIs:

```python
import pandas as pd, numpy as np, scipy.stats as ss
R = pd.read_csv('/Users/gdehol/git/tms_risk/notes/data/cvr2_vs_null_m0-5.tsv', sep='\t')
for m in range(6):                                    # each model vs the null
    print(m, ss.ttest_rel(R[f'm{m}_mean'], R.null_mean))
print(ss.ttest_rel(R.m5_mean, R.m4_mean))             # magnitude vs tuning
cols = [f'm{m}_mean' for m in range(6)]
print(R[cols].idxmax(axis=1).value_counts())          # per-subject winner
print(R[cols].rank(axis=1, ascending=False).mean())   # mean rank
```

---

## 6. Traps — read before running anything

1. **Do not compute this locally.** `/data/ds-tmsrisk/derivatives/encoding_model2.model-{0,1,2}.smoothed.cv/`
   on the Mac are the **November 2025 pre-refit, under-converged** fits, and m3/m4/m5 do
   not exist there at all. `cvr2_vs_null.py` silently uses "whatever is in `.cv` on THIS
   machine" (it says so in its own printed header). Run it on the cluster against
   `/shares/zne.uzh/gdehol/ds-tmsrisk`, or rsync the trees first.
2. **cvR² = 0 is not the null** (see §3). Quote Δ vs −0.01776, not the raw negative
   number, which looks alarming and is not.
3. **m4 is degenerate in the main fits** — with gain pinned and tuning free, exp(mu) runs
   to ~1e12 against payoffs of 5–80 and sd reaches ~15 log units
   (`notes/encoding_model_set_2026-08.md` §3). Its respectable cvR² here has not been
   reconciled with that; do not present m4 as a healthy model on the strength of this
   table alone.
4. **Do not threshold voxels on functional fit for the amplitude contrast.** That is the
   main finding of `notes/encoding_model_set_2026-08.md` §2: baseline fit quality carries
   no information about which voxels show a cTBS effect (Spearman +0.005, p = 0.80), and
   thresholding only costs power. This CV table is a *model*-selection tool, not a
   *voxel*-selection tool.
5. **These are per-subject means over the ROI.** They cannot show whether m5 wins in a
   distinct subset of voxels that m1 loses — that needs a per-voxel win-count (§7).
6. **Mean cvR² is unusable whole-brain** (dominated by near-zero-variance voxels, runs to
   −1e12). Inside an ROI it is fine; whole-brain, use the fraction of voxels > 0.
7. The **`ses1cvr2` option in `monte_carlo_decode.py` is a no-op** for `encoding_model2`:
   that tree fits sessions 2+3 only, so its cvR² has no session axis. A genuinely
   independent (session-1) cvR² exists only in the old
   `encoding_model.cv.denoise.smoothed/sub-XX/ses-1/` tree.

---

## 7. Open items, in priority order

1. ~~Per-voxel win-count across all six models~~ — **DONE 2026-08-19** (SLURM 5106922,
   `cvr2_vs_null.py` extended with selected-voxel means + per-voxel argmax; output
   `notes/data/cvr2_vs_null_m0-5_selected.tsv`). Results:
   - **Union selection ("any model beats the null", 61.4 % of voxels): nothing
     changes.** m1 still beats every model pairwise (p ≤ 0.001), m5 − m4 still null
     (Δ = +0.0001, p = 0.93). Among these signal voxels m1 is the per-voxel winner in
     **35.0 %** (chance 16.7 %; next best m3 at 15.8 %) — so the m1 average is not
     hiding a mixture.
   - **m0-based selection (28 %): m5 ≈ m1 (p = 0.21) and m5 ≫ m4 (p = 6e-11) — but do
     not quote this as evidence for m5**: selecting on the pooled-tuning model m0
     preselects voxels whose tuning was *stable* across sessions, which handicaps m4
     by construction. It is a consistency check, not a fair race.
2. **Same table for `NPC12r`** (`--roi NPC12r`), to check the ordering holds off the
   stimulation site. One-line change to the command in §3.
3. **The model-comparison figure**:
   `python -m tms_risk.modeling.scripts.plot_encoding_model_comparison --roi NPCr2cm-cluster`
   — it was blocked on cvR² for m3/m4/m5 and is now unblocked. Verify it reads the
   `m0-5` TSV, not the superseded one.
4. **Fold §4 into `notes/encoding_model_set_2026-08.md`** — its §4 still says "there is
   no cvR² for m3, m4 or m5 … that is the outstanding job", which is now stale — and add
   a row to `notes/PROVENANCE.md` for `cvr2_vs_null_m0-5.tsv`.
5. **Decide whether any of this reaches the paper.** v9 (`notes/v9_plan.md`) needs no
   refits and reports m1 throughout; the honest use of this table is as a supplementary
   justification that the canonical model is the CV-preferred one, plus the explicit
   statement that CV does not separate tuning from magnitude. Do not upgrade the
   specificity claim on the strength of the m4/m5 comparison — it is a null result at
   p = 0.27.

## 8. One-paragraph version, if that is all that is wanted

Across all six nPRF encoding-model variants, cross-validated on held-out runs within the
stimulation-site ROI and scored against a properly computed training-mean null
(−0.0178), the canonical model — amplitude free per session, tuning and baseline pooled —
gave the best held-out fit (mean cvR² −0.0047, +0.0131 over the null, t(34) = 2.67,
p = 0.011), beat every other variant pairwise (all p ≤ 0.003) and was the best model in
26 of 35 subjects. Held-out fit declined monotonically with the number of per-session
free parameters, and the two variants that dissociate tuning (mu, sd free: −0.0102) from
response magnitude (amplitude, baseline free: −0.0092) were statistically
indistinguishable (Δ = 0.0010, 95 % CI [−0.0009, +0.0027], p = 0.27), so cross-validation
does not by itself adjudicate which of the two the cTBS effect acts on.
