# The encoding-model set, and why voxel selection is the thing that matters

Written 2026-08-07. Supersedes nothing; read alongside `notes/encoding_model_choice.md`
(m1 vs m2, and the cvR² convergence problem) and
`notes/amplitude_effect_voxel_selection.md` (the first pass at the selection issue).

Fits: `derivatives/encoding_model2.refit2026.model-{0..5}.smoothed`, all 35 subjects,
main fit on all of sessions 2+3 with no folds held out. Deliberately kept out of the
legacy `encoding_model2.model-{0,1,2}.smoothed` tree, which every published analysis
reads.

## 1. The model set

Each voxel gets `amplitude · exp(−½((log n − mu)/sd)²) + baseline`. Models differ only
in which parameters may differ between the two TMS sessions — and "per session" *is* the
IPS-vs-vertex contrast, since sessions 2 and 3 are the two arms.

| | amplitude | mu | sd | baseline | what it asks |
|---|---|---|---|---|---|
| m0 | — | — | — | — | nothing changes |
| **m1** | ✓ | — | — | — | pure gain change (the paper's canonical model) |
| m3 | ✓ | — | ✓ | — | mixes gain and shape; superseded by m4/m5 |
| **m4** | — | ✓ | ✓ | — | **tuning**: what the population is tuned to |
| **m5** | ✓ | — | — | ✓ | **response magnitude**: how strongly it responds |
| m2 | ✓ | ✓ | ✓ | ✓ | everything |

m4 vs m5 is the contrast that tests the paper's specificity claim — "a specific effect
of TMS on the amplitude of the nPRF but not their preferred numerosity tuning" —
directly, rather than assuming it. m1 cannot test it: it pins tuning by construction.
`sd` belongs with `mu`, not with `amplitude`: it is the WIDTH of the tuning function,
not a noise term.

**All six specs verified.** `extract_prf_params_by_condition.py` prints which parameters
come out bit-identical across arms; each matches its pooled set exactly (m0 all four,
m1 mu/sd/baseline, m2 none, m3 mu/baseline, m4 amplitude/baseline, m5 mu/sd). A wrong
grid ordering would have produced plausible-looking but meaningless fits, so this check
is not optional.

## 2. THE MAIN FINDING: do not threshold on functional fit

Amplitude contrast (IPS − vertex), NPCr2cm-cluster, paired across subjects:

| selection | n voxels | m1 mean | m1 median | m5 mean | m5 median |
|---|---|---|---|---|---|
| **none (anatomical ROI)** | 11022 | **0.0084** | **0.0218** | **0.0038** | **0.0102** |
| r² > 0 | 11021 | **0.0084** | **0.0218** | **0.0038** | **0.0101** |
| r² > 0.05 | 3775 | 0.035 | 0.143 | 0.031 | 0.055 |
| r² > 0.10 | 1356 | 0.207 | 0.471 | 0.099 | 0.095 |

Unthresholded the effect is significant in **every** cell — both models, both
aggregations. Thresholding degrades it monotonically until it is gone.

**The independent-selection check.** Session 1 is the pre-TMS baseline, so a threshold on
its cvR² is the only one genuinely independent of the contrast (a threshold on pooled
sessions 2+3 cvR² is computed from the very data the contrast is taken on). A real
per-session cvR² exists only in the OLD tree,
`encoding_model.cv.denoise.smoothed/sub-XX/ses-1/` — for `encoding_model2` the
`ses1cvr2` option in `monte_carlo_decode.py` is a **no-op**, because that tree fits
sessions 2+3 only and its cvR² has no session axis.

| selection | n voxels | m1 mean | m1 median | m5 mean | m5 median |
|---|---|---|---|---|---|
| none | 10805 | 0.0143 | 0.0297 | 0.0058 | 0.0115 |
| session-1 cvR² > 0 | 1652 | 0.087 | 0.119 | 0.060 | **0.0198** |
| session-1 cvR² > 0.02 | 1152 | 0.101 | 0.202 | 0.584 | 0.435 |
| session-1 cvR² > 0.05 | 690 | 0.235 | 0.665 | 0.232 | 0.163 |

**The effect SIZE is stable while only significance decays** — m1 runs −0.139 → −0.143 →
−0.161 across those rows. That is the signature of losing power by discarding 85% of the
voxels, not of removing noise. Selection is not sharpening anything.

**Why selection cannot help here**, three independent ways:

1. Per-subject Spearman(session-1 cvR², |Δamplitude|) = **+0.005, p = 0.80.** Baseline
   fit quality carries no information about which voxels show a cTBS effect.
2. Session-1 and session-2/3 cvR² correlate at **r = +0.011.** Cross-validated fit is
   not a stable property of a voxel here, so any functional threshold is close to a
   random 15% subsample.
3. Pooled cvR² correlates with |Δamplitude| at **+0.22 (p = 1e-8)** — if anything the
   better-fitting voxels carry *bigger* effects, the opposite of the usual assumption.

**Recommendation: report the anatomically defined ROI with no functional selection.**
It is the more conservative analysis, not the more permissive one, and it is robust to
model and aggregation. Cite the session-1 analysis as the robustness check, precisely
because it is the selection that cannot be accused of peeking at the contrast.

Caveats to keep attached: the one cell surviving selection is m5/median (p = 0.0198), so
it is noisy rather than uniformly destroyed; and the higher thresholds drop subjects
entirely (n = 30, then 20), making those rows a different comparison rather than a
stricter one.

## 3. What the parameters do, per model

Signal voxels are NOT used here (see §2); per-subject median over the whole ROI, paired.

| model | free | result |
|---|---|---|
| m1 | amplitude | amplitude **−0.121, p = 0.030** |
| **m5** | amplitude + baseline | amplitude **−0.087, p = 0.012**; baseline flat |
| m3 | amplitude + sd | amplitude negative; sd flat |
| **m4** | mu + sd | **nothing** — and degenerate, see below |
| m2 | all four | amplitude negative; mu, sd flat |

**Amplitude moves negative in every model where it is free. Tuning never moves in any
model that frees it** (mu and sd give p = 0.51–0.93 throughout). That is the specificity
claim tested rather than assumed.

**m4 is degenerate.** With gain pinned and tuning free, exp(mu) runs to 1e12 against
payoffs of 5–80 and sd reaches 15 log units. Tuning is not identifiable without gain free
to absorb amplitude differences. The m4-vs-m5 comparison is therefore lopsided — but the
degeneracy is itself informative: these data cannot be explained by moving tuning.

Note also the **grid-floor problem**: the preferred-numerosity distribution is bimodal
with a large spike near 0.2, i.e. voxels whose mu collapsed below the fitting grid's
lower bound of 5 (`mus = log(linspace(5, 80, 50))`), while gradient descent afterwards
runs unbounded. About 25% of voxels sit there.

## 4. Is m5 the better model? Not established yet

The parameter-level evidence favours it: the amplitude effect is largest and most robust
under m5, its baseline is flat so the second free parameter is not diluting anything, and
it is well-conditioned where m4 is not. Freeing baseline plausibly *helps* because when
baseline is pooled some of the gain change is absorbed into the shared offset — and we
know amplitude and baseline trade off (within-subject r = −0.58 in m2).

**But "works better" is a claim about out-of-sample fit, and there is no cvR² for m3,
m4 or m5.** Only m0/m1/m2 have been cross-validated. Until m4 and m5 are cross-validated
the model comparison cannot be drawn, and `plot_encoding_model_comparison.py` cannot
rank the set. That is the outstanding job.

Read the cvR² numbers against the **null of −0.0184** (`null_cvr2.py`): cvR² = 0 is not
the null, because `get_rsq` puts the held-out fold's own mean in the denominator.

## Scripts

| Output | Script |
|---|---|
| the fits | `modeling/fit_regression_nprf.py` + `slurm_jobs/submit_regression_nprf.sh <label> '' .refit2026` |
| per-voxel parameters by condition | `modeling/scripts/extract_prf_params_by_condition.py` → `notes/data/prf_params_by_condition.tsv` |
| parameter figure, per model | `modeling/scripts/plot_prf_parameters_by_condition.py --model_label 5` |
| model comparison figure | `modeling/scripts/plot_encoding_model_comparison.py` (needs cvR² for all labels) |
| the null | `modeling/scripts/null_cvr2.py`, `modeling/scripts/cvr2_vs_null.py` |
