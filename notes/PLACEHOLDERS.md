# Placeholder tokens for v12

Write the prose now, with these tokens where a number goes. Every token has a
current best estimate so the sentence can be judged for sense, and a note on
what will change it. When the final sweep lands I substitute them mechanically,
so **do not paraphrase a token or split it across a line** — keep the ⟦…⟧ intact.

Tokens marked **FINAL** will not change. Tokens marked *pending* move with the
τ_noise decision and the one-prior sweep, but only in the second decimal or so;
none of them changes a sign or a conclusion.

## Results — the model paragraph

| token | current | status |
|---|---|---|
| ⟦MODEL_NAME⟧ | power-law PMCM, stage-indexed, cTBS on perceptual noise + the prior means (`log-power-percpmu`) | **FINAL** |
| ⟦RHAT⟧ / ⟦ESS⟧ | 1.000 / 11016 | **FINAL** — clears the gate with room |
| ⟦DNU7⟧ | +18.3% (0.029 log units) | pending — final sweep |
| ⟦DNU7_CRI⟧ | [+0.003, +0.061] log units | pending |
| ⟦DNU7_P⟧ | 0.96 | pending |
| ⟦DNU_HIGH⟧ | +0.1% at 112 CHF, not credible | pending |
| ⟦CRED_UPPER⟧ | ~18 CHF | pending |
| ⟦SAFE_PMU⟧ / ⟦SAFE_PMU_P⟧ | −14% (10.7 → 9.2 CHF) / 0.944 | pending |
| ⟦RISKY_PMU⟧ | +4%, p 0.34 — null | pending |
| ⟦MEM_RATIO⟧ | 0.41 (baseline memory noise, 112 vs 7 CHF) | **FINAL** |

## Results — model comparison

Every ⟦ELPD_*⟧ comes from the single LOO pass over the placement ladder; all are
PAIRED against ⟦MODEL_NAME⟧ with the dSE of the paired difference.

| token | current | status |
|---|---|---|
| ⟦ELPD_NULL⟧ / ⟦ELPD_NULL_DSE⟧ / ⟦SE_NULL⟧ | 111.4 / 13.7 / 8.1 | pending |
| ⟦ELPD_MEM⟧ / ⟦ELPD_MEM_DSE⟧ / ⟦SE_MEM⟧ | 46.9 / 9.9 / 4.7 | pending |
| ⟦ELPD_SPMU⟧ / ⟦ELPD_SPMU_DSE⟧ / ⟦SE_SPMU⟧ | 25.7 / 6.8 / 3.8 | pending — **the rung that makes the noise effect necessary** |
| ⟦ELPD_PERCMEM⟧ / ⟦SE_PERCMEM⟧ | 5.8 / 1.5 | pending — do NOT read as establishing the prior shift |
| ⟦ELPD_PERCMEMPMU⟧ | 0.01 (dSE 1.15) — an exact tie | pending |
| ⟦ELPD_PW⟧ / ⟦ELPD_PW_DSE⟧ | 1.6 / 2.9 | pending — Weber memory, 14 params vs 16 |
| ⟦GRID_COV⟧ | 21 | pending — of 22 design-grid cells |

**Retired tokens** (the claims they served are withdrawn): ⟦ELPD_WEBER⟧,
⟦ELPD_N1⟧, ⟦ELPD_N2⟧, ⟦ELPD_UNRESOLVED⟧, ⟦ASYM_FRACTION⟧, ⟦DNU_FIRST_P1⟧,
⟦DNU_FIRST_P2⟧, ⟦DNU_SECOND_HIGH_P⟧, ⟦N_INCREASE⟧.

## Results — posterior predictive checks

| token | current | status |
|---|---|---|
| ⟦PPC_N_PASS⟧ / ⟦PPC_N_TOTAL⟧ | 7 / 8 | pending |
| ⟦PPC_FIRST⟧ | +0.006, [−0.037, +0.025], p = 0.23 | pending |
| ⟦PPC_ORDER⟧ | +0.046, [−0.020, +0.060], p = 0.09 | pending |
| ⟦PPC_STAKE_SECOND⟧ | −0.023, [−0.075, +0.070], p = 0.69 | pending |
| ⟦PPC_THREEWAY⟧ | −0.028, [−0.131, +0.076], p = 0.51 | pending |
| ⟦PPC_SLOPE_SECOND⟧ | −0.103, [−0.108, +0.108], p = 0.97 | pending — barely covered; do not lean on it |
| ⟦PPC_SLOPE_ORDER⟧ | −0.094, [−0.158, +0.130], p = 0.84 | pending |
| ⟦PPC_HIGH_STAKE⟧ | +0.037, [−0.036, +0.071], p = 0.20 | pending |
| ⟦PPC_OBS_SECOND⟧ | +0.053 | **FINAL** — observed cTBS effect on risky-second choice proportions |
| ⟦PPC_FAIL⟧ | +0.024 predicted, ppp 0.04 | pending |
| ⟦PPC_FAIL_NOISEONLY⟧ | +0.006 (perc) / +0.011 (percmem), ppp < 0.01 | pending |
| ⟦PPC_FRACTION⟧ | about a quarter | pending |
| ⟦PPC_CELLS⟧ / ⟦PPC_R⟧ / ⟦PPC_COVERAGE⟧ | 420 / 0.93 / 97% | pending |
| ⟦ASYM_OBS⟧ / ⟦ASYM_MODEL⟧ | +4.7 / +1.9 percentage points | pending |
| ⟦ASYM_FRACTION⟧ | 40% (33% under τ_noise 0.10) | pending |

## Results — brain and behaviour

| token | current | status |
|---|---|---|
| ⟦BB_R⟧ / ⟦BB_INTERVAL⟧ / ⟦BB_RHO⟧ | 0.53 / **unknown** / 0.59 | **BLOCKED** — the hierarchical version is still being fitted; the first attempt did not converge (r̂ 2.7). Leave the whole clause tokenised and do not write a number. |
| ⟦BB_FIRST⟧ | −0.04 | FINAL |
| ⟦BB_DIFF_P⟧ | 0.008 | FINAL |
| ⟦BB_CEILING_PERC⟧ / ⟦BB_CEILING_MEM⟧ | 0.71 / 0.36 | FINAL |

## Methods

| token | current | status |
|---|---|---|
| ⟦N_BASELINE⟧ / ⟦N_BASELINE_CLEAN⟧ | 75 / 73 | **FINAL** |
| ⟦N_TMS_SELECTED⟧ / ⟦N_TMS⟧ | 37 / 35 | **FINAL** — v11 said 35 selected, which is wrong |
| ⟦EXCL_P1⟧ / ⟦EXCL_P2⟧ | 11.7% / 13.7% | **FINAL** |
| ⟦EXCL_Z1⟧ / ⟦EXCL_Z2⟧ / ⟦EXCL_NEXT⟧ | −2.79 / −2.64 / 31.4% (z = −1.40) | **FINAL** |
| ⟦PAYOFF_RISKY⟧ / ⟦PAYOFF_SAFE⟧ | 36.15 / 15.82 CHF | **FINAL** |
| ⟦N_TRIALS⟧ | 8335 | **FINAL** |
| ⟦SPLINE_ORDERS⟧ | 2, 3, 4, 5 and 7 anchors | pending — v11's "3 to 9" is wrong either way |
| ⟦SAMPLER⟧ | 8 chains, 10 000 tuning, 15 000 draws, target_accept 0.99 | pending |
| ⟦TAU_PRIOR⟧ | the half-normal scale on between-participant SDs | pending — needed only if a non-default τ is reported |
| ⟦CONV_GATE⟧ | r̂ ≤ 1.01 and ESS ≥ 400 on all group-level parameters | **FINAL** |

## Sentences that are FINAL and can be written outright

* The consistent choice rule and its equation (§1 of the handoff).
* The participants and exclusion paragraphs (§7) — copy verbatim.
* The reliability paragraph (§5).
* The deletion of the model-parameter brain–behaviour correlation (§4).
* The two limitation sentences: the ~7% flat psychometric function, and that
  the model does not generate the order dependence (§10, §"No model in the
  family…").
* The prior-shift justification (§9).
