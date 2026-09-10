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


---

# Answers to the writing chat's verification queries (2026-09-10)

Sources: `notes/data/anchor_mechanism.log-power-percpmu.mapjitter.klw.tsv`
(`behavior/scripts/extract_anchor_mechanism.py`),
`notes/data/ppc_anchor/ppc_stats.*.tsv` (`extract_anchor_ppc.py`),
`notes/data/anchor_curves_baseline_shared.tsv` (`extract_anchor_curves.py`),
and the trace attrs read directly.

## 1. Mechanism decomposition (Results ¶81)

| token | value | note |
|---|---|---|
| ⟦XOVER_RS⟧ | **24 CHF** | risky-second; log-interpolated between the 20 and 28 CHF cells |
| ⟦XOVER_RF⟧ | **no crossover within the tested range** | risky-first: decision SD +7.3% still exceeds perceived ratio +6.8% at 28 CHF |
| ⟦DSD_7_RS⟧ / ⟦DRATIO_7_RS⟧ | +13.7% / +3.8% | |
| ⟦DSD_28_RS⟧ / ⟦DRATIO_28_RS⟧ | +7.2% / +9.0% | |
| ⟦DRATIO_7_RF⟧ | +3.0% | risky-first, and POSITIVE — the draft's "starts from a small negative value (−1.8%)" no longer holds |
| ⟦DVAL_RISKY⟧ | +1.0% [−2.0, +4.1] at 7 CHF to +4.0% [−1.5, +9.6] at 28 | |
| ⟦DVAL_SAFE⟧ | −2.2% [−6.4, +1.8] at 7 CHF to −4.5% [−9.2, +0.4] at 28 | |

**Both option-value intervals cover zero in all ten cells**, so the sentence you
wanted is supported. **The draft's ¶81 numbers (17 CHF, 14 CHF, 10.7%, 2.5%,
8.1%, −2.1%, −1.8%) are from the earlier model and must all be replaced.**

## 2. ⟦N_PARAMS_PP⟧ = **12** per participant

Two noise anchors x2 conditions on the perceptual channel (4), two on the
memory channel with no cTBS term (2), risky and safe prior means x2 conditions
(4), and the two prior SDs with no cTBS term (2).

## 3. ⟦PRIORS⟧

> Group-level means were given Normal priors — N(log 0.25, 0.75) on each noise
> anchor, N(log-payoff mean, 1.0) on each prior mean and N(log empirical SD,
> 0.5) on each prior width — with the same scale halved to 0.25 for every
> stimulation contrast. Between-participant SDs were half-Cauchy with scale
> 0.30 (noise), 0.75 (prior means) and 0.40 (prior widths), and 0.30 / 0.15 /
> 0.15 on the corresponding contrasts. Individual parameters used a non-centred
> ("offset") parameterisation throughout.

**⟦TAU_PRIOR⟧ is confirmed unused**: the trace stamps
`tms_risk_slope_priors = sigma_slope=None tau_slope=None` and the label carries
no `.tn`/`.ti` suffix, so the reported fit is at the default prior spec
`v1-2026-08-28`. **Delete ⟦TAU_PRIOR⟧ from the manuscript.**

## 4. ⟦SAMPLER⟧ — the current token is WRONG

Trace stamp: `chains=8 tune=8000 draws=9000 ta=0.95 find_init=mapjitter`.
⟦SAMPLER⟧ currently reads "8 chains, 10 000 tuning, 15 000 draws, target_accept
0.99". **Replace with 8 chains, 8 000 tuning, 9 000 draws, target_accept 0.95,
initialised from a jittered MAP.** LOO is ArviZ `az.compare`, i.e. PSIS-LOO
(Vehtari et al. 2017); Pareto-k is reported per model.

## 5. The three inconsistencies

* **⟦PPC_FRACTION⟧ = 0.46**, not "about a quarter": 0.024 predicted against
  0.053 observed. "About a quarter" was `n1n2`'s number (0.013/0.053) and is
  withdrawn with it. Write "roughly half" or "a factor of two".
* **⟦PPC_N_PASS⟧ / ⟦PPC_N_TOTAL⟧ = 8 / 8.** The 7/8 was stale. There is no
  contradiction with ppp = 0.04: "covered" means the observed value lies inside
  the 95% predictive interval, i.e. ppp between 0.025 and 0.975.
  Your sentence is **correct with one caveat**: among the models evaluated on
  all eight statistics, `percpmu` is the only one at 8/8 (`n1n2`, `percmem`,
  `power+weber-percpmu` and `spl5-n1n2` are each 7/8). Several older fits show
  "7/7" only because their extraction predates the eighth statistic — do not
  count those as ties.
* **⟦ELPD_PERCMEMPMU_DSE⟧ = 1.15** (provisional; final value from the LOO pass).

## 6. Figure 4 in stage coordinates — baseline, n = 73

| token | value |
|---|---|
| ⟦B_PERC⟧ | **+0.294** [+0.224, +0.373] |
| ⟦B_MEM⟧ | **−0.325** [−0.570, −0.083] |
| perceptual fold-change 7 → 112 CHF | **2.26x** [1.86, 2.82] |
| ⟦MEM_RATIO⟧ | **0.41** [0.21, 0.80] — confirmed, and it is the memory channel's 112/7 ratio |

Note these are NOT the draft's b = 0.081 / 0.357: those were the first- and
second-presented options in the position parameterisation. Since
σ(second-presented) = σ_perceptual exactly, the old 0.357 and the new +0.294 are
the same claim about the same channel fitted two ways — but only the stage fit
converges (r̂ 1.000 / ESS 8 388 against r̂ 1.05 / ESS 115), so quote the new
pair. **⟦SPLINE_ORDERS⟧ for Fig. 4d is still pending** — weber, power and affine
are fitted in the stage parameterisation; cspl3/5/7 are running.

## 7. Power-law parameterisation — as you wrote it

Confirmed. The free parameters are the noise SD's **values at 7 and 112 CHF**
(`log_perc_power_sd7`, `log_perc_power_sd112`), interpolated log-linearly in log
payoff, and the stimulation regressor acts on each of those two anchor values.
It is **not** parameterised as (a, b); the exponent is a derived quantity.

## 8. ⟦SFIG_N1N2⟧ — position-indexed robustness

cTBS effect at the lowest well-sampled anchor, group level:

| fit | r̂ | ESS | first-presented | second-presented |
|---|---:|---:|---|---|
| `spl5-n1n2` | 1.000 | 2 822 | +22.5% @13 CHF, P = 0.97 | +27.8% @13, P = 0.93 |
| `spl7-n1n2` | 1.000 | 14 412 | +27.8% @14 CHF, P = 0.98 | +25.3% @14, P = 0.90 |
| `cspl5-n1n2` | 1.000 | 9 621 | +19.0% @13 CHF, P = 0.97 | +19.6% @13, P = 0.90 |

The "same answer" sentence carries a number: **a ~19–28% increase at low
payoffs, on both presented options, with no credible difference between them.**

## 9. ⟦PSD_RISKY_P⟧ / ⟦PSD_SAFE_P⟧ — close, but not from the sweep

`percpsd.pathfinder.klw` gives P(<0) = 0.377 for the risky prior width and
P(>0) = 0.187 for the safe one. The draft's 0.39 / 0.22 are near these but come
from a **pathfinder-initialised fit outside the one-prior sweep**. Either refit
`percpsd` under the sweep's settings or attribute the numbers explicitly.

## 10. Supplementary Table 1 — pending

The single LOO pass is still running. It covers the placement ladder (null, mem,
spmu, perc, percmem, percpmu, percmempmu, each with its `power+weber` twin) and
the noise-form sweep at `percpmu`. The retired rows (⟦ELPD_WEBER⟧, ⟦ELPD_N1⟧,
⟦ELPD_N2⟧, ⟦ELPD_UNRESOLVED⟧) are not in it.

## 11. Choice rule — confirmed, with one wording caution

`bauer/core.py:169-197` and `models/risky_choice.py:596-607`: the **noise is
indexed by presentation POSITION** (`n1_evidence_sd = perceptual + memory`,
`n2_evidence_sd = perceptual`) and the **prior by option TYPE**
(`n1_prior_mu = where(risky_first, risky_prior_mu, safe_prior_mu)`, and the
mirror for n2). The decision variable is the difference of the two noisy
posterior means, normalised by √((w₁ν₁)² + (w₂ν₂)²) with w = σ_p²/(σ_p² + ν²)
(`posterior_mean_sd`), evaluated through the cumulative normal.

So your equation is right, but ν_x and ν_c must be read as "the noise of
whichever POSITION that option occupied on that trial", not as properties of the
risky and safe options. Add half a sentence saying so, or the equation implies
noise is a property of option type, which is the one thing the model does not
assume.
