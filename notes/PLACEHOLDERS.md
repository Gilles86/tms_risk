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

---

# Round 3 — 2026-09-10 (answers to the 66-token inventory)

## Block C — the placement-ladder LOO pass: **DONE, all FINAL**

Paired against `log-power-percpmu.mapjitter.klw`, ArviZ PSIS-LOO, pointwise,
`notes/data/loo_anchor/`. Regenerate any time with
`python -m tms_risk.behavior.scripts.make_supp_table1`.

| Token | Value |
|---|---|
| ⟦ELPD_NULL⟧ | −111.4 |
| ⟦ELPD_NULL_DSE⟧ | 13.7 |
| ⟦SE_NULL⟧ | 8.1 |
| ⟦ELPD_MEM⟧ | −46.9 |
| ⟦ELPD_MEM_DSE⟧ | 10.1 |
| ⟦SE_MEM⟧ | 4.6 |
| ⟦ELPD_SPMU⟧ | −25.7 |
| ⟦ELPD_SPMU_DSE⟧ | 6.7 |
| ⟦SE_SPMU⟧ | 3.8 |
| ⟦ELPD_PERCMEM⟧ | −5.8 |
| ⟦SE_PERCMEM⟧ | 1.4 |
| ⟦ELPD_PERCMEMPMU⟧ | +0.0 |
| ⟦ELPD_PERCMEMPMU_DSE⟧ | 1.1 |
| ⟦ELPD_PW⟧ | −1.6 |
| ⟦ELPD_PW_DSE⟧ | 2.7 |

Marginal ELPDs, if the table wants them: percpmu −4148.1 (SE 46.8),
percmempmu −4148.1 (46.8), percmem −4153.9 (46.7), perc −4155.2 (46.7),
spmu −4173.8 (47.0), mem −4195.0 (46.7), null −4259.5 (46.3).

**Supplementary Table 1 is regenerated** at `notes/supp_table1.md` — 20 rows,
every one passing r̂ ≤ 1.01 and ESS ≥ 400.

## Block G — two prose claims

**1. "LOO computed with ArviZ PSIS-LOO."** ✅ TRUE. `extract_anchor_loo.py:25`
calls `az.loo(idata, pointwise=True)`, which is PSIS-LOO (Vehtari, Gelman &
Gabry 2017). Keep the sentence.

**2. "percpmu is the only model at 8/8."** ❌ **FALSE — delete or rewrite this.**
Now that all 20 models have a PPC extraction on the same eight statistics,
**eleven** reach 8/8: percpmu, percmempmu, `power+weber-percmempmu`, spl3, spl4,
spl5, spl6, spl7, cspl3, cspl5, cspl7 — **and `spmu`, the priors-only model with
no noise change at all.**

This matters more than a wording fix. The eight targeted statistics do **not**
separate the mechanism; the priors-only model passes all of them. What separates
it is:

- **ELPD**: spmu is 25.7 worse, 3.8 SE. This is the number that rules out a
  pure prior shift, and it should carry that claim in the text.
- **the psychometric-SLOPE view of the design grid**: spmu covers 5/6 slope
  cells against percpmu's 6/6. It is the only PPC view that is diagnostic,
  which is exactly as expected — a prior shift moves the indifference point,
  a noise change flattens the slope.

Design-grid coverage (28+ cells the design fixes, `notes/data/ppc_grid_placement.tsv`):
percpmu 32/34, percmempmu 32/34, spmu 31/34, percmem 30/34, perc 29/34,
mem 28/34, null 27/34.

**Consequence for ¶80 and the Discussion**: do not write that the PPCs single
out the reported model. Write that ELPD rules out the priors-only and
noise-only accounts, and that the PPCs confirm the reported model reproduces
every targeted statistic. That is what the numbers support.

## Block E — the four with no value

**⟦SPLINE_ORDERS⟧.** The stage (perceptual/memory) parameterisation was fitted
at these anchor counts. Note the two cohorts differ, and Fig. 4d is BASELINE:

- **cTBS cohort** (n = 35, `cogmodels.anchor`, `*-percpmu`): 2 (power), 3, 4, 5,
  6, 7 piecewise-linear; 3, 5, 7 smooth. Weber (1), smooth 4 and smooth 6 are
  fitting now (job 5715656).
- **Baseline** (n = 73, `cogmodels.baseline`, `*-null`) — **this is the set
  Fig. 4d draws from**: 1 (Weber), 2 (power), 5 piecewise-linear; smooth 3 and 5
  present, smooth 7 and affine fitting (5714965); smooth 4/6 and linear 3/4/6/7
  fitting (5715644).

So the current draft's "2, 3, 4, 5 and 7 anchors" describes neither set exactly.
Hold this token until the two jobs land — I will give you one sentence per
cohort.

**⟦SFIG_N1N2⟧.** Recommend **Supplementary Figure 3**, since Supp. Fig. 2 is
already the ELPD ladder. Final numbering is yours once the supplement is
assembled; what matters is that all three uses point at the same figure. Its
content is settled (§8 above: spl5/spl7/cspl5-n1n2, ~19–28% at low payoffs on
both presented options).

**⟦PSD_RISKY_P⟧ / ⟦PSD_SAFE_P⟧.** Refit submitted under the sweep's own
settings (mapjitter, 8 chains, 5000 tune / 9000 draws — job 5715682), so the
"pathfinder-initialised, outside the sweep" caveat can be dropped. Until it
lands the honest form is the explicit attribution: 0.377 / 0.187 from
`percpsd.pathfinder.klw`. **Do not quote the draft's 0.39 / 0.22** — those are
from neither fit.

**⟦BB_INTERVAL⟧.** Still blocked; the hierarchical probit has not converged.

## Block F — three prose numbers that can still move

All three are baseline-fit quantities, and the baseline flexibility set is
being refitted right now (5714965, 5715644), so tokenise them:

- "2.25-fold [1.86, 2.82]" → ⟦PERC_RATIO⟧ ⟦PERC_RATIO_CRI⟧ (¶74, Fig. 4c caption)
- "roughly 19 to 28%" → ⟦N1N2_RANGE⟧ (¶80). Current value stands: 19–28%.
- "about a factor of two" / "by about half" → derive from ⟦PPC_FRACTION⟧ once
  the final percpmu PPC is in; do not hand-round independently in two places.

## A correction to what I told you on 2026-09-09

I previously said the PPCs carry the prior-shift claim and ELPD does not. Half
right. ELPD does not separate percpmu from percmem (5.8, 1.4 SE) — that part
stands. But the PPCs do not separate them decisively either (32/34 vs 30/34),
and the priors-only model passes all eight targeted statistics. The prior shift
rests on the ELPD tie with percmempmu, the design-grid edge, and the direct
posterior on `safe_prior_mu` — not on the targeted PPCs.

---

# Round 4 — 2026-09-10

## ⟦BB_INTERVAL⟧ — resolved, not dropped

The blocked thing was a hierarchical *probit*, which is not what this token
needs. `anchor_brain_behavior_posterior.py` correlates the nPRF amplitude
change with the per-participant noise change **per posterior draw**, so it
gives both the point estimate from posterior means and a genuine credible
interval, with no probit anywhere. Run on the reported model
(`notes/data/bb_posterior.log-power-percpmu.mapjitter.klw.tsv`):

| parameter | r (posterior means) | r [95% CrI] | P(r < 0) |
|---|---:|---|---:|
| perceptual noise @ 112 CHF | **−0.266** | −0.195 [−0.405, +0.027] | 0.957 |
| perceptual noise @ 7 CHF | +0.296 | +0.192 [−0.084, +0.442] | 0.083 |
| safe prior μ | +0.173 | +0.039 [−0.294, +0.361] | 0.410 |
| risky prior μ | −0.072 | −0.046 [−0.308, +0.229] | 0.630 |

Report the **posterior-mean r** as the headline (that is the quantity the
scatter plot shows) and the CrI beside it. Spearman agrees with Pearson to
within 0.02 throughout. Note the sign: the amplitude change tracks the noise
change at the HIGH payoff anchor, and the shrinkage between the two columns is
partial pooling doing its job.

## The `8/8` column was PPCs, and it does not discriminate

`8/8` meant the eight targeted posterior predictive statistics. Eleven of
twenty models scored 8/8, so the column is necessary and not diagnostic.

Following the "use more PPCs" fix, Supplementary Table 1 now carries a second
column, `Grid` — the 34 cells the DESIGN fixes (5 safe payoffs x 2 orders x 2
arms, across four views: P(risky) vs safe payoff, vs risky/safe ratio, vs
stake, and the psychometric SLOPE vs stake). Four times as many cells, none of
them chosen post hoc, and they do order the placement ladder:

    percpmu 32/34 · percmempmu 32 · spmu 31 · percmem 30 · perc 29 · mem 28 · null 27

**But the priors-only model still reaches 31/34.** No posterior-predictive
criterion at any resolution separates it from the reported model; **ELPD does,
at 25.7 nats and 3.8 SE.** So the paper's claim that the noise change is
necessary must rest on ELPD, with the PPCs supporting rather than carrying it.

---

# Round 5 — 2026-09-10: the supplementary figure set

Four figures, each tied to one claim the main text makes. Files are named by
their proposed number, so the number is unambiguous everywhere.

| # | File (`notes/figures/`) | What it settles | Cited at |
|---|---|---|---|
| **S1** | `SUPP_S1_ppc_design_grid.pdf` | **Which** posterior predictive checks, and how the alternatives do on them | ¶80, ¶82, Methods |
| **S2** | `SUPP_S2_model_comparison.pdf` | Where the cTBS effect acts — the placement ladder | ¶80, Supp. Table 1 |
| **S3** | `SUPP_S3_noise_flexibility.pdf` | Whether the power law's shape is the data's or the form's | ¶74, Fig. 4 caption |
| **S4** | ⟦SFIG_N1N2⟧, not yet built | Position-indexed (n1/n2) robustness fits | ¶80, Methods ×2 |

The draft already cites "Fig. S2" for the ELPD ladder, so S2 keeps that slot
and nothing has to be renumbered. ⟦SFIG_N1N2⟧ = **S4**.

`SUPP_S3b_noise_flexibility_ctbs.pdf` is the same three panels on the cTBS
cohort. Keep it as a reviewer-response figure rather than a numbered
supplement — it answers a different question (how flexible must the noise
function be when a stimulation effect is also being fitted) and having both
numbered invites exactly the cohort confusion S3's caption exists to prevent.

## Replacing the "only model at 8/8" sentence

The eight targeted statistics are still fine to report; they are just not
diagnostic, and the text must not claim they single out anything. Suggested
substance for ¶80 (wording yours):

> The reported model reproduces all eight targeted posterior predictive
> statistics, but so do ten of the nineteen alternatives, including one in
> which cTBS changes only the magnitude priors — so we assessed fit against
> the design's own cells instead (Fig. S1): the 34 IPS − vertex contrasts
> defined by five safe payoffs × two presentation orders × two stimulation
> arms, read four ways, each computed per posterior draw. The reported model
> covers 32 of 34. Coverage falls monotonically as the mechanism is removed —
> 30 without the prior shift, 29 with perceptual noise alone, 27 with no cTBS
> effect — but the priors-only model still reaches 31, so **no
> posterior-predictive criterion separates it; the model comparison does**
> (ΔELPD = 25.7, dSE 6.7).

## S3 changes the flexibility claim — read this before writing ¶74

With the full baseline ladder fitted (1 to 7 anchors, both bases, n = 73),
paired ΔELPD against the power law:

| anchors | piecewise linear | smooth |
|---:|---:|---:|
| 1 (Weber) | −47.4 (11.2) | — |
| 2 (power) | reference | reference |
| 3 | **+19.5 (9.5)** | **+16.9 (10.2)** |
| 4 | +16.2 (11.6) | +6.6 (11.5) |
| 5 | −3.2 (12.2) | −14.8 (13.0) |
| 6 | −20.6 (13.3) | −16.6 (14.5) |
| 7 | — | −27.3 (14.9) |

**On the baseline data, three anchors beats two by about 2 SE.** The claim
"two anchors is not significantly worse than three" is true on the cTBS cohort
(+5.0, dSE 5.6 — 0.9 SE) but NOT on the baseline. Do not write it unqualified.

What is safe to claim, and what S3's panels b and c show, is that the SHAPE is
not an artefact of the form: every fit from 2 to 7 anchors traces the same
monotone rise in perceptual noise (0.17 to 0.35 over the payoff range) and the
same fall in memory noise. The 3-anchor gain buys curvature in the memory
channel around 14-20 CHF, not a different perceptual story. And Weber is
rejected outright at 47 nats / 4.2 SE, which is the claim Figure 4c actually
needs.

---

# Round 6 — 2026-09-10: levels vs contrast, and the final supplementary set

## Ranking models by the LEVELS would mislead — say so, don't hide it

Supplementary Fig. S3's panels a-h plot choice PROPORTIONS (levels); its panel
i counts the paired IPS − vertex CONTRAST. Different denominators, and they
rank the model set differently:

| model | contrast (of 34) | levels (of 68) |
|---|---:|---:|
| Perceptual + memory noise, prior means | 32 | 57 |
| **Perceptual noise + prior means** (reported) | **32** | 57 |
| Perceptual + memory noise | 30 | 57 |
| Perceptual noise only | 29 | 55 |
| Prior means only, no noise change | 31 | **58** |
| Memory noise only | 28 | 55 |
| No cTBS effect | 27 | 53 |

The levels span 53-58 across the entire ladder, and the **priors-only model
comes first** — ahead of the reported one. Even "no cTBS effect at all"
reaches 53/68. The levels are dominated by the psychometric function itself,
which every model in the family fits; the stimulation effect is a small
perturbation on top and is swamped. The contrast removes the part every model
gets right and orders the ladder monotonically, 32 down to 27.

Both are now drawn in S3 panel i, titled *"The contrast separates the models;
the levels do not."* Reporting only the contrast would invite exactly the
question this pre-empts.

**A limitation to state, not to bury**: the reported model covers 57 of 68
levels but 32 of 34 contrasts, and the weakest view is the risky/safe ratio at
18/24. The model fits the DIFFERENCE better than the levels. That is expected
in direction — the contrast cancels subject-level heterogeneity a group-level
band does not model — but a reviewer will find it, so a sentence conceding it
is cheaper than being caught.

## Final supplementary figure set

| # | File in `notes/figures/` | Content |
|---|---|---|
| S1 | *(the draft's existing S1 — unchanged)* | |
| **S2** | `SUPP_S2_model_comparison.pdf` | **a** placement ladder · **b** the effect estimated under every noise form · **c** the perceptual noise functions themselves, IPS vs vertex, with ΔELPD per form and a directional significance rug |
| **S3** | `SUPP_S3_ppc_design_grid.pdf` | **a-h** choice proportions against the design's cells, IPS red vs vertex green, misses ringed · **i** coverage on the contrast and on the levels |
| **S4** | `SUPP_S4_noise_flexibility.pdf` | Baseline (n = 73): what extra anchors buy, and the noise functions from 1 to 7 anchors inside the power law's own credible band |
| S5 | ⟦SFIG_N1N2⟧, still to build | Position-indexed (n1/n2) robustness fits |

Not numbered, keep as a reviewer-response figure:
`SUPP_Sx_noise_flexibility_ctbs.pdf` — S4's panels on the cTBS cohort.

## The rug in S2c is DIRECTIONAL — the text must match

It marks P(IPS > vertex) > 0.95. At the reported model's 7 CHF anchor that is
**0.963**, so the two-sided 95% interval marginally includes zero:
**+0.029 log units [−0.003, +0.061]**. The draft's ⟦DNU7⟧ ("0.029, [0.002,
0.057], p = 0.020") is not what the current fit gives — same point estimate,
but the interval crosses zero. Write it as a directional posterior probability,
not as a two-sided credible interval.
