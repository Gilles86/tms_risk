# What changed in the Results since v10

Context for a fresh chat. Paper: *Risk Attitudes Causally Rely on Parietal
Magnitude Representations* (de Hollander, Moisa & Ruff). Draft = `TMS_paper_v10.docx`.

## Unchanged

The whole neural half: experimental approach, nPRF amplitude reduction, the
six-variant cross-validated encoding-model comparison, trial-wise decoding.
Figures 1 and 2 stand. So do the psychophysics numbers (probit slopes and
risk-neutral probabilities, order-specific cTBS effect).

## 1. New section: Weber's law fails

The single biggest addition. v10 motivated magnitude-dependent noise from the
nPRF preferred-numerosity IQR ([6,10] vs presented [13,30]) plus a post-hoc
median split on stake. That is now replaced by a **model-free result in the
baseline session, all 73 participants, before any stimulation**:

- choice consistency drops **−37%** from low to high stakes when the risky
  option is presented second, **−22%** when it is first (both p < 0.001);
- replicates in the TMS and non-TMS subgroups separately;
- fitted noise on the second-presented option grows **2.69×** across the payoff
  range, the first-presented option only **1.25×**.

This turns flexible noise from a *fix for a misfit* into a *finding*. New
Figure 4.

## 2. The cognitive model changed family

**Old (v10):** 5-knot B-spline "Flexible PMC", 16 variants, cTBS effect
attributed to **shared perceptual noise** (vs working-memory noise).

**New:** an *anchor* parameterisation where the free parameters *are* the noise
SDs at named payoffs, with a **2-parameter power law** as the reported form
(splines retained as a robustness check). Variants are indexed by **which
presented option carries the cTBS effect**, not by perceptual vs memory.

Consequences for the text:

- *"the cTBS effect on shared perceptual noise alone"* — **gone**. The effect
  localises to the **second-presented option**.
- The whole perceptual-vs-memory decomposition (2.0 / 59.9 / 64.3 / 69.9 nats)
  — gone.
- *"models confined to a single presentation position fit substantially worse"*
  — **reversed**; single-position models are now among the best.
- 16 variants → **155 fitted, 114 of which pass the convergence gate**
  (r̂ ≤ 1.01, ESS ≥ 400). Supplementary Table 1 must be regenerated from
  converged models only.

## 3. The mechanism paragraph is wrong as written

- *"cTBS added a roughly constant absolute amount of noise (about 0.2 CHF)"* —
  **wrong**. In natural space fitted noise grows **supra**-linearly
  (ν ∝ x^1.28 for the second-presented option) and the cTBS effect is directly
  magnitude-specific.
- *"the first-presented option already carries working-memory noise, so added
  noise compresses it most strongly"* — that was the old mechanism.
- "Leverage" and the decision-space heatmaps are retired with old Figure 5.

Replacement: the two channels of the decision variable (perceived risky/safe
ratio vs decision SD) plotted against safe payoff, per order. They **cross at
≈17 CHF** when the risky option is second — decision noise dominates at small
payoffs (**+10.7%** at 7 CHF vs +2.5% for the ratio), the perceived-ratio shift
dominates at large ones (**+8.1%** vs −2.1% at 28 CHF).

## 4. New paragraph to add: what the model does not explain

Every candidate under-produces the observed effect. Predicted ΔP(risky) with the
risky option second is **0.017–0.030 against an observed 0.053**, and all models
get the slope change too small (predicted ≈ −0.02 vs observed −0.10; posterior
predictive p 0.93–0.99). Worth stating outright rather than leaving for review.

## 5. Open decision that gates §2 and §3

Which model to report. Models with a cTBS effect on **prior width** fit much
better but weaken the mechanism claim:

| Model | ELPD rank | Δν₂ credible? | Reproduces ΔP? |
|---|---|---|---|
| `log-power-n2` | 75 | yes (p = 0.002) | **no** (ppp 0.005, slope sign wrong) |
| `log-power-n1n2` | 27 | yes (p = 0.020) | weak (ppp 0.010, slope sign wrong) |
| `log-power-n2psd` | 4 | yes (p = 0.022) | ok (ppp 0.050, sign right) |
| `log-power-n1n2psd` | 1 | **no** (p = 0.06) | best (ppp 0.092) |

`n1n2psd` beats `n1n2` by **30.3 ± 6.9** ELPD. Current lean: `n2psd` — near-top
fit, credible noise effect, and its prior-width term does little
(Δσ_safe = −0.05, p = 0.12). Caveat: `n2` and `n2psd` do not yet converge
(funnel in the group-level SD hyperparameters); regularised refits are running.

## Figures

| # | Status |
|---|---|
| 1, 2 | unchanged |
| 3 | rebuilt (`fig3_probit.pdf`) — cell-mean parameters, narrower |
| **4** | **new** (`fig4_weber.pdf`) — Weber fails, baseline n = 73 |
| 5 | rebuilt 3×3 (`fig5_<model>.pdf`) — model not yet chosen |
| S1 | promoted into Figure 3 |
| S2, S3 | PPC gallery + model comparison, rebuild converged-only |
