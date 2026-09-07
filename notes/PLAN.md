# The plan, as of 2026-08-31

**Read this first.** Everything else in `notes/` is background; anything in
`notes/archive/` is superseded and kept only for provenance.

---

## 1. Where we are in one paragraph

The behavioural result is model-free and solid: cTBS over parietal cortex makes
people **more risk-seeking and less consistent**, but only when the risky option
came **second** and only at **low stakes**. That is the published Figure 3 and it
does not depend on any cognitive model. The Perceptual-and-Memory-based Choice
model was refit under a new **anchor parameterisation** (`fit_anchor.py`), where
the free parameters *are* the noise SDs at named payoffs rather than spline
coefficients on a softplus scale. That refit reproduces the direction, the
order-specificity and the stake-specificity of the effect, and **underpredicts
its magnitude** — by roughly 2x on choice consistency and 5x on risk attitude.
Closing that gap is the one open scientific question.

## 2. The primary model — UNDER REVIEW

`log-power-n1n2` was the primary model until 2026-08-31. Three independent lines
now say the two-anchor `power` form is the wrong choice, because it cannot
produce the *shape* of the effect:

- its cTBS-induced shift in the choice index decays from 0.04 to 0.007 log units
  across the ratio range, so the predicted effect peaks at ratio ~1.35, **below
  the lowest ladder rung the experiment presented**;
- `logflex2` (spline, old parameterisation) holds +0.026 to +0.038 flat across
  all five safe payoffs, where `log-power-n1n2` decays from +0.017 to +0.012;
- `power_ppc_tms.pdf`, made before any of this, already shows the spline model
  tracking the data's peak at +0.10 while the power law sits flat at +0.03.

**Pending decision.** `log-spl5+affine-percmem` — five perceptual anchors, two
memory anchors — is fitting (job 5432728). If its shift stays flat across the
ratio range it becomes primary and `power` drops to a robustness line.

## 3. What is settled, and should not be relitigated

| Question | Answer | Evidence |
|---|---|---|
| Log or natural space? | **Log.** | Natural space fails the derived probit badly (Δslope −0.06 vs −0.40); ELPD margin is only ~10 once the noise function is flexible, but the Figure-3 quantities are decisive. |
| Which choice rule? | **KLW-consistent is correct**; bauer's default is not. Conclusions barely move, but state it in Methods. | `notes/klw_variance_analysis.md`; Rule-A grid (53 fits). |
| Does prior width explain the effect? | **No.** It gains +30 ELPD and contributes nothing to either Figure-3 quantity (Δslope ~0.00). | `prior_sensitivity.pdf`, derived probit across 20 models. |
| Are the priors shrinking the effect? | **No.** Widening σ_slope and τ_slope 4x grows the noise parameter 61% and changes predictions by 0.002-0.005. Keep PRIOR_SPEC as it is. | `ppc_prior_loose.pdf` |
| Is the ΔP magnitude a bug? | **No.** Independent re-derivation matches to 2e-16 and matches the simulated-choice PPC at r = 0.991. | `notes/audit_dp_magnitude.md` |
| Do more mechanisms help? | **No.** Role-scale, order-interactions, prior-mean shifts, EV-indexing: none closes the gap, and role-conditioned cTBS is indefensible from a magnitude-code premise anyway. | derived probit ranking |

## 3b. Figure layout: Figure 5 is retired into a full-page Figure 4

Decided 2026-09-01. The old Figure 5 does not survive the anchor refit: its
per-option perceived-value panels show the two halves of a ratio separately,
which is not what drives choice, and its bottom row spends a whole axis on a
two-point difference. Everything worth keeping moves into one Figure 4, built by
`tms_risk/behavior/scripts/plot_fig4_big.py`:

    row 1  a b   the two noise terms, IPS vs vertex
           c d   the cTBS effect on each, with a rug where the 95% CrI excludes 0
    row 2  e     where the priors sit
           f g   the two ingredients of the decision variable, by order
    row 3  h i   P(chose risky) vs safe payoff, by order
           j     model comparison

Output: `notes/figures/fig4_big_log-power-n1n2.pdf` (7.25 x 6.7 in).

### Why there are no probit panels

A slope / risk-neutral-probability version of the choice PPC was built and then
dropped. **The model's choice function is not a probit in log(frac).** nu depends
on payoff and the risky payoff is `frac * n_safe`, so `w_R` and `diff_sd` both
move along the ladder and the probit index is not affine in log frac.
`extract_anchor_probit` handles this by linearising at each cell's mean payoffs,
which its own docstring flags as an approximation.

Measured at the fitted parameters of `log-power-n1n2` (vertex arm, both choice
rules), the approximation costs:

| cell | max deviation from the best-fit probit | linearised vs true slope |
|---|---|---|
| risky first, all payoffs | 0.6 pp | within 1-3% |
| risky second, safe 7-14 | 0.9-1.7 pp | within 1-4% |
| risky second, safe 20 | 2.5 pp | 6% low |
| risky second, safe 28 | 3.7 pp | **14% low** |

Small on average, but systematic, growing with payoff, and worst for
risky-second -- which is exactly the order x payoff contrast the paper claims.
So the earlier reading of those panels ("the model underpredicts risky-first
consistency by ~1 probit unit") is **not safe to report**: part of that gap is
the linearisation, not the model. P(chose risky) needs no linearisation on
either side, so panels h and i carry the consequence instead.

The code is still there behind `--with_probit` (off by default), and so is the
observed-side machinery, in case a genuinely non-linearised version is wanted
later. Two facts to keep if it is:

* **Cells must be the stake median split, not safe payoff.** Per subject per
  safe payoff is 12 trials and fits **zero** of 700 cells
  (`fit_observed_probit_subject --by n_safe`). Pooling subjects within a
  safe-payoff cell fits but measures the wrong thing -- averaging psychometric
  functions with different indifference points flattens the aggregate, giving
  slopes of 0.4-1.8 against a per-subject 2.5-3.5. `fit_observed_probit_hier`
  rescues it hierarchically if ever needed.
* **The two sides parameterise the indifference point from opposite ends.**
  `extract_anchor_probit` reports `p_R * frac*`, the EV ratio at indifference;
  `fit_observed_probit_subject` reports `1/frac*`, the risk-neutral probability
  itself. They agree via `RNP = p_R / (p_R frac*)`. Without that conversion the
  comparison is 1.10 against 0.50.

## 4. Open, in priority order

1. **Does `spl5+affine` close the shape gap?** The one thing that could change
   the primary model. Job 5432728.
2. **Which probit variant is the target?** `probit_stake_group_posterior.ri.tsv`
   (`--random_effects intercept`, the published structure) gives Δslope −0.666
   [−1.043, −0.286]; the full-random-effects file gives −0.398 [−0.880, +0.113].
   Every comparison figure currently uses the *second*. The published structure
   is `.ri`, so figures should be re-pointed at it — and against that target the
   model underpredicts consistency too, not only risk attitude.
3. **The amplitude gap itself.** If no model closes it, report it: the PMC
   explains the pattern and about a fifth of the size, and say why — a noise
   change converts into consistency directly and into bias only through
   shrinkage.

## 5. Things deliberately NOT in the paper

- the subject-wise Δrnp anti-correlation (r = −0.43, n = 32, reliability 0.49) —
  real but fragile, and it argues against our own model on the weakest data;
- prior-width as a mechanism (fit caveat only);
- natural-space arm (supplementary at most);
- role-conditioned and order-interaction placements (diagnostics only).

## 6. Where things live

| What | Where |
|---|---|
| Anchor grid fits | `<bids>/derivatives/cogmodels.anchor/model-<label>[.variant]_trace.netcdf` |
| Label grammar | `tms_risk/behavior/fit_anchor.py` — `<space>-<form>-<placement>` |
| Per-figure provenance | `notes/PROVENANCE.md` |
| Noise-form maths | `notes/anchor_noise_forms_derivation.md` |
| Choice-rule maths | `notes/klw_variance_analysis.md` |
| ΔP audit | `notes/audit_dp_magnitude.md` |
| Superseded notes | `notes/archive/` |
| Superseded figures | `notes/figures/archive/` |

### bauer clones on sciencecluster (never check out the shared `libs/bauer`)

| Path | What it adds |
|---|---|
| `/scratch/gdehol/bauer_anchor` | the anchor parameterisation (baseline) |
| `/scratch/gdehol/bauer_role` | `role_scale` — EV-indexed or free risky/safe noise multiplier |
| `/scratch/gdehol/bauer_multi` | `noise_form='A+B'` — different form per noise channel |
