# Raw-RT signatures of a within-trial diffusion-noise increase

Companion to `notes/rdm_magnitude_rt_plan.md`. Question (Gilles, 2026-08-22):
bauer's RDM is a race-*diffusion* model, not an LBA — so what does a
cTBS-induced increase in within-trial noise predict for raw RTs, and is it
in the data?

## 1. What "noise" is in bauer's race — both channels at once

`_drifts_from_post_and_prior` (race.py) maps front-end encoding noise σ_e
into the accumulators along TWO paths:

- **Within-trial diffusion σ** of each accumulator: `σ_k = w_k·σ_e,k`
  (`posterior_mean_sd`), with w = σ²_p/(σ²_p+σ²_e). This is genuine
  diffusion — dσ_k/dσ_e > 0 whenever σ_e < σ_p, which holds in the
  log-space regime (σ_e ≈ 0.2 vs σ_p ≈ 0.4 log-units). An LBA has no such
  channel at all.
- **Drift signal**: posterior-mean tilde values shrink toward the prior as
  σ_e rises, compressing the w_d·(Δ) term (a drift-decrease component).

So a cTBS noise increase in the fitted RDM is a *mixture* of the diffusion
and drift stories below; the fits decompose it, the raw data can only show
which pattern dominates.

## 2. Signature table (racing Wald accumulators, simulated, 200k trials)

Baseline v=(2.6, 2.0), σ=1, a=1.5, t0=0.3. Winner = min of two independent
IG first passages (bauer's own generative process).

| manipulation      | mean RT | RT SD | CV   | q10  | q90  | P(err) | median err−corr |
|-------------------|---------|-------|------|------|------|--------|-----------------|
| baseline          | .764    | .198  | .259 | .551 | 1.03 | .376   | +.010 |
| **diffusion +40%**| **.708 ↓** | .239 ↑ | **.338 ↑↑** | **.476 ↓↓** | 1.01 ≈ | **.415 ↑** | **.001 → 0** |
| drift −25%        | .888 ↑  | .287 ↑↑ | .323 ↑ | .590 ↑ | 1.26 ↑↑ | .394 ↑ | +.008 |
| threshold +25%    | .900 ↑  | .229 ↑ | .255 ≈ | .646 ↑↑ | 1.20 ↑ | .359 ↓ | +.018 ↑ |

The dissociating signature of **diffusion↑** is unique: *faster* mean and
especially faster 10th percentile (selection: noisier racers finish
sooner), higher CV, more errors, and errors as fast as corrects. Drift↓ and
threshold↑ both slow everything. An LBA (no within-trial noise) cannot
produce the speed-up-with-more-errors pattern from a "noise" change at all
— this is the RDM-specific prediction.

This also reinterprets the earlier finding (`rdm_magnitude_rt_plan.md`):
the IPS-faster-at-low-stakes trend (−0.047 log-units, p = 0.11) is the
*diffusion* signature's sign, not the drift story's.

## 3. The data (model-free; 35 subjects, sessions 2/3, rt ≥ 0.20 s)

Per subject × stimulation × within-subject stake tercile (× order),
IPS − vertex, paired t across subjects. Error = choosing the lower-EV
option (0.55·risky vs safe).

**Home cell (risky second, lowest stake tercile)** — every one of the four
diffusion-direction signs is present, none individually significant:

| metric | IPS − vertex | p |
|---|---|---|
| median RT | −0.026 s (faster) | .43 |
| q10 | −0.027 s (fatter left tail) | .30 |
| CV | +0.022 (more dispersed) | .24 |
| P(error) | +0.032 (more errors) | .24 |

Composite diffusion score (equal-weight standardized combination of the
four, direction fixed a priori by the simulation table):
**mean +0.17, t(34) = 1.90, p = 0.066**; 19/35 subjects show ≥3/4 signs.

Elsewhere: collapsed over order, low-stake median/q10 are −0.029/−0.029
(p = .36/.18), dispersion flat. Risky-first shows no coherent pattern; the
single nominally significant cell in the 30-test sweep (risky-first,
mid-stake, errors *slower* under IPS, p = .029 uncorrected) is best treated
as noise.

**Verdict**: the raw RTs lean the diffusion way, in the right cell, with
the right four-sign pattern, at p ≈ 0.07 for the pre-directed composite —
suggestive, not sufficient. The proper weighing of all trials and all
signatures simultaneously is the fitted RDM, which models exactly this
mixture (σ_k = w·σ_e and drift shrinkage jointly). That is what the
lapse-equipped fits (job 5227421) will deliver: the fitted decomposition of
the cTBS effect into diffusion vs drift channels, plus the w_s
(magnitude→speed) test via `rdm_logflex2b` vs `rdm_logflex2b_ws0` vs the
matched DDM.

## 4. Fitting status (2026-08-22)

- Lapse mixture implemented in bauer `303bad4` (both mixins; likelihood +
  LOO path + simulate/ppc; p_lapse = 0.05 fixed default; p_lapse = 0
  reproduces the old likelihood to 1e-10, tested). Stamped into traces as
  `tms_risk_p_lapse` (tms_risk `df98005`).
- The first (lapse-free) GPU wave 5225709 **never sampled** — all six tasks
  died at startup on the node-local /tmp tempfile quota (fixed in
  `fit_rdm_logflex.sh`, tms_risk `b924d7e`), so "cannot fit without
  lapse" remains untested; the lapse-equipped wave is the first real
  attempt either way.
- Submitted: GPU array **5227421** (6 tasks: rdm_logflex2_null / 2b /
  2b_ws0, ddm_logflex2_null / 2b / 2_threshold; L4, numpyro, mapjitter,
  tune 2000 / draws 1000, target_accept 0.99). Smoke 5227418 queued in
  parallel (env/graph already GPU-validated by 5224998).

## 5. What to check when the fits land

1. Convergence first (Wald/WFPT + mapjitter; r̂, divergences, and whether
   the lapse mixture indeed rescued fitting).
2. `w_s` posterior sign/size and the two ELPD gaps: rdm vs rdm_ws0
   (magnitude→speed channel) and rdm vs matched ddm (sum-channel
   architecture test). Joint choice+RT ELPD compares only within the
   accumulator set.
3. cTBS decomposition: does the fitted noise increase land in σ (diffusion)
   or the drift terms — and does its simulate() reproduce the four-sign
   pattern of §3 (the PPC that can fail)?
4. Priors under RT constraint: do the lognormal priors stay at ~17/10 CHF
   (the ridge-breaking question — RT dispersion pins w·σ_e absolutely)?
