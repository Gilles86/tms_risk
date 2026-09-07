# RDM/DDM × log-space flexible PMC — design memo (2026-08-22)

Question: can the Race-Diffusion (and DDM) versions of the model family test
the TMS-risk data further — specifically, can RTs break the level/prior
identifiability ridge that choices alone cannot, and does the low-stake cTBS
localization have a chronometric signature? Design only; nothing fitted.

## 1. Current state

- Accumulator machinery exists and is production-tested: `RaceMixin`
  (`libs/bauer/bauer/models/race.py:148`, Wald race, `w_0/w_d/w_s/a/t0`,
  advantage decomposition per van Ravenzwaaij 2020) and `DDMMixin`
  (`ddm.py`, WFPT). Both compose with cognitive front-ends by multiple
  inheritance; the front-end only has to emit the standard
  `n{1,2}_prior_mu/sd`, `n{1,2}_evidence_mu/sd` (+ `threshold` or `p1/p2`)
  in `get_model_inputs`.
- Existing labels `ddm_*`/`rdm_*` (`fit_model.py::_build_ddm_or_rdm`) already
  fit the *natural-space* Weber and flexible front-ends to these data.
- The new `LogFlexibleNoiseRiskModel` (log evidence, `threshold =
  log(p2/p1)`, lognormal priors, spline σ(log n); `risky_choice.py`, tail of
  file) emits exactly the keys the mixins consume.

## 2. What composition costs

**DDM × LogFlexible: essentially free (~25 lines).** `_drift_from_snr`
(`ddm.py:500`) is front-end-agnostic and *explicitly* uses
`model_inputs['threshold']`, so the log-space EU comparison flows into the
drift unchanged: drift = ((post₂−post₁)+log(p₂/p₁)) / √(σ₁²+σ₂²) with
σᵢ = wᵢ·σ_e,i (posterior-mean noise — same composition as the static rule).

```python
class DDMLogFlexibleNoiseRiskRegressionModel(DDMMixin,
                                             LogFlexibleNoiseRiskRegressionModel):
    def __init__(self, paradigm, regressors, ..., fit_v_scale=False, fix_z=True):
        self.fit_v_scale, self.fix_z = fit_v_scale, fix_z
        LogFlexibleNoiseRiskRegressionModel.__init__(self, ...)
    def _get_drift(self, model_inputs, parameters):
        v_scale = parameters['v_scale'] if self.fit_v_scale else None
        return _drift_from_snr(model_inputs, v_scale=v_scale)
```

**Race × LogFlexible: near-free (~30 lines) plus one 2-line fix and one
substantive caveat.**
- Fix: `_drifts_from_post_and_prior` (`race.py:387`) detects risk front-ends
  via `'p1' in model_inputs`; `LogFlexibleNoiseRiskModel.get_model_inputs`
  does not currently expose `p1/p2` (they live only in `threshold`).
  Composing as-is would silently drop the probabilities. Add
  `mi['p1'] = model['p1']; mi['p2'] = model['p2']` to
  `LogFlexibleNoiseRiskModel.get_model_inputs` (harmless for the static rule).
- Caveat to decide *before* fitting: under the race's prior-centered
  advantage decomposition (`race.py:433-447`), the `log p_k` added to both
  the posterior and its centering baseline **cancels in `tilde_k`** — the
  probabilities influence the race only through the role-specific
  (risky vs safe) priors, not as an EU term in the drift. That is a
  substantive stance (deviation-from-expectation racing) inherited by all
  existing `rdm_*` risk fits, not specific to log space; the DDM path does
  carry the threshold explicitly. Recommendation: lead with the DDM variant,
  add the RDM as the architecture comparison afterwards.

## 3. Scientific payoff (ranked)

1. **RTs can break the level ridge.** Choices constrain only
   signal-to-noise *ratios*, which is why the tiny-Gaussian-prior and
   lognormal-prior regimes tie (Δ ≈ 15 ± 11 ELPD) while disagreeing
   radically about latent scale (w ≈ 0.15 vs w ≈ 0.6 at 20 CHF). In an
   accumulator, drift and diffusion enter the *time* domain separately:
   the same choice probability can be reached by (small drift, small noise)
   or (large drift, large noise), but those predict different RT
   distributions (mean RT ∝ a/v; skew/variance tie down σ within trial).
   Concretely, the two regimes imply very different w·σ_e (the accumulator
   noise, `posterior_mean_sd`, `bayes.py:26`): ≈0.75 (tiny-Gaussian) vs
   ≈0.12 log-units (lognormal) — an order-of-magnitude difference that RT
   dispersion should see even after `a`, `t0`, `v_scale` absorb overall
   scaling. This is the single strongest reason to fit RT models here.
2. **Chronometric signature of the cTBS localization.** If cTBS raises
   perceptual noise at small payoffs, drift falls there → IPS sessions
   should show slower, more variable RTs specifically on low-stake trials
   (and mostly risky-second, via the noise asymmetry). This is a new,
   untouched prediction — the paper has never used RTs — and it is testable
   model-free (RT ~ stimulation × stake quantile) before any fit.
3. **Threshold vs evidence placement of the TMS effect.** The accumulator
   separates `a` (caution) from noise: labels with TMS on `a` vs on the
   spline coefficients test "cTBS made people careless" against "cTBS
   degraded the magnitude code" — a confound the static model cannot even
   express. The existing `ddm_flexible_threshold` labels do this for the
   natural-space model (`fit_model.py:208`).
4. **OV/common-signal term.** `w_s` (summed value → faster RTs) connects to
   the overall-value RT literature and is a free by-product.

**Hard limitation to state up front:** joint choice+RT likelihoods live on a
different outcome space — their ELPD is **not comparable** to the
choice-only ladder. Model comparison happens *within* the accumulator set
(and via each model's choice-marginal PPC against the static fits).

## 4. Feasibility

- Data: `rt` ships in `get_all_behavior`; existing accumulator labels
  already filter `rt >= 0.20 s` (`fit_model.py:594`, ~3–4% of trials);
  RaceMixin enforces rt > 0 and caps t0 at 0.95·min(rt) per subject
  (`race.py:200`).
- Recipe (documented, validated): numpyro backend on one GPU,
  `tune=2000, draws=1000, target_accept=0.99`, `chain_method='vectorized'`,
  bauer's `mapjitter` init (`libs/bauer/notes/fitting_ddm_models.md`;
  `fit_model.py` accumulator branch already encodes it).
- Where: **sciencecluster GPU** fits the migration decision —
  `--gres=gpu:L4:1`, `bauer_cuda` env (hssm/jax/numpyro; known CUDA env),
  `standard` partition, `--time=04:00:00` (the DDM brief's own template).
  The sciencecloud T4 boxes remain a fallback.
- Wall-time: prior tms_risk DDM/RDM fits at 2000+1000 ran overnight-scale
  on T4s; L4 + vectorized chains ≈ 1–3 h per fit.

## 5. Proposed first wave (6 fits)

| Label (proposed) | Model | Purpose |
|---|---|---|
| `ddmlog2_null` | DDM × logflex, no TMS | baseline; does the log front-end hold up with RTs |
| `ddmlog2b` | TMS on perceptual splines | the headline: localization with RT constraint |
| `ddmlog2_threshold` | TMS on `a` only | caution confound control |
| `rdm_flexible_null` (exists) / `rdmlog2_null` | race versions | architecture check (after the p-cancellation decision) |
| `ddm_flexible*` (existing natural-space fits) | re-use | regime comparison: does RT dispersion prefer tiny-Gaussian or lognormal w·σ? |

Decision metric for the ridge question (§3.1): compare `ddmlog2_null`
(lognormal regime) against the existing natural-space `ddm_flexible_null`
analogue on the *joint* likelihood — same outcome space, so ELPD is fair —
plus posterior w·σ_e against the RT-implied accumulator noise.

## 6. Risks

- Prior-mean wandering may recur; the mild centering hyperprior fix should
  ship with the first accumulator fits, not after.
- `v_scale`/`a` can partially re-absorb the absolute scale (drift is only
  identified up to the diffusion constant σ=1 convention); the ridge test
  is therefore "does the *shape* of RT distributions prefer one regime",
  not a clean point identification — temper expectations accordingly.
- WFPT/Wald fits are seed-sensitive without `mapjitter` (12% → 100%
  convergence per bauer's experiments); keep the finder on.
- hssm import must exist in the cluster `bauer_cuda` env — verify before
  the array (`python -c "import hssm"` in a debug-QoS job).
