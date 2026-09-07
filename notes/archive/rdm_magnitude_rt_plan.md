# The magnitude→RT effect and the RDM × log-space PMC — plan (2026-08-22)

Extends `notes/rdm_logflex_memo.md` (design memo). New here: the theoretical
case for why only a race captures magnitude→RT effects, a model-free
verification that the effect exists in OUR data (it does, decisively), and
the concrete first fitting wave (launched on sciencecluster; labels below).
Exploratory — outside the paper tree.

## 1. Why only a race can carry a magnitude→RT effect

A DDM's drift is a function of the evidence **difference** (here:
(post₂−post₁)+log(p₂/p₁), `ddm.py::_drift_from_snr`). Two gambles with the
same perceived ratio but different stakes produce the same drift, the same
RT distribution. The overall payoff level cannot speed or slow a DDM except
through the difference — the classic argument of Teodorescu & Usher and
Pirrone et al., and the robust "overall value speeds RTs" finding in
value-based choice.

A race gives each option its own accumulator, so the **sum** of the two
evidence streams drives speed. bauer's RaceMixin makes the channel explicit
(van Ravenzwaaij 2020 advantage decomposition, `race.py:437-447`):

    v₁ = w₀ + w_d·(tilde₁ − tilde₂) + w_s·(tilde₁ + tilde₂)

with `tilde_k = posterior_k − prior_mean_k` — the **deviation of the
posterior from the prior**, in the front-end's own space. Three consequences
worth being precise about:

- **`w_s` is the magnitude→RT channel**, but it operates on the summed
  *prior deviations*, not raw payoffs. In the log-space front-end
  tilde_k = w_k·(log n_k − μ_k): the sum term is (up to the shrinkage
  weights) the **log-stake relative to prior expectations**. `w_s > 0`
  predicts: stakes above expectation race faster — log-linear in stake.
- **The probability-cancellation stance** (memo §2): with `p1/p2` exposed,
  log p_k is added to both the posterior and its centering baseline and
  cancels in tilde — probabilities act on the race only through the
  role-specific priors. Inherited by all existing `rdm_*` fits; the DDM
  variant carries log(p₂/p₁) explicitly in the drift. We keep both, and the
  DDM/RDM contrast brackets the stance.
- **The ablation is built in**: `fit_w_s=False` (`race.py:443`) deletes the
  sum channel. RDM(w_s free) vs RDM(w_s=0) is the *within-architecture*
  magnitude-effect test; matched DDM vs RDM is the *between-architecture*
  version.

## 2. The effect exists in our data (model-free, N = 35, 8,325 trials)

Per-subject regression of log RT on log stake ((n_safe+n_risky)/2), TMS
sessions, rt ≥ 0.20 s, t-tests across subjects:

| Contrast | slope | t(34) | p |
|---|---|---|---|
| log RT ~ log stake | **−0.049** | −3.68 | **0.0008** |
| … controlling difficulty \|log ratio − log 1.82\| | −0.047 | −3.32 | 0.0022 |
| … risky second only | **−0.084** | −4.73 | **4×10⁻⁵** |
| … risky first only | −0.015 | −0.84 | 0.41 |

Higher stakes → faster decisions, ~5% RT per e-fold of stake, and the
effect is almost entirely a **risky-second** phenomenon — the same cell
that carries the choice-level cTBS localization. The order-specificity is
qualitatively what deviation-racing predicts: in risky-second trials the
large-payoff option is the well-trusted (second) accumulator, so the summed
deviation varies most with stake. A pure-DDM account has no channel for any
of this once difficulty is controlled.

Chronometric cTBS check (memo §3.2 predicted slower low-stake RTs under
IPS): observed the **opposite trend** — IPS − vertex log RT = −0.047
(p = 0.11) at low stakes, −0.034, −0.011 at mid/high (graded), with no
variability difference (p = 0.42). Faster-and-more-random at low stakes is
the signature of a **within-trial noise increase** (more diffusion → earlier
threshold crossings), not a drift decrease. Not significant, but it points
the accumulator analysis at exactly the right question: does cTBS act on
drift or on diffusion noise? The static model cannot ask this.

## 3. The identifiability prize, restated with the mechanics

Choices constrain ratios; the tiny-Gaussian and lognormal regimes tie in
choice-ELPD while implying accumulator noises (w·σ_e) of ≈0.75 vs ≈0.12
log-units (memo §3.1) — and RaceMixin feeds exactly w·σ_e into the Wald
first-passage as per-accumulator diffusion (`race.py:400-404, 423-426`).
`w_d`, `w_s`, `a` can rescale drifts globally, but the *stake- and
order-profile* of RT means and dispersion is fixed by the front-end's w(n)
profile — that is what breaks the ridge. Expectations tempered per memo §6:
this is a shape test, not point identification.

## 4. First wave (submitted; out_folder cogmodels.rdmlogflex)

| Label | Model | Question |
|---|---|---|
| `ddm_logflex2_null` | DDM × logflex | RT baseline; regime comparison anchor |
| `ddm_logflex2b` | + TMS on perceptual splines | localization under RT constraint |
| `ddm_logflex2_threshold` | + TMS on `a` only | careless vs degraded-code |
| `rdm_logflex2_null` | Race × logflex | architecture baseline |
| `rdm_logflex2b` | + TMS on perceptual splines | headline joint fit |
| `rdm_logflex2b_ws0` | + `w_s = 0` | the magnitude-channel ablation |

Comparisons that are fair (same outcome space): everything within this
table; plus ddm_logflex2_null vs the existing natural-space
`ddm_flexible_null` for the ridge/regime question. NOT comparable to the
choice-only ladder. Accumulator recipe throughout: numpyro on one GPU,
tune=2000, draws=1000, target_accept=0.99, mapjitter init, vectorized
chains, rt ≥ 0.20 s. Prior-mean wandering mitigation shipped with these
labels: prior-μ hyperprior centering tightened to σ = 0.5 (log units).

## 5. What to check when fits land

1. Convergence (r̂, divergences) — Wald/WFPT fits are init-sensitive.
2. `w_s` posterior in `rdm_logflex2b` (sign, CrI) and the ELPD gaps
   rdm vs rdm_ws0 (within-race magnitude test) and rdm vs ddm (architecture).
3. Whether the priors stay lognormal-realistic under the RT constraint —
   the ridge question. Compare w·σ_e posteriors against the natural-space
   ddm_flexible fits.
4. Where TMS lands: perceptual splines vs threshold `a` (ELPD + coefficients),
   and whether the fitted noise increase reproduces the faster-at-low-stakes
   IPS trend from §2.
