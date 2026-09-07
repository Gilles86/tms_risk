# How ν and σ jointly set the variance of the decision variable

*(Khaw, Li & Woodford 2021 noisy-coding model, as implemented in `bauer`)*

Everything below is derived from first principles, checked against the KLW paper
(local copy: `/Users/gdehol/git/bauer/notes/papers/Cognitive Imprecision and
Small-Stakes Risk Aversion.pdf`, OUP advance-article proof, pages 1–35 — **not**
the final 1979–2013 pagination, so page numbers below are proof pages; equation
numbers are stable across versions), and verified numerically. The verification
log is at the end: every claim is tagged **[A]** analytic, **[N]** numeric, **[P]**
confirmed in the paper, or **[?]** unverified.

---

## 1. Setup

### 1.1 KLW's model (confirmed from the paper)

Encoding is on the **log** payoff, with a **single shared** ν and a **single
shared** Gaussian prior over log payoff **[P]**:

> r_x ∼ N(log X, ν²),  r_c ∼ N(log C, ν²)   — eq. (2.1), p. 11
> log X, log C ∼ N(μ, σ²)   — eq. (2.3), p. 11

The observer chooses the risky lottery iff `p·E[X|r_x] > E[C|r_c]` (eq. 2.2).
With log-normal posteriors, `E[X|r] = e^{α+βr}` with the shrinkage weight

> β ≡ σ²/(σ² + ν²)   — eq. (1.3)

Taking logs of (2.2): `log p + β r_x > β r_c`, so **β divides out of both sides**:

> r_x − r_c > β⁻¹ log p⁻¹   — eq. (2.4), p. 12
> Prob[accept risky | X, C] = Φ( (log X/C − β⁻¹ log p⁻¹) / (√2 ν) )   — eq. (2.5), p. 12

Two facts that matter enormously and are easy to get wrong:

- **p is treated as known**, not noisily encoded (p. 11; p = 0.58 fixed in their
  experiment). It therefore enters the comparison *un-shrunk*, after inference.
- **There is no separate late/choice-stage noise.** The model has exactly two free
  parameters, σ and ν (p. 12) **[P]**. (The 2022 WTP follow-up *does* add response
  noise ν_c; the 2021 binary-choice model does not.)
- ν_x = ν_c and σ_x = σ_c are *assumed equal* — this is what produces the paper's
  central prediction of scale invariance **[P]**.

Note the equivalence that trips people up. Eq. (2.5) divides by the **raw** √2ν —
but that is *not* a claim that the decision variable has SD √2ν. The decision
variable on the posterior-mean scale is `β(r_x − r_c)`, whose SD **is** `β√2ν`,
and its threshold is `log p⁻¹`. KLW simply divided numerator and denominator by β.
Writing it as

  `Φ( [β·log(X/C) + log p] / (β√2ν) )`  ⟺  `Φ( [log(X/C) − β⁻¹log p⁻¹] / (√2ν) )`

is the same model **[A][N]**. So **KLW *are* the posterior-mean-SD ("rule A")
normalisation.** Shrinkage attenuates signal and noise identically, so β cancels
from the *slope* and survives only in the *threshold*.

### 1.2 What `bauer` implements

`bauer/utils/bayes.py`, called as `get_posterior(prior_mu, prior_sd, evidence_mu,
evidence_sd)`:

```python
def get_posterior(mu1, sd1, mu2, sd2):                                   # L8
    var1, var2 = sd1**2, sd2**2
    return mu1 + (var1 / (var1 + var2)) * (mu2 - mu1), pt.sqrt((var1 * var2) / (var1 + var2))

def get_diff_dist(mu1, sd1, mu2, sd2):                                   # L18
    return mu2 - mu1, pt.sqrt(sd1**2 + sd2**2)

def posterior_mean_sd(prior_sd, evidence_sd):                            # L26
    w = prior_sd**2 / (prior_sd**2 + evidence_sd**2)
    return w * evidence_sd

def cumulative_normal(x, mu, sd, s=pt.sqrt(2.)):                         # L58
    return pt.clip(0.5 + 0.5 * pt.erf((x - mu) / (sd * s)), 1e-9, 1 - 1e-9)   # = Φ((x-mu)/sd)
```

So `post_mu = w·r + (1−w)·μ_p` and `post_sd² = σ²ν²/(σ²+ν²) = w·ν²`, with
`w = σ²/(σ²+ν²) = β`. **`w` in bauer is exactly KLW's `β`.**

**Two different spaces.** `RiskModel` is KLW's log space
(`risky_choice.py:574,576`: `n1_evidence_mu = pt.log(n1)`, `threshold =
pt.log(p2/p1)`). `FlexibleNoiseRiskModel` — the paper's PMC / Flexible PMC — is in
**natural payoff space** (`risky_choice.py:828`: `n1_evidence_mu = model['n1']`),
with p multiplying the posterior mean rather than entering as a log threshold.

**Two different choice rules.** Write ν_k for evidence SD, w_k for the weight.

| | decision-variable SD used |
|---|---|
| **Rule B — "raw evidence SD"** | `sqrt(ν₁² + ν₂²)` |
| **Rule A — "posterior-mean SD" (= KLW)** | `sqrt((p₁w₁ν₁)² + (p₂w₂ν₂)²)` |

Where each lives:

```python
# bauer/core.py:97  -- BaseModel._get_choice_predictions  (RiskModel, log space)  = RULE B
diff_mu, diff_sd = get_diff_dist(post_n2_mu, model_inputs['n2_evidence_sd'],
                                 post_n1_mu, model_inputs['n1_evidence_sd'])

# ecc6454 bauer/models.py:1458 -- FlexibleNoiseRiskModel, 'payoff' branch        = RULE B
diff_mu, diff_sd = get_diff_dist(post_n2_mu * p2, model_inputs['n2_evidence_sd'],
                                 post_n1_mu * p1, model_inputs['n1_evidence_sd'])

# HEAD bauer/models/risky_choice.py:853 -- same method                           = RULE A
n1_noise = post_n1_sd**2 / model_inputs['n1_evidence_sd'] * model_inputs['p1']
n2_noise = post_n2_sd**2 / model_inputs['n2_evidence_sd'] * model_inputs['p2']
diff_mu, diff_sd = get_diff_dist(post_n2_mu * p2, n2_noise, post_n1_mu * p1, n1_noise)
```

`post_sd²/ν = (wν²)/ν = wν = posterior_mean_sd(σ, ν)` — so **the new flexible rule
is exactly the KLW-consistent normalisation** (× p) **[A][N]**.

**State of the code, which is not uniform.** The tms_risk submodule is detached at
`4cd98a4`, which does **not** contain `62eaf11` ("core: add
consistent_choice_noise"); that commit is on `origin/main` only. So:

| model | space | rule at submodule `4cd98a4` |
|---|---|---|
| `RiskModel` / `RiskRegressionModel` (PMC) | log | **B** (unconditional) |
| `FlexibleNoiseRiskModel` (Flexible PMC) | natural | **A** (unconditional) |
| `PowerLawNoiseRiskModel` (`risky_choice.py:1252`) | natural | **B** |
| `DDMMixin` / `RaceDiffusion` (`ddm.py:534`, `race.py:423`) | either | **A** |
| `BaseModel` on `origin/main` | log | B by default; A iff `consistent_choice_noise=True` |

`notes/race_diffusion_math.md` §2 already flags this: *"In bauer's implementation,
the variance is computed from ν_k² directly (not β_k²ν_k²); both work and just
shift the SNR scale, but strictly only the latter matches 'variance of the
posterior mean'."* The analysis below makes "just shift the SNR scale" precise —
it is not innocuous.

**Notation from here on**: ν = encoding SD, σ = prior SD, w = β = σ²/(σ²+ν²),
μ_p = prior mean, μ̂ = posterior mean, p_R = risky win probability, and R/S
subscripts for risky/safe.

---

## 2. Derivations

### Q1 — Across-trial SD of the posterior mean

μ̂ = w·r + (1−w)·μ_p is affine in r, and r ∼ N(n, ν²) at fixed true n. Hence

  **E[μ̂ | n] = w·n + (1−w)·μ_p**,  **Var[μ̂ | n] = w²ν²**,  **SD[μ̂ | n] = w·ν = σ²ν/(σ²+ν²)**

Three algebraically related but conceptually distinct quantities:

| quantity | value | meaning |
|---|---|---|
| encoding SD | ν | noise on one observation |
| posterior width | σ_post = √w · ν | the observer's *self-reported* uncertainty |
| **SD of the posterior mean** | **w·ν = σ_post²/ν** | trial-to-trial spread of the estimate |

Verified by simulation (4×10⁶ draws, ν = 1.3, σ = 2.1): sim SD = 0.939704,
w·ν = 0.939836 **[N]**. And `post_sd**2/nu == w*nu` to machine precision **[N]**.

### Q2 — SD(μ̂) is non-monotonic in ν. Confirmed.

Let f(ν) = wν = σ²ν/(σ²+ν²).

  **f′(ν) = σ²(σ² − ν²)/(σ² + ν²)²**

exactly as conjectured **[A][N]** (checked against `np.gradient` on a 4×10⁵ grid).
Sign is that of (σ² − ν²):

- **f′ > 0 for ν < σ** — rises **[N]**
- **f′ = 0 at ν = σ**, **f(σ) = σ/2** — peak **[N]**
- **f′ < 0 for ν > σ** — falls, f(ν) → σ²/ν → 0 **[N]**

A one-line proof of the peak: the *reciprocal* is a sum of one decreasing and one
increasing term,

  **1/(wν) = 1/ν + ν/σ²**   **[A][N]**

minimised by AM–GM at ν = σ with value 2/σ, i.e. max f = σ/2.

**Consequence: SD(μ̂) ≤ σ/2 for every ν** — the across-trial spread of the
posterior mean can never exceed half the prior SD, no matter how bad the encoding
gets **[A][N]** (checked over ν ∈ [10⁻⁴, 10⁴]).

**Intuition.** Below ν = σ the observer still mostly trusts the signal, so noise
in r passes through to μ̂. Above ν = σ the observer discounts the signal so hard
that μ̂ collapses onto μ_p: the estimate becomes *less* variable, not more. In the
limit ν → ∞ the observer answers "μ_p" every trial — perfectly reliable and
perfectly uninformative.

**Behavioural consequence — and the caveat that matters most here.** Under rule A
this non-monotonicity is **invisible in choice**, because the *signal* shrinks by
the same w as the noise, and d′ is a ratio. The non-monotonicity is observable
only where absolute internal variability is not divided by a co-shrinking signal:

- in **estimation / reproduction / WTP** tasks, where the response *is* μ̂ (this is
  precisely why KLW's 2022 WTP model carries `var(log WTP) = γ_p²ν² + ν_c²`) **[P]**;
- under **rule B**, where the code divides by ν while the signal carries w;
- in **accumulator models**, where drift and diffusion noise are specified
  separately.

So: the non-monotonicity is real and is the correct statement about the internal
variable, but it does **not** predict non-monotonic choice behaviour in this
paradigm. Do not claim that it does.

### Q3 — What choice data actually identify

#### (A) KLW / posterior-mean-SD normalisation — the claim is **confirmed**

Symmetric case (shared prior, shared ν), log space. Numerator
`w·log(n₂/n₁) + log(p₂/p₁)`, denominator `w·ν·√2`:

  **d′ = log(n₂/n₁)/(√2 ν)  +  log(p₂/p₁)/(√2 w ν)**   **[A][N]**

- The **slope in log(n₂/n₁) is 1/(√2 ν)** — *w cancels exactly*. This is KLW's own
  γ = 1/(√2ν) (eq. 3.2 vs 2.5) **[P]**. **Choice accuracy identifies ν alone.**
- σ survives **only** in the threshold term, via 1/w. Equivalently, using
  `1/(wν) = 1/ν + ν/σ²`:

  **d′ = [log(n₂/n₁) + log p_R] / (√2 ν)  +  log(p_R)·ν / (√2 σ²)**   **[A][N]**

  first term = risk-neutral EV signal, second = the pure KLW risk-aversion bias.

The confirmation in the paper is exact: the indifference point is at
`X/C = (1/p)^{1/β}` (p. 14), i.e. **π ≡ e^{δ/γ} = p^{1+ν²/σ²}** (eq. 5.3, p. 29)
**[P][N]**, and KLW themselves invert (γ, π) → (ν, σ): slope gives ν, bias then
gives σ. With a *shared* prior the model is exactly just-identified.

So the user's claim (A) is **verified**, with one refinement: the σ-carrying "bias"
channel is not primarily "different priors for the two options" — it is the
**un-shrunk probability** `log p_R`, which is compared against shrunken magnitudes
and so gets divided by w. Different priors add a *second*, independent bias channel
(Q4).

Also worth knowing: if the prior is *also* shared and p₁ = p₂ (pure magnitude
comparison, no lottery), then σ has **literally no effect on choice at all** —
d′ = Δlog n/(√2ν) exactly, for every σ from 0.5 to 10⁶ **[N]**.

#### (B) Raw-evidence-SD normalisation — the claim is **verified**, with the ridge characterised

Same numerator, denominator `√2 ν`:

  **d′ = w·log(n₂/n₁)/(√2 ν) + log(p₂/p₁)/(√2 ν)**

The slope is now `w/(√2ν)`, i.e. the **effective noise is**

  **ν_eff = ν/w = ν + ν³/σ²**   **[A][N]**

strictly increasing in ν, strictly decreasing in σ. So ν and σ are entangled in
accuracy: d′ rises monotonically with σ at fixed ν (0.033 → 0.416 as σ goes
0.5 → ∞ at ν = 1.7) **[N]**.

**The iso-accuracy ridge.** Setting ν/w = c:

  **σ² = ν³/(c − ν)**,  equivalently  σ = ν^{3/2}/√(c − ν),  valid for **ν < c**

**[A][N]** (constant ν_eff to machine precision along the curve). Properties: ν and
σ move *together* along the ridge; σ → ∞ as ν → c⁻; and **ν is bounded above by the
observed psychometric width c**. That upper bound is the precise sense in which
rule B "pins ν".

#### The decisive comparison: the two rules give the SAME bias, different slopes

Indifference does not depend on the denominator, so **rules A and B have identical
PSEs**. Hence β — and therefore KLW's risk-neutral probability π — is
**rule-invariant**. Only the slope differs (1/(√2ν) vs w/(√2ν)). Consequently, in
the shared-prior log model, fitting the same choices under the two rules is an
exact reparameterisation:

  **ν_B = β·ν_A,  σ_B = β·σ_A,  β_B = β_A**   **[A][N]**

(identical curves to 2×10⁻¹⁶ over 4001 log-ratios). *The old rule reports ν and σ
deflated by β; it does not change the inferred shrinkage.* This is the clean
baseline — and it means the empirical divergence in `notes/pmc_refit_results.md`
(ν(7 CHF) = 0.42, w = 0.98 versus 1.49, w = 0.34) **cannot** be explained by the
rule change alone in the symmetric log model.

#### Why the real fits diverge anyway: rule A has an exact flat direction

The Flexible PMC breaks every symmetry: natural space, **separate** priors
(μ_R, σ_R) and (μ_S, σ_S), separate ν₁ ≠ ν₂, and p multiplying the SD under rule A
but not under rule B. In natural space, rule A is

  E[D] = p_R(w_R n_R + (1−w_R)μ_R) − p_S(w_S n_S + (1−w_S)μ_S),
  SD[D] = √((p_R w_R ν_R)² + (p_S w_S ν_S)²)

Now scale **both** weights by a common λ: w_k → λw_k. SD[D] scales by λ, and the
payoff-dependent part of E[D] scales by λ. Matching the constant part needs only

  p_R(1−λw_R)μ′_R − p_S(1−λw_S)μ′_S = λ·[p_R(1−w_R)μ_R − p_S(1−w_S)μ_S]

— **one equation, two unknowns (μ′_R, μ′_S), always solvable.** So there is an
**exact, zero-cost flat direction** in (σ_R, σ_S, μ_R, μ_S) at fixed ν **[A][N]**
(loss ≈ 2×10⁻¹⁸ nats/trial across λ = 0.3…1.0 on the 96-cell design; the same path
costs 2.4×10⁻² nats/trial under rule B). It is exact when ν₁ = ν₂ and approximate
otherwise (only two σ's are available to rescale four w's).

Profile likelihood over w under rule A, re-optimising everything else, on a
realistic 96-cell design with refit-like parameters:

| w_risky fixed at | 0.10 | 0.20 | **0.34** | 0.60 | 0.90 |
|---|---|---|---|---|---|
| loss (nats/trial) | 1.9e-4 | 1.1e-4 | 0 (truth) | 2.4e-2 | 1.1e-1 |

**[N]** — w can fall from 0.34 to 0.10 for ~1.6 nats total over 8335 trials
(indistinguishable), while rising to 0.60 costs ~200 nats. The valley runs
*downhill toward small w and small σ*, with μ_R, μ_S sliding to compensate. That
is exactly the "wandered along a flat direction, needed `--constrain` to converge"
behaviour reported for the HEAD refits, and it is why the refit sits at w = 0.34
with prior means pulled down to 4.90 / 9.95 CHF.

Cross-rule check: generating choices from published-like rule-B parameters
(ν₁ = 1.32, ν₂ = 0.42, w = 0.98, μ_R = 18.58, μ_S = 10.74) and refitting with rule
A recovers the choice probabilities to max|Δp| = 7×10⁻³ while landing at **ν₂ = 1.23
(2.9× higher) and w → 0** — the same direction and roughly the same magnitude as the
reported 0.42 → 1.49, w 0.98 → 0.34 **[N]**.

**Summary of Q3.** Both (A) and (B) as stated are correct *for the symmetric case*.
The ridge in (B) is σ² = ν³/(c−ν) with ν < c. But the operative degeneracy in the
actual application is the opposite one: **rule A, combined with separate priors,
has an exact flat direction along (σ_R, σ_S, μ_R, μ_S) that leaves accuracy *and*
bias untouched.** Rule B does not have it, because w does not cancel from its slope.

### Q4 — Asymmetric priors, separate p's

Risky option R (win prob p_R < 1), safe option S (p_S = 1).

**Log space** (`RiskModel`). Indifference `w_R log n_R + (1−w_R)μ_R + log p_R =
w_S log n_S + (1−w_S)μ_S` gives the log risk premium over risk neutrality:

  **log premium = (w_S/w_R − 1)·log n_S  +  [(1−w_S)μ_S − (1−w_R)μ_R]/w_R  +  ((1−w_R)/w_R)·log(1/p_R)**  **[A][N]**

Three structurally distinct channels:

| channel | source | signature |
|---|---|---|
| **weight ratio** w_S/w_R | unequal shrinkage of the two options | premium **varies with log n_S** → a *stake-size* effect |
| **prior-mean difference** | μ_S ≠ μ_R | constant offset, n_S-independent |
| **un-shrunk probability** | log(1/p_R) × (1−w)/w | the core KLW term |

With a shared prior (w_R = w_S = w, μ_R = μ_S) the first two vanish and

  **log premium = (ν²/σ²)·log(1/p_R)**,  since (1−w)/w = ν²/σ²

which is exactly KLW's Λ = p^{−(β⁻¹−1)} (p. 14) and π = p^{1+ν²/σ²} (eq. 5.3)
**[A][N][P]**. More generally, with shared w but different prior means:

  **log premium = (ν²/σ²)·[log(1/p_R) + μ_S − μ_R]**   **[A][N]**

**Natural space** (`FlexibleNoiseRiskModel`, the paper's model). With shared w:

  **n_R* = n_S/p_R + (ν²/σ²)·(μ_S − p_R μ_R)/p_R**   **[A][N]**

i.e. the premium is **additive in CHF and independent of n_S** — a constant
surcharge, not a proportional one. (Log space gives a proportional premium; this is
a real, testable difference between the two model families.) With unequal weights
a multiplicative w_S/w_R term reappears on top.

**So where does the risk-attitude shift come from? Both — but they are separable
by their n_S-dependence:**

1. **Degree of shrinkage** (1−w)/w = ν²/σ² is the *master scale factor*. It
   multiplies everything. Without shrinkage (σ → ∞, w → 1) there is no risk
   attitude shift at all, for any prior means **[P]** (KLW p. 14: risk neutrality
   "only in the case that β = 1 … ν = 0 … or σ is unboundedly large").
2. **The asymmetry term** it multiplies: `log(1/p_R) + μ_S − μ_R` (log) or
   `(μ_S − p_R μ_R)/p_R` (natural). With a shared prior this reduces to the
   un-shrunk probability and is always risk-*averse*.
3. **Unequal weights** w_S ≠ w_R add a *stake-dependent* premium
   (slope w_S/w_R − 1 in log n_S). This is the KLW-2022 "stake-dependent risk
   attitudes" mechanism.

Quantitatively, with the fitted natural-space prior means and shared-w algebra:
`μ_S − p_R μ_R` = **+0.52** for the published fit (10.74 / 18.58) and **−0.57** for
the refit (4.90 / 9.95) **[N]** — i.e. the fitted prior means very nearly *cancel*
against p_R in both fits (because the design's risky payoffs really are ≈ 1/p_R
times the safe ones), so the prior-mean channel is small and even sign-flips
between the two fits. **[?]** I could not evaluate the w_S/w_R channel for the real
fits: `notes/pmc_refit_results.md` reports a single w per fit, not σ_R and σ_S
separately. That is worth extracting from the traces, because with a near-cancelling
prior-mean channel the weight-ratio channel is likely carrying the risk attitude.

### Q5 — Does the prior variance amplify or damp a TMS-induced Δν?

**σ damps. Quadratically.**

Take the log model, rule A, shared prior, and evaluate at the EV-matched point
(log(n_R/n_S) = log(1/p_R), where the design's indifference cells sit). From
d′ = [log(n_R/n_S) + log p_R]/(√2ν) + log(p_R)·ν/(√2σ²), the first term vanishes and

  **∂d′/∂ν |_(EV-matched) = log(p_R) / (√2 σ²)**   **[A][N]**

Two remarkable properties, both verified against numerical differentiation **[N]**:

- it is **independent of ν** — the sensitivity does not depend on baseline noise;
- it is **∝ 1/σ²** — with p_R = 0.55: −1.691 at σ = 0.5, −0.423 at σ = 1,
  −0.106 at σ = 2, −0.026 at σ = 4. **A fourfold increase in prior SD shrinks the
  behavioural TMS effect 16-fold.**

Equivalently, on the risk-premium scale: premium ∝ ν²/σ², so
**∂(premium)/∂ν = 2ν/σ² × (asymmetry term)** — larger at high baseline ν, smaller
at large σ.

**When is a bigger prior variance better or worse for detecting the manipulation?**

- **Absolute effect size**: σ ↑ ⇒ effect ↓, as 1/σ². A subject (or a session) with
  a *tight* prior converts a given Δν into a large choice shift; a subject with a
  diffuse prior barely moves. In the σ → ∞ limit the manipulation is behaviourally
  **invisible in bias** and only flattens the psychometric curve.
- **Relative effect size**: ∂ln(premium)/∂ν = **2/ν**, which is **σ-free** **[N]**.
  So σ rescales the whole risk-attitude axis without changing the *fractional*
  sensitivity.
- **The accuracy channel**: under rule A the slope is 1/(√2ν), so σ does not modulate
  the flattening at all. Under rule B the effective noise is ν + ν³/σ², so
  ∂ν_eff/∂ν = 1 + 3ν²/σ² — again damped by σ, toward a floor of 1.

**Practical reading for the TMS paper.** The observation in
`notes/pmc_refit_results.md` §4 — that the cTBS effect "shows up as a bias, not as
extra randomness", with the bias channel reproducing +0.049 of a +0.052 total and
the randomness channel contributing +0.001 — is exactly what this predicts, and it
depends on the prior being *tight* (small σ, small w). It is also why the note's
statement **"prior attraction is therefore necessary: without a prior, a noise
increase can only flatten the psychometric curve"** is correct: with w = 1 the
entire bias term vanishes. Under rule A the flattening is *σ-independent* while the
bias is *∝1/σ²*, so the bias/randomness ratio of the observed effect is itself an
estimate of 1/σ².

---

## 3. Intuition (no algebra)

A Bayesian observer's estimate is a **weighted blend** of what it saw and what it
expected, with weight w = σ²/(σ²+ν²) on the signal.

1. **Adding encoding noise does two things at once**: it makes the raw signal
   jitter more, *and* it makes the observer trust that signal less. The second
   effect eventually wins. Past ν = σ the estimate stops getting noisier and starts
   getting *stereotyped*: it collapses onto the prior mean. The across-trial spread
   of the estimate peaks at ν = σ at a value of σ/2 and can never exceed that.

2. **You cannot see this in a choice task**, because both the signal and the noise
   shrink by the same factor w, and choice depends only on their ratio. What you
   see instead is: accuracy falls smoothly with ν, and *bias* grows smoothly with ν.
   The peak is a fact about the internal variable, not about behaviour.

3. **Risk aversion is what the shrinkage leaves behind.** The magnitudes get
   compressed toward the prior; the probability p does not, because it is known
   exactly. So the required payoff ratio inflates from 1/p to (1/p)^{1/w}. All of
   the risk aversion in this model rides on the single factor
   **(1−w)/w = ν²/σ²** — the noise-to-prior variance ratio.

4. **Hence the answer to the core question.** A TMS-induced increase in ν moves
   risk attitude in proportion to **1/σ²**. A subject with a tight prior shows a
   large behavioural effect; a subject with a diffuse prior shows nearly none, for
   the *same* neural change. The prior variance **damps**, and it damps quadratically.

5. **And the reason two fits of the same data disagree so wildly.** When the two
   options are given *separate* priors, the shrinkage weights and the prior means
   trade off against each other exactly: you can halve both w's and slide both
   prior means to compensate, and not a single predicted choice probability changes.
   The data cannot tell "mild compression toward a high prior" from "severe
   compression toward a low prior". Only the parameter priors decide — which is
   precisely what changed between the two fits.

---

## 4. What ν and σ do, under each rule

Symmetric case (shared prior, shared ν), log space. "Accuracy" = psychometric slope
in log(n₂/n₁); "bias" = PSE / risk-neutral probability π.

| | **Rule A (KLW / bauer HEAD flexible, `consistent_choice_noise=True`)** | **Rule B (bauer default, `ecc6454` flexible)** |
|---|---|---|
| decision-variable SD | w·ν·√2 (= √2·`posterior_mean_sd`) | ν·√2 |
| **slope (accuracy)** | **1/(√2ν)** — σ has *no* effect | **w/(√2ν) = 1/(√2·(ν+ν³/σ²))** — both matter |
| effect of ν on accuracy | strictly ↓, ∝ 1/ν | strictly ↓, ∝ 1/(ν + ν³/σ²) |
| effect of σ on accuracy | **none** | ↑ (σ↑ ⇒ w↑ ⇒ steeper) |
| **PSE / π (bias)** | π = p^{1+ν²/σ²} | **identical** — π = p^{1+ν²/σ²} |
| effect of ν on bias | ↑ risk aversion, ∝ ν²/σ² | ↑ risk aversion, ∝ ν²/σ² |
| effect of σ on bias | ↓, ∝ 1/σ² | ↓, ∝ 1/σ² |
| ν identified by | slope alone | slope **and** PSE jointly |
| σ identified by | PSE alone (given ν) | PSE **and** slope jointly |
| accuracy/bias degeneracy | none (symmetric case: just-identified) | ridge σ² = ν³/(c−ν), ν < c |
| **degeneracy with separate priors** (real model) | **exact flat direction in (σ_R,σ_S,μ_R,μ_S)** — severe | absent (w does not cancel from the slope) |
| relation between the two fits | ν_A, σ_A | **ν_B = w·ν_A, σ_B = w·σ_A**, same w |
| is it KLW? | **yes**, eq. (2.5) | no |

SD(μ̂) = wν is non-monotonic in ν under **both** — the rule changes only whether
that non-monotonicity reaches behaviour (it does not, under A).

---

## 5. Open / needs checking against the original paper

1. **[resolved, cite carefully]** All KLW claims above are from the OUP advance
   proof (pages 1–35). The mapping to the published 1979–2013 pagination is **not
   confirmed** — cite equation numbers (stable), not page numbers.
2. **[?]** KLW's model has **no** separate ν for safe vs risky and **no** separate
   priors. `prior_estimate='full'` (separate `risky_prior_mu/sd`,
   `safe_prior_mu/sd`) and `fit_seperate_evidence_sd=True` are de Hollander
   extensions. The identifiability results KLW state (just-identification of ν, σ
   from γ, π) hold for *their* symmetric model, and **do not carry over** to the
   asymmetric extension — that is where the flat direction lives. Worth stating
   explicitly in the paper's methods.
3. **[?]** KLW impose the boundary constraints γ ≥ 0 and π ≤ p (eqs. 5.1–5.2,
   p. 28), "required in order for there to exist values of σ² and ν² consistent
   with those coefficients". A risk-*seeking* subject admits **no** (ν, σ) at all.
   Whether the tms_risk fits ever sit in that region — and what the hierarchical
   prior does when they do — has not been checked.
4. **[?]** KLW's ν is on **log** payoff and is a Weber fraction (ν = 0.07 for
   Mosteller–Nogee, p. 13; σ = 0.26). The Flexible PMC's ν is in **CHF on the
   natural scale**, with a spline over magnitude. The reported sub-Weber behaviour
   (ν/n falling 0.21 → 0.048 over 7 → 112 CHF) is *not* directly comparable to
   KLW's ν, and no dimensionless comparison has been made.
5. **[?]** The natural-space model's risk premium is **additive in CHF and
   n_S-independent**, while KLW's is **multiplicative / scale-invariant** — their
   headline prediction, and their main empirical test (§3.1). Whether the tms_risk
   data prefer an additive or a proportional premium is a direct, cheap test that
   has not been run, and it bears on whether the natural-space parameterisation is
   the right one.
6. **[?]** Which rule should the paper report? The two are **not** a
   reparameterisation in the asymmetric natural-space model. Rule A is internally
   consistent and matches KLW; rule B (the published `ecc6454` fits) is not. But
   rule A is the one with the exact flat direction, so its ν and σ are only as
   trustworthy as the parameter priors. Recommendation: report rule A, and report
   the *identified* combinations (psychometric slope; π or the CHF risk premium;
   ν²/σ²) alongside the individual ν and σ.
7. **[?]** `posterior_mean_sd` is used unconditionally by the DDM/RDM front-ends
   but `BaseModel` still defaults to rule B, so **a static `RiskModel` fit and its
   DDM counterpart use different noise normalisations**. Any static-vs-DDM
   parameter comparison currently mixes the two conventions; the static ν is
   deflated by w relative to the DDM's.
8. **[?]** The tms_risk submodule (`4cd98a4`) predates `62eaf11`, so
   `consistent_choice_noise` is not even available there. Any plan to fit KLW-
   consistent log-space PMC models needs the submodule advanced to `origin/main`.

---

## 6. Verification log

Scripts in `/private/tmp/claude-1763273667/.../scratchpad/verify{,2,3,4,5}.py`,
run with `~/mambaforge/envs/tms_risk/bin/python`. `verify5.py` imports and executes
**bauer's own** `get_posterior` / `get_diff_dist` / `posterior_mean_sd` /
`cumulative_normal` through pytensor.

**Verified numerically [N]** — all passed:

| claim | check |
|---|---|
| SD[μ̂ \| n] = w·ν | 4×10⁶-draw simulation: 0.939704 vs 0.939836 |
| `post_sd**2/nu == w*nu == posterior_mean_sd` | machine precision |
| f′(ν) = σ²(σ²−ν²)/(σ²+ν²)² | vs `np.gradient`, 4×10⁵ grid |
| f′ > 0 for ν<σ, = 0 at ν=σ, < 0 for ν>σ; max = σ/2 | sign tests, 10⁵ points |
| SD(μ̂) ≤ σ/2 for all ν ∈ [10⁻⁴, 10⁴] | grid |
| 1/(wν) = 1/ν + ν/σ² | machine precision |
| **rule A is the exact generative model**, rule B is not | 2×10⁶-draw choice simulation: sim 0.17162, A 0.17180, B 0.26996 |
| d′_A = Δn/(√2ν), σ-free for σ ∈ {0.5, 2, 10, 10⁶} | exact |
| d′_B strictly increasing in σ | 500-point monotonicity |
| ν_eff = ν/w = ν + ν³/σ²; ridge σ² = ν³/(c−ν) | constant ν_eff along the curve |
| log premium = (ν²/σ²)(log(1/p_R) + μ_S − μ_R) | vs numeric indifference solve |
| natural premium = (ν²/σ²)(μ_S − p_R μ_R)/p_R, n_S-free | tested at n_S = 7, 14, 28 |
| asymmetric-w decomposition (3 channels) | vs numeric solve, n_S = 7…56 |
| ∂d′/∂ν = log(p_R)/(√2σ²) at EV-match, ν-free, ∝1/σ² | central differences, ν ∈ {0.3…5}, σ ∈ {0.5, 2, 4} |
| ∂ln(premium)/∂ν = 2/ν, σ-free | central differences |
| **bauer + `consistent_choice_noise=True` = KLW eq. (2.5)** | executed bauer code; agrees to 4×10⁻⁹ (pytensor `erf`) |
| bauer default (raw ν) ≠ KLW eq. (2.5) | 0.8103 vs 0.8271 at KLW's own (ν, σ) |
| indifference at X/C = (1/p)^{1/β} | numeric PSE 0.584210 vs 0.584212 |
| KLW eq. (5.3) π = p^{1+ν²/σ²} | machine precision |
| **ν_B = β·ν_A, σ_B = β·σ_A ⇒ identical curves** | max\|Δp\| = 2×10⁻¹⁶ over 4001 log-ratios |
| **exact flat direction under rule A** (ν₁=ν₂) | loss ≈ 2×10⁻¹⁸ nats/trial, λ = 0.3…1.0 |
| same path is *not* flat under rule B | 2.4×10⁻² nats/trial |
| profile over w under rule A: 0.10–0.34 free, 0.60 costly | 1.9×10⁻⁴ vs 2.4×10⁻² nats/trial |
| rule-A refit of rule-B data ⇒ ν₂ 2.9× higher, w → 0 | max\|Δp\| = 7×10⁻³ |

**Verified analytically [A]**: every formula above; the derivations are elementary
Gaussian algebra and each was cross-checked numerically as listed.

**Confirmed in the KLW paper [P]**: eqs. (1.3), (2.1)–(2.5), (5.1)–(5.3); p. 11
(p known, shared ν), p. 12 (two free parameters, no late noise), p. 13 (ν = 0.07,
σ = 0.26), p. 14 (indifference at (1/p)^{1/β}, Λ = p^{−(β⁻¹−1)}, risk neutrality
only at β = 1), pp. 28–29 (identification via γ and π, boundary constraints).
I read pages 11–14 and 28–29 directly.

**Not verified [?]**: everything in §5, plus the quantitative attribution of the
real fits' risk attitude to the w_S/w_R channel (needs σ_R, σ_S from the traces —
`pmc_refit_results.md` reports only a single w).
