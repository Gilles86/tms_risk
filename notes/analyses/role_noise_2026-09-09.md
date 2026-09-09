# Are risky options inherently noisier? Asked, fitted, and no

**The proposal.** Risky options must be combined with a probability, so perhaps
their magnitude is encoded less precisely than a safe option of the same payoff.
Implement as an interaction of the noise channels with `risky_first`, regularised
toward zero.

**It already exists.** `fit_anchor.py --role_scale free` is exactly this in its
main-effect form: `ν_k *= exp(δ · 1[option k is risky])`, one free hierarchical
parameter with a N(0, 0.5) prior. The literal `n1 × risky_first` version is its
unconstrained generalisation, and `n1n2x` is the saturated superset that also
carries role × cTBS. All three have been fitted.

**Every fit puts δ at zero** (`log_risky_noise_scale_mu`, read from the traces):

| fit | δ | 95% CrI | implied |
|---|---|---|---|
| `percmem.free` | −0.222 | [−0.630, +0.144] | risky 20% *less* noisy |
| `n1n2.free` | −0.012 | [−0.355, +0.314] | −1% |
| `n1n2x.free` | +0.078 | [−0.392, +0.545] | +8% |
| `nullind.free` | +0.066 | [−0.288, +0.420] | +7% |

ELPD against the same model without `free`: −0.1, +0.6, +0.6, +3.6 — nothing.
Forcing the role scaling to the expected-value form (`--role_scale ev`, i.e.
σ evaluated at p·x) is **rejected by 12–17 nats**.

## Three reasons it was never going to work

**1. The premise does not follow from this observer.** The probability never
touches the magnitude channel: evidence is log n_k, the prior is on log n_k, and
p enters only as the decision threshold log(p₂/p₁), known exactly. "Multiplying
by a probability adds noise" would be an additive variance on the risky option's
decision variable *after* shrinkage — which is a different object from a
multiplier on ν, does not pass through w = σ_p²/(σ_p²+ν²), and therefore changes
the slope without producing any bias or order asymmetry.

**2. δ is badly confounded with the noise function's shape.** Over presented
options, corr(role, log-payoff coordinate) = 0.57, and a risky payoff coincides
with a safe one on 8% of trials. Fisher information at the fitted parameters:
corr(δ, perceptual slope) = +0.90, corr(δ, log risky prior SD) = **+0.94**.
SE(δ) is 0.014 with everything else fixed and **0.64 with everything free** — a
45-fold inflation. Scaling ν_risky and σ_p,risky together leaves the shrinkage
weight almost unchanged: the same degeneracy that sank `percpsd`.

**3. The implied direction is wrong-signed.** Forward simulation on the real
paradigm (observed: ΔP risky-second +0.053, slope asymmetry −0.106):

| variant | ΔP (1st / 2nd) | slope asymmetry | ΔIP 2nd |
|---|---|---|---|
| `perc` as fitted | +.006 / +.005 | −.003 | −.011 |
| role main effect, δ = +0.5 | +.007 / +.007 | −.006 | −.013 |
| cTBS hits the **risky** option harder | −.038 / −.032 | **+.029** | **+.040** |
| cTBS hits the **safe** option harder | +.014 / +.013 | −.020 | −.026 |

A role main effect gives ≤15% of the asymmetry even at δ = +0.5, a 65% noisier
risky option the data reject. And the interaction *in the proposed direction*
flips the sign of ΔP, of the indifference-point shift and of the asymmetry. The
direction the data would want is the opposite one — which is approximately what
payoff-indexing already does for free, and which `n1n2x` estimates at zero.

## What to do with it

One defensive sentence in the paper, written from the fits already on disk: a
role-indexed noise multiplier, with or without its own cTBS contrast, is null,
and indexing the noise function by expected value rather than payoff is rejected
by 12–17 nats.

**Provenance to fix:** these fits used the shipped clone
`/scratch/gdehol/bauer_role`, whose `COMMIT` file is **empty**, so every
`.free`/`.ev` trace carries `tms_risk_bauer_commit = ''`. Write the commit in
and note it in `notes/PROVENANCE.md`.
