# Why the `power` form says "log σ is linear in log x"

Written 2026-08-28, after the anchor-grid refit. Companion to the module
docstring in `libs/bauer/bauer/models/anchor_noise.py`.

## The setup: σ here is already a log-space quantity

The log-space observer represents a payoff `x` on a log scale. Its internal
evidence for an option is

    e = log x + ε,        ε ~ N(0, σ(x))

so **σ is the SD of the log-payoff percept, in log-CHF units**, not an SD in
CHF. This is the single fact that makes everything below feel like it has one
log too many: there is one log in the *representation*, and the noise function
lives on top of it.

A useful rule of thumb for reading σ: for small noise, `exp(ε) ≈ 1 + ε`, so
σ = 0.2 means roughly ±20% uncertainty about the payoff. σ is essentially a
**coefficient of variation**.

## What σ in log units means in CHF

Push the percept back to CHF with the delta method. If `x̂ = exp(e)` then

    SD(x̂) ≈ x · σ(x)                                             (1)

So the natural-space noise is the log-space noise times the payoff.

Three consequences, and they are the whole reason the forms are named the way
they are:

| σ(x) in log units | ⇒ SD in CHF | name |
|---|---|---|
| σ = w (constant) | `w·x` | **Weber's law** — constant relative noise |
| σ = a + b·log x | `x(a + b log x)` | `affine` |
| σ = C·x^b | `C·x^(1+b)` | `power` |

**In log space, Weber's law is a flat noise function, b = 0.** Not slope 1.
Slope 1 is the Weber reference for a *natural-space* observer, whose σ is in CHF
and for whom Weber means σ ∝ x. Getting this backwards is easy; an earlier draft
of Fig 4b drew the slope-1 line and called it Weber.

## The derivation you asked for

"Power law" means σ is a power function of the payoff:

    σ(x) = C · x^b                                               (2)

Take logs of both sides:

    log σ(x) = log C + b · log x                                 (3)

which is linear in `log x` with slope `b`. That is all the `power` form asserts.
The name is not "log of a log of something"; it is a power law in σ, and (3) is
just the standard fact that a power law is a straight line on log–log axes.

Where the "double log" feeling comes from is that the *quantity being raised to
a power* is itself an uncertainty about a log-scale representation. Two logs
appear, doing different jobs:

- the **first** log is the observer's representation (`e = log x + ε`);
- the **second** log is the link that turns (2) into the straight line (3).

## How the anchor parameterisation implements it

Anchors are payoffs `a₁ … a_K` at which the free parameters *are* the noise SD.
Write `θ_k = log σ(a_k)`, and let `B(x)` be the partition-of-unity interpolation
weights in the coordinate `u = (log x − log a₁)/(log a_K − log a₁)`. Then

    value link (affine, genweber, spl3, spl5):  σ(x) = Σ_k B_k(x) · exp(θ_k)
    log link   (power):                         σ(x) = exp( Σ_k B_k(x) · θ_k )

With two anchors, `B = [1−u, u]`, so the log link gives

    log σ(x) = (1−u)·θ₁ + u·θ₂

and since `u` is affine in `log x`, `log σ` is affine in `log x` — equation (3),
with

    b = (θ₂ − θ₁) / (log a_K − log a₁) = log(σ₂/σ₁) / log(a_K/a₁)   (4)

Equation (4) is how the exponents in the results tables are computed: take the
fitted noise at the two ends and divide the log ratio by the log payoff range.
It applies to the spline forms too, as a summary of a curve that was not
constrained to be a power law — which is why `spl3` and `spl5` can be quoted as
"b ≈ 0.29" even though nothing forced them into that shape.

`affine` and `power` share the design matrix `B` and differ *only* in whether the
convex combination happens on σ or on log σ. Under the old spline code that
difference was an accident of the softplus link; here it is a declared choice.
Empirically the two are indistinguishable on this dataset (< 3 ELPD apart), which
is unsurprising: over 7–112 CHF the fitted σ changes by less than a factor of
two, and any two smooth two-parameter forms agree over such a short range.

## What the fitted exponents mean

For the `n1n2` placement, vertex condition:

| | b(σ_n1) | b(σ_n2) |
|---|---|---|
| affine | 0.04 | 0.35 |
| power | 0.05 | 0.33 |
| genweber | 0.04 | 0.32 |
| spl3 | 0.03 | 0.29 |
| spl5 | 0.09 | 0.25 |

Read through (1): the first-presented option's noise is essentially **Weber**
(b ≈ 0.05, constant ~20% relative uncertainty). The second-presented option is
**super-Weber**, b ≈ 0.3, meaning its relative uncertainty itself grows as
x^0.3 — from about 10% at 7 CHF to about 22% at 112 CHF — equivalently
SD in CHF ∝ x^1.3.

Note the direction: the option seen at the moment of choice is the *less* noisy
one at small payoffs and catches up at large ones, while the remembered option
carries a roughly constant relative noise throughout.
