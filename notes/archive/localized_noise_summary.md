# Is the cTBS effect "just randomness on the left arm"?

Analysis: `tms_risk/behavior/scripts/analyze_localized_noise.py`
Figure:   `tms_risk/behavior/scripts/plot_localized_noise.py` → `notes/figures/localized_noise.pdf`
Numbers:  `notes/data/localnoise_stats.txt` (+ the `localnoise_*.tsv` source-data files)

## What the data say

**1. The effect really is confined to the left arm, and only when the risky option
came second.** ΔP(chose risky), IPS − vertex, per risky/safe-ratio bin:

| ratio bin | 20% | 32% | 44% | 56% | 68% | 80% |
|---|---|---|---|---|---|---|
| Risky second | +0.056 | **+0.137** | +0.056 | −0.006 | +0.034 | +0.028 |
| Risky first | −0.025 | +0.016 | +0.042 | +0.032 | −0.004 | −0.010 |

**2. Sorted by the size of the risky payoff, the effect is even sharper.** Risky-second
trials, ΔP by risky payoff: **+0.160** (7–17 CHF, p = 0.001), then +0.039 / +0.046 /
+0.034 / +0.029. In the smallest band, P(risky) goes from 0.349 [0.271, 0.427] after
vertex to 0.509 [0.411, 0.607] after IPS — i.e. **exactly chance** (t(32) = 0.18,
p = 0.86). That band is the one the nPRFs are tuned to (preferred-numerosity
IQR [6, 10]). Contrast vs the four larger bands: +0.106 ± 0.054, t(32) = 1.96,
p = 0.059 two-sided (0.030 one-sided; the direction was predicted a priori).

## Two things that did *not* work out

**3. The probit is not statistically refuted by these data.** A proper posterior
predictive check on the fitted `probit_order` model: every observed per-bin Δ falls
inside the 95% predictive interval, and an omnibus test on the risky-second profile
gives p = 0.389. The fitted probit's own Δ profile is already left-weighted
(+0.085, +0.083, +0.068, +0.046, +0.026, +0.007), because a slope decrease *plus* an
indifference shift is exactly what produces a left-weighted profile.

**4. "Pure randomness" alone does not fit either.** A flattening pulls choices toward
0.5 from *both* sides, so with the indifference point held fixed it must produce
**negative** Δ wherever baseline P > 0.5. Grid-searching a magnitude-dependent
noise-only generator (σ scaled by 1 + a·exp(−(n_risky − 7)/τ), indifference fixed)
against the observed profile lands at the edge of the grid and still misses badly:

```
observed  +0.056  +0.137  +0.056  -0.006  +0.034  +0.028
best fit  +0.065  +0.032  +0.003  -0.015  -0.014  -0.012
```

The data are positive at every bin. There is a genuine shift component.

Also, no credible violation of scale invariance: adding TMS × log(safe payoff) terms
to the probit gives e0 = −0.32 [−0.75, +0.07] and e1 = +0.33 [−0.17, +0.83].

## What is defensible

**5. Local noise does generate a spurious preference shift — about a third of the
observed one.** Simulating choices where cTBS changes *only* the noise (indifference
point identical by construction) and applying the paper's read-out returns
ΔRNP = +0.006 … +0.026 across five simulations, versus +0.054 [+0.019, +0.091] in the
real data (my PyMC re-fit reproduces the paper's 52.4% → 57.9%). The slope reduction
recovers correctly (−0.10 … −0.37 vs −0.23 observed).

So: the probit's "increased risk-seeking" is *partly* an artefact of a read-out that
cannot express a locally-confined change — but not entirely. The honest claim is
**"the effect is local, not global"**, not "the shift is not real".

## Figure caption (draft)

**Figure X. The cTBS effect on choice is confined to small risky payoffs.**
**a**, Proportion of risky choices as a function of the risky/safe payoff ratio, for
trials in which the risky option was presented second (n = 35). Points are
within-subject means ± SEM; lines and bands are the posterior mean and 95% credible
interval of the hierarchical probit model, aggregated over trials exactly as the data
are. The two conditions separate on the left arm of the curve and converge on the
right. **b**, Difference in choice proportions (IPS − vertex) per ratio bin; filled
symbols, risky option second; open symbols, risky option first (error bars 95% CI).
Curves show what each mechanism predicts: a pure change in choice consistency (slope
of the vertex fit replaced by the IPS slope, indifference point held fixed), a pure
change in preference (indifference point replaced, slope held fixed), a
magnitude-dependent noise increase with the indifference point held fixed, and the
fitted probit. A pure flattening of any kind predicts negative differences wherever
baseline choice proportions exceed 50%; the data show none. **c**, The same difference
binned by the size of the risky payoff. The shaded band marks the interquartile range
of nPRF preferred numerosities (Fig. 3B). For the smallest risky payoffs, cTBS moves
choices from 35% to 51% risky, i.e. to chance. Grey line and band, fitted probit
(posterior mean and 95% CI). **d**, Posterior of the change in risk-neutral
probability estimated by the hierarchical probit. Red, real data; grey, five
simulations in which cTBS increased noise on small payoffs and the indifference point
was identical between conditions by construction. Local noise alone produces a
spurious risk-seeking shift of roughly a third of the observed size.

---

# Follow-up: the decomposition done inside the fitted Flexible PMC

Analysis: `tms_risk/behavior/scripts/decompose_pmc_channels.py`
Figure:   `tms_risk/behavior/scripts/plot_pmc_channels.py` → `notes/figures/pmc_channels.pdf`
Numbers:  `notes/data/pmc_channels_stats.txt`, `pmc_channels_by_{ratio,nrisky}.tsv`

## The result

The PMC turns one knob for cTBS (the noise function ν(n)) but that knob feeds two
channels of its own choice rule `P = Φ(m/s)`: the perceived advantage `m` (bias,
via prior shrinkage) and the spread `s` (randomness). Evaluating `m` and `s` per
trial under both conditions from the fitted `flexible2.6` posterior:

| Channel (risky-second trials) | mean ΔP | 95% CrI | mean \|ΔP\| |
|---|---|---|---|
| Full model | +0.0300 | +0.008 to +0.058 | 0.097 |
| **Bias channel alone** | **+0.0312** | +0.009 to +0.064 | 0.094 |
| Randomness channel alone | −0.0028 | −0.007 to +0.002 | 0.023 |

The bias channel alone reproduces the full model. The randomness channel is
slightly *negative* — the same signature the data reject.

**So prior attraction is necessary**: in this model it is the entire route by which
the noise increase reaches behaviour, not one mechanism among several. Strip the
prior and noise can only flatten; flattening predicts a negative right arm.

Two consequences for the text:
- The claim that the effect "reflects a flattening of the psychometric function
  rather than a shift in underlying preferences" is contradicted by the paper's own
  best-fitting model.
- The model under-predicts the observed effect by roughly half (+0.030 vs +0.051
  mean) and misses its concentration at the smallest payoffs (+0.059 vs +0.160).

Validation: reconstruction exact to 4e-9 against the model's own stored `p`;
posterior predictive grand mean 0.559 vs observed 0.560.

## Reproducibility hazard — read before regenerating any PMC figure

**`model-flexible2.6_trace.netcdf` cannot be re-evaluated with the current
`libs/bauer` HEAD, and it fails silently.** All free parameters are present, the
design matrices and stimulus columns match the stored `constant_data` exactly, and
the subject-level parameters recompute to machine precision — but the predicted
choice probabilities are off by +0.12 (grand mean 0.65 vs observed 0.53). No error.

Two changes since the fit, neither renaming a parameter:

1. **Likelihood.** Commit `b66c806` (2026-04-03) changed the `'payoff'` branch of
   `_get_choice_predictions` from `diff_sd = sqrt(ν1² + ν2²)` to a
   posterior-variance- and probability-scaled version.
2. **Noise composition.** At the fitting commit, `_get_trialwise_evidence_sd` built
   the first option's noise with `stack([parameters[l1] for l1 in labels1])` for
   *both* spline terms — the memory coefficients twice. So in the published fits the
   perceptual noise function never entered the first-presented option, contrary to
   the Methods' ν₁ = ν_perceptual + ν_memory. HEAD uses `labels2` for the second term.

Faithful reconstruction requires **`bauer@ecc6454` (2024-11-05)**:

```bash
git -C libs/bauer worktree add --detach /tmp/bauer_ecc6454 ecc6454
python -m tms_risk.behavior.scripts.decompose_pmc_channels --bauer_path /tmp/bauer_ecc6454
```

The script aborts if the posterior predictive grand mean is off by more than 0.02,
so a wrong pin cannot pass unnoticed. Anything reading these traces — Fig 4B, 4C,
Fig 5, Supplementary Text 1, the ELPD table — should be regenerated against a
pinned commit, and point 2 needs a decision: refit with the intended noise
composition, or document what was actually fitted.

---

# When the corrected refits land — checklist

Runner: `/data/run_waves.sh` on `ssh sciencecloud`, detached with `setsid nohup`, so it
survives disconnection and proceeds from wave 1 (5-spline, the paper's model) to
wave 2 (6-spline) on its own. Logs in `/data/logs/`; progress with

```bash
ssh sciencecloud 'for f in /data/logs/*.head.log; do printf "%-30s " "$(basename $f .head.log)"
  tr "\r" "\n" < "$f" | grep -oE "[0-9]+%[^│]*" | tail -1; echo; done'
```

Traces land in `/data/ds-tmsrisk/derivatives/cogmodels.noisefix/` as
`model-flexible2[.6]_noisefix[_null|_memory|_perception].head_trace.netcdf`, each with
`posterior.attrs['tms_risk_bauer_commit']` stamped. Then, in order:

1. **Convergence.** Divergences are printed at the end of each log; check r_hat too.
   The published `flexible2` had 204 divergences in 20 000 draws.
2. **Rebuild Table 1** on the refits — they already carry `log_likelihood`, so no pin
   is needed and the gate should pass trivially:
   `python -m tms_risk.behavior.scripts.model_comparison_table --cogmodels_dir derivatives/cogmodels.noisefix`
   Question to answer: does "TMS affects both noise terms" still win once the
   perceptual function actually reaches the first-presented option?
3. **Re-run the channel decomposition** with `--bauer_path libs/bauer` (no pin needed
   for a fresh posterior) on `flexible2_noisefix`. Question: does the bias channel
   still carry the whole effect, and does the *memory* term now show a credible cTBS
   effect (it does not in the published fit)?
4. **Re-extract the parameters** and regenerate `pmc_parameters.pdf` /
   `pmc_mechanism.pdf`. Question: with the corrected composition, is ν₁ really
   ν_perceptual + ν_memory, and where does the noise increase sit?
5. **Compare 5- vs 6-spline** (wave 1 vs wave 2) — the Methods say 5; the 6-spline
   family exists only as an exploratory variant. If they agree, say so and move on.

Only after all five should any manuscript text change. Open items that do not depend
on the refits: the three permuted Weber row names in Table 1, and the Fig 4B panel
titles.
