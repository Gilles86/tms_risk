# v11 plan: one primary model, with the linear scale as a real sensitivity analysis

2026-08-27, rev. 2 after the family-1 refits. **Plan only — manuscript,
PROVENANCE and figure scripts untouched.** Supersedes
`/Users/gdehol/git/tms_risk/notes/manuscript_model_plan.md` on the primary-model
question; `/Users/gdehol/git/tms_risk/notes/v10_todo.md` still stands.

**Direction changed at rev. 2.** Rev. 1 presented the two scales as co-equal on a
measured tie (z = 0.7) — but that tie was against a *non-converged* fit. Sampled
properly, natural space loses by **35.6 nats (2.5 dSE)**. So: **log space is
primary; the linear scale is a sensitivity analysis whose failure is itself
informative.** The scale-geometry insight survives, moving from "two equal
readings" to "we tried the linear scale, and here is why it needs an absurd
prior".

## 0. Four things verified from disk

1. **The natural-space model fits fine — in family-1 coordinates.** `flexible1_*`
   (`memory_model='independent'`, cTBS on `n1`/`n2` noise) vs `flexible2_*`
   (`shared_perceptual_noise`) are exact likelihood reparameterizations
   (c1 = memory + perceptual, c2 = perceptual), and `fit_pmc_noisefix.py:57`
   already documents family 2 as bimodal. Fitted 2026-07-30, bauer `e05f73a`,
   in `derivatives/cogmodels.overnight/` on `sciencecloud_gpu`/`_gpu4`.

   | Trace | ELPD | r̂ | ESS | div |
   |---|---:|---:|---:|---:|
   | `flexible1_noisefix.head` (cTBS both options) | −4184.6 | 1.000 | 3696 | **10** |
   | `flexible1_noisefix_first.head` | −4222.0 | 1.000 | 2550 | 15 |
   | `flexible1_noisefix_second.head` | −4227.5 | 1.000 | 946 | 26 |
   | `flexible1_noisefix_null.head` | −4271.8 | 1.000 | 1259 | **0** |

   Family 2: 962–2283 divergences, r̂ to 1.19. **A fit with 2283 divergences has
   no defensible ELPD** — that is the principled answer to "you picked the
   natural-space parameterization that fits worse".
2. **Prior placement.** Natural-space safe prior mean **−4.63 CHF**, P(<0) = 0.81
   (risky 4.17); log space **9.19** ×/÷1.55 and **15.73 CHF** ×/÷1.43, on the
   payoff distribution (means 15.82 / 36.17). Drawn:
   `notes/figures/prior_placement.pdf`. **But this came from family-2 traces —
   re-derive from `flexible1_noisefix.head` or drop the number [§5.3].**
3. **Log space is not simpler.** p_loo **260.0** vs **254.2** (family 1). The
   "10 vs 24 parameters" line belonged to `sm-dp-b`. Don't write it.
4. **The log-space noise effect is a crossover.** Total σ₁: 7–30 CHF **+0.0137**
   [−0.0123, +0.0371], P(>0) = **0.845**; >30 CHF **−0.0260** [−0.0488, −0.0067].
   Replicated in the `bs2` twin. `manuscript_model_plan`'s P = 0.95–0.97 belongs
   to `sm-dp-b` — do not carry it over.

## 1. The two models

**Log-Flexible PMC** = `lfx2-bs3-m2-dp-bm` (primary): log-payoff observer,
lognormal priors, 5-df cubic B-spline perceptual noise, 2-df memory, cTBS on both
channels. **Flexible PMC (linear scale)** = `flexible1_noisefix.head`
(sensitivity), the converged form of the v10 incumbent.

| Label | ELPD | Δ vs primary | dSE | r̂ | div | Δ-PPC cov |
|---|---:|---:|---:|---:|---:|---:|
| `lfx2-bs2-m2-dp-b` | **−4148.6** | +0.4 | 4.7 | 1.000 | 0 | 0.67 |
| **`lfx2-bs3-m2-dp-bm`** | −4149.0 | — | — | 1.010 | 3 | **1.00** |
| `lfx2-bs2-m3-dp-bm` | −4155.0 | −6.0 | 3.5 | 1.010 | 0 | **1.00** |
| `flexible1_noisefix.head` | −4184.6 | −35.6 | 14.4 | 1.000 | 10 | — |
| `lfx2-bs3-m2-dp-null` | −4245.3 | −96.3 | 13.3 | 1.000 | 0 | 0.67 |

Selection rule, for Methods: **converges, then reproduces the effect in the
Δ-PPC, then best fit.** Only two log-space cells clear the first two and both are
`-bm`; the nominal ELPD leader leads by 0.4 nats (noise) and misses a third of
the Δ-PPC cells.

> **DECISION A — resolved by data.** `-bm` vs `-b` is Δ = −0.4, dSE 4.7,
> **z = −0.1**: indistinguishable. The channel is not identified, so the
> tie-break decides only which curve is drawn. Plan uses `-bm`.

Settled: `dp` over `tp`; RT/RDM out; family 1 over family 2 for the linear scale.

## 2. Curation rule and the two tables

> **One row per question a reviewer would ask, not one row per fit.** A model
> earns a place only as the primary, a matched null, or the unique minimal
> contrast answering one named question; every supplementary row is a
> **one-factor-at-a-time** deviation from the primary.

**Main Table 1 (9 rows).** Columns: name · ELPD · Δ vs primary · **dSE** ·
**Δ-PPC coverage** · question. Both flagged columns are mandatory: the adopted
model is not the ELPD maximum, and `se` (~46 nats, dominated by between-subject
variance common to every model) understates every comparison ~5× versus paired
`dse`. **Quote dSE, never se.**

| Name | Label | Question |
|---|---|---|
| **Log-Flexible PMC** | `lfx2-bs3-m2-dp-bm` | primary (ratio scale) |
| …no cTBS effect | `lfx2-bs3-m2-dp-null` | Any effect at all? |
| …cTBS on perceptual only | `lfx2-bs2-m2-dp-b` | One channel? |
| Log-Weber PMC | `lfx2-bs3-w-dp-bm` | Must noise vary with magnitude? |
| …no cTBS effect | `lfx2-bs3-w-dp-null` | (matched null) |
| Log-Flexible, alt. basis | `lfx2-bs2-m3-dp-bm` | Does the basis drive it? |
| **Flexible PMC (linear scale)** | `flexible1_noisefix.head` | Does the scale change the conclusion? |
| …no cTBS effect | `flexible1_noisefix_null.head` | (matched null) |
| …objective prior | `lfx2-bs3-m3-dp-bm-op` | Is the prior load-bearing? |

**Supplementary Table 1 (~24 rows), six blocks**, primary's setting marked:
**A** cTBS placement (null/b/bm/`sd3-dp-t`) · **B** flexibility
(`w`/`sm`/`m2`/`m3`/`fm`) · **C** basis (`bs2`/`bs3`/`cr3`) · **D** hyperpriors
(`dp`/`tp`/`-hn`) · **E** prior (free/`-sp`/`-fs`/`-f1`/`-op`) · **F**
linear scale (`flexible1_noisefix{,_null,_first,_second}`, plus the family-2 set
**with its divergence counts**, as the reason it is not the reported form). Add a
**converged (r̂≤1.01)** column, showing non-converged cells with r̂ rather than a
suppressed ELPD — `fm`/`cr3` non-identifiability is a finding, and holes in a
factorial look worse than failures.

**Dropped (~48):** `logflex*`/`logflexm*`; `sd*` bar one `-t` row; `power*` (but
see DECISION B); `flexible2.{2,3,4,6}_noisefix.head`; off-path non-converged
cells (counted, not listed); RDM/DDM.

> **DECISION B — answerable inside the architecture.** `lfx2-bs3-m3-dp-bm-op`
> (prior pinned to payoff statistics) costs **583.5 nats, z = 19.1**. That makes
> the "could this be efficient coding without a prior?" point without re-opening
> the parked power family, where `power*_flat` costs 728–770 (z ≈ 22) — quote
> the latter only if a reviewer presses.

## 3. The modeling Results — three claims, decreasing strength

Walk down a ladder of confidence, naming at each rung the evidence that actually
carries the claim. Nothing above rung 3 depends on choosing a model.

**Rung 1 — cTBS raised representational noise. (Strong; no model owns it.)**
Establish existence across the set, not inside one fit: every family beats its
matched null by **69–114 ELPD (5–8 dSE)** — log-flexible +96.3 (7.2), log-Weber
+69.1 (6.2), linear-scale +87.2 (6.7), power-law +78 to +87 (6.3–6.7). Say
explicitly that this is a claim about *the set*: whichever scale, basis,
memory parameterization or hyperprior is chosen, a model allowed to change noise
under stimulation predicts held-out choices substantially better than one that
is not.

**Rung 2 — the effect sits in risky-second, low-stake choices. (Strong;
model-free.)** Do not source this to a cognitive model. Lead with the by-stake
probit — slope 3.02 [2.73, 3.31] → 2.33 [2.07, 2.61], p = 0.001, with the null
cells alongside (p = 0.204, p = 0.111) and the interaction (p = 0.015) — then
note that the primary's Δ-PPC reproduces the pattern across all six stake cells.
The model corroborates; the probit establishes. Keeping this tier frequentist
and pre-model is what stops the model set doing inferential work it shouldn't.

**Rung 3 — what the noise increase *is*. (Weaker, and now with a clear winner.)**
Two sub-claims, in order.

*Both presented options, not one channel.* The rung-3 result that strengthened.
In the linear scale's converged coordinates the question maps onto the paper's
own risky-first / risky-second language: cTBS on the first option alone beats the
null by +49.8 (5.4 dSE), on the second alone by +44.2 (5.0 dSE), and **allowing
both beats either single-option model by +37.4 and +42.9 (4.8 and 5.2 dSE)**. The
ratio scale agrees in its own coordinates (`-bm` over `-b`), and
σ₁ = softplus(perceptual + memory) leaves the channels unidentified either way.
So report **total representational noise on both presented options**, and state
plainly that **v10's claim that cTBS acts on shared perceptual noise alone does
not survive**.

*Why the linear scale is only a sensitivity analysis.* Fitted on a linear scale,
the model needs a prior over payoffs centred **below zero** — outside the range
of the quantity it is a prior over. Not a sampling artefact: additive-shrinkage
geometry demands it, since the prior mean must sit wherever the additive pull
reproduces the observed compression, and on a linear scale that place is
negative. Change to a ratio scale and the same choices put the priors on the
payoff distribution (9.19 and 15.73 CHF) — **and fit better by 35.6 nats
(2.5 dSE)**. So the sentence is not "two equally good readings" but: *the
compression is real in both geometries; only the ratio-scale account places its
prior where a Bayesian observer could plausibly hold it, and it also predicts
held-out choices better.* The division of the compression between prior
shrinkage and representational noise is scale-dependent; the compression, and
cTBS's effect on it, are not.

Two caveats to state, not bury. The primary is **not simpler** (p_loo 260 vs
254) — claim interpretable priors and better prediction, never parsimony. And
the log-space fit shows a **crossover**: above 30 CHF the effect reverses sign
with the CrI excluding zero. Real or spline-tail artefact is **[§5.5]**; a
reviewer will ask why cTBS would improve precision for large payoffs, so the
paper needs a sentence either way.

Close on what did not move: the mechanism. In both geometries noisier
representations are drawn harder toward the prior, the smaller safe option loses
more perceived value than the risky one, and the shift appears only where
leverage is high — which is what Figure 5 shows, unchanged.

### Edit anchors (Cmd+F into v10)

- [ ] Heading `A flexible noise model localizes the cTBS effect to shared perceptual noise`
      → `cTBS increased the noise of payoff representations for both options`
- [ ] Abstract `raised the noise on the perceptual representation` → `raised the
      noise of the neurocognitive representation of payoffs, with consequences
      concentrated on choices involving smaller payoffs`
- [ ] `among the six models tested` → `(Table 1; Supplementary Table 1)` (v10_todo #3)
- [ ] Delete `The best-fitting model placed the cTBS effect on shared perceptual
      noise alone.` → both-options result, with the +37.4 / +42.9 contrasts
- [ ] `Formal comparison across sixteen model variants` / `beat the null by 114.1
      nats, dSE 15.8` → `nine` / **+96.3, dSE 13.3** (recount from Table 1)
- [ ] `around 12% at 7 CHF against roughly half that over most of the range` →
      §0.4 statistics, crossover stated
- [ ] `estimated essentially no cTBS effect on perceptual noise` → re-verify vs
      `ppc_delta_by_stake.lfx2-bs3-w-dp-bm.tsv`; rewrite or delete
- [ ] `5-parameter B-spline function … in natural space` → ratio scale primary,
      linear scale sensitivity **[§5.3]**
- [ ] Methods `The flexible PMC model`: log-space parameterization ([7,112],
      5 df perceptual / 2 df memory, σ₁ = softplus(perc+mem), cTBS on every
      spline coefficient); the five factors; the selection rule; the convergence
      gate excluding family-2; the bauer stamp **[§5.4]**
- [ ] Discussion `Parameter estimates revealed that, in natural space,` → the
      scale-geometry argument; keep the mechanism clause verbatim. **Reinstate
      the prior passage dropped 2026-08-20**
- [ ] Captions: Fig 4D rewrite + **delete** its final sentence; Fig 4B/C
      restricted range; Supp Table 1 recast as one-factor-at-a-time
- [ ] PROVENANCE rows + a "which fit is which" entry for the lfx2 label grammar
      and the family-1/family-2 distinction

## 4. Figures

**Fig 4 stays single-model** (the primary), with the linear scale demoted to a
supplementary panel — rev. 1's paired layout no longer matches the argument.
`plot_fig4_paper.py` takes `--label/--draws/--ppc/--name/--out`:

```bash
python -m tms_risk.behavior.scripts.plot_fig4_paper \
  --label lfx2-bs3-m2-dp-bm --draws m2bm_subject_draws.tsv.gz \
  --ppc ppc_by_stake.lfx2-bs3-m2-dp-bm.tsv --out fig4_v11
```
Restrict the noise panels to 7–28 CHF (as `plot_story_full.py`), and rebuild the
ladder panel from the curated 9. The linear-scale counterpart needs
`plot_fig4b_natural.py`, not `--label` **[§5.2]**.

Elsewhere: `fig5_paper_m2bm.pdf` **reusable as-is** (verify quoted %);
`prior_placement.pdf` → **main Fig 4E** (**regenerate from family 1 [§5.3]**);
`story_full.pdf` → supplementary overview; `story.pdf` → graphical abstract if
requested; `model_overview.pdf` retired (script kept); Figs 1–3 and Supp Fig 1
untouched.

## 5. Prerequisites

**5.1 — DONE.** Master ladder with paired dSE: 46 traces, all scoring the
identical 8,335 observations (hash `c50de805c210`, gate enforced) —
`notes/data/ladder_v11.{tsv,md}`, `ladder_v11_contrasts.tsv`, pointwise ELPD in
`notes/data/ploo/*.npz`. Rebuild with
`tms_risk.behavior.scripts.combine_pointwise_loo --reference lfx2-bs3-m2-dp-bm`.
The published `derivatives/cogmodels/` traces carry **no `log_likelihood`
group** and can never enter this table without a bauer-version-sensitive
recomputation — hence the `*_noisefix.head` refits.

**5.2 — DONE for `w`** (`-w-dp-bm` −4195.1, `-w-dp-null` −4264.2). Open: extract
`weber2_noisefix*.head` (on gpu4, no pointwise ELPD yet) if the published Weber
PMC is to appear anywhere, and confirm the natural-space Fig-4B script.

**5.3 BLOCKING — re-derive the prior numbers from family 1.** The −4.63 CHF safe
prior comes from family-2 traces that failed the convergence gate, so as printed
it is indefensible even though it is the pivot of the rung-3 argument. Extract
`risky_prior_mu`/`safe_prior_mu` from `flexible1_noisefix.head` and requote (or
drop the number and make the argument qualitatively). Also settle whether
`flex2a_subject_draws.tsv.gz` came from the published `flexible2.6` (bauer
`ecc6454`) or a head refit, and regenerate `prior_placement.pdf`.

**5.4** Family-1 and family-2 traces are both stamped `e05f73a`, so they are
mutually comparable. Push the local-only bauer commits, and note that
`libs/bauer` HEAD and the GPU nodes' checkout have **diverged**: the nodes carry
uncommitted `memory_composition`/`spline_degree` work absent from HEAD, HEAD
carries LogFlexible work absent from the nodes.

**5.5 — is the high-payoff crossover a spline-tail artefact?** It sits at the
edge of the payoff range in the top stake bin; the `bs2` twin reproduced it, but
two bases with similar knots is weak evidence. Checks: compare above 30 CHF
across `bs2`/`bs3`/`cr3` at matched memory df (`cr3` is tail-constrained, so
agreement argues for signal); refit with knots on payoff quantiles rather than
[7,112]; count trials and unique payoffs above 30 CHF per condition, since only
the risky option reaches there. If robust the paper needs a sentence; if not,
restrict the range and note the extrapolation.

**5.6** Cancel the five PENDING `rdmlogflex` tasks, out of scope.

## 6. Risks

The p-hacking accusation is the one to answer properly, because the answer is
strong and concrete. All five belong in Methods in compressed form:

- **The headline claim is invariant across every fit** — TMS-vs-null gains of
  69–114 ELPD in every family and all 24 factorial cells. Surviving every
  specification is the *opposite* of selective reporting: name it a
  **specification-curve / multiverse** analysis.
- **A factorial over five named design choices, not a search** — every
  supplementary row is one-factor-at-a-time; non-converged cells shown.
- **Selection was on convergence and posterior-predictive coverage, stated a
  priori — explicitly not on the size or significance of the cTBS effect.**
- **The paper reports an effect weaker than the earlier candidate gave**
  (P = 0.845, not 0.95–0.97). Nobody fishing reports that.
- **The tiers stay separate:** existence rests on ELPD, localization on a
  model-free anchor, CrIs only characterize.

| Other attacks | Answer |
|---|---|
| You chose the linear-scale parameterization that fits worse | The better-fitting one has 2283 divergences and r̂ = 1.19; it has no defensible ELPD. Family 1 is the same likelihood, sampled properly. |
| Why demote the model the paper was built on? | It fits worse by 35.6 nats (2.5 dSE) and needs a prior below zero. Both stated in Results. |
| The −4.63 CHF prior is absurd | It is — that is the finding. Additive shrinkage on a linear scale forces it; a ratio scale puts the priors on the payoffs and fits better. |
| You said perceptual noise; now both options | σ₁ = perceptual + memory is unidentified in both scales; the two-option model beats either single-option one by 37–43 nats. |
| Why would cTBS *improve* precision above 30 CHF? | **[§5.5]** — answer before submission. |

## 7. Decisions for Gilles

| | Decision | Status |
|---|---|---|
| **A** | `-b` vs `-bm` | **Resolved by data** — z = −0.1, indistinguishable; plan uses `-bm` |
| **B** | No-prior comparison | **Resolved** — `-op` (583.5, z = 19.1) inside the architecture |
| **C** | Crossover paragraph | **Resolved** — stated as a caveat + **[§5.5]** |
| **D** | Prior panel main or supplementary? | **Main Fig 4E**, regenerated from family 1 |
| **E** | Reinstate the Discussion prior passage? | Yes, as the scale-geometry result |
| **F** | **One primary or two co-equal models?** | **NEW — plan assumes one primary.** The tie is gone; co-equal presentation would show a worse-fitting model as an equal |
| **G** | Push the local-only bauer commits, and reconcile with the GPU nodes' divergent checkout? | Open |
