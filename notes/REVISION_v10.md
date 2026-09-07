# Revising `TMS_paper_v10.docx` — section by section

Written 2026-09-07 against `notes/paper/TMS_paper_v10.txt` (the extracted text
of the .docx; line numbers below refer to it). Companion to
`notes/PAPER_EDITS.md` (which holds the *arguments*); this file is the
**checklist of what to open and change**.

**Scale of the job:** the neural half (Abstract ¶2 first half, Results §1–§3,
most of Methods) is **unchanged**. Everything from the cognitive model onward
is rewritten, because the model changed: the anchor-parameterised power-law
family replaces the 5-knot B-spline Flexible PMC, and it locates the cTBS
effect differently.

---

## 0. The one decision that gates everything

**Which model the paper reports.** Not yet settled — see §6 of
`notes/PAPER_EDITS.md` and the live status in
`notes/analyses/weber_affine_convergence.md`. The candidates and their
trade-offs:

| Model | ELPD rank | Δν₂ credible? | Reproduces ΔP? | Mechanism story |
|---|---|---|---|---|
| `log-power-n2` | 75 | yes, p = 0.002 | **no** (ppp 0.005, slope sign wrong) | cleanest |
| `log-power-n1n2` | 27 | yes, p = 0.020 | weak (ppp 0.010, slope sign wrong) | clean |
| `log-power-n2psd` | 4 | yes, p = 0.022 | ok (ppp 0.050, slope sign right) | + a prior-width term |
| `log-power-n1n2psd` | 1 | **no**, p = 0.06 | best (ppp 0.092) | + a prior-width term |

**Do not draft §4–§5 of the Results until this is fixed.** Everything else in
this file can be drafted now.

**Blocking issue:** `log-power-n2` and `log-power-n2psd` both fail the
convergence gate, and a refit (8 chains, tune 5000, mapjitter) did **not** fix
them — r̂ 1.086 / 1.127, and unlike `weber+affine` there is no single rogue
chain, so it is diffuse funnel-type mixing in the group-level SD
hyperparameters, not a stuck chain. They need a non-centred reparameterisation
before they are quotable. `log-power-n1n2` and `log-power-n1n2psd` both pass.

---

## 1. Abstract (line 8–9)

**Rewrite the last three sentences of ¶2.** Currently:

> "cTBS raised the noise on the perceptual representation of payoffs, in
> relative terms most strongly for smaller magnitudes. Because noisier
> representations are drawn more toward prior expectations, this underestimated
> the smaller safe options more than the larger risky ones…"

Three problems:
1. **"perceptual representation"** encodes the old finding that the effect sits
   on a *shared* perceptual channel. The new model localises it to the
   **second-presented option**.
2. **"in relative terms most strongly for smaller magnitudes"** — keep, but the
   reason changed (see §5 below); it is no longer "a constant absolute
   increment against growing baseline noise".
3. The safe/risky asymmetry sentence follows from the old mechanism and needs
   redoing once the model is chosen.

Everything before "A computational cognitive model…" stands.

---

## 2. Introduction (line 17–27)

**No changes needed.** It sets up the perceptual account and the causal gap;
neither depends on which noise model won.

Optional, only if you promote the Weber result (see §4b): one sentence flagging
that scalar invariance is itself an assumption worth testing.

---

## 3. Results §1–§3 — the neural half (lines 29–57)

**Unchanged.** "Experimental approach", "Decreased amplitudes of numerically-
tuned parietal cortex after cTBS", the six-variant cvR² comparison, and the
decoding paragraph all stand as written. Figures 1 and 2 unchanged.

---

## 4. Results §4 — psychophysics (lines 47–57, Fig. 3)

Numbers stand. Two structural changes:

**4a. Figure 3 is rebuilt** → `notes/figures/fig3_probit.pdf`. Same argument,
now with explicit cell-mean parameters and a narrower layout. Caption needs
rewriting to match panels (A = curves by order, B = slope and RNP per cell with
Δ and pBayesian printed on the panel).

**4b. NEW SUBSECTION, and the biggest addition: "Noise grows with magnitude —
Weber's law fails".** Goes **after** the psychophysics and **before** the
cognitive model. Figure: `notes/figures/fig4_weber.pdf`.

Why it belongs: the old draft motivates flexible noise from the nPRF preferred-
numerosity IQR ([6,10] vs [13,30]) plus a post-hoc median split (current lines
78–80). That is an inference from the neural data plus a model-based split. The
new analysis shows the assumption fails **model-free, in the baseline session,
in all 73 participants, before any stimulation** — choice consistency drops
−37% from low to high stakes when the risky option is second, −22% when first,
replicating in both the TMS and non-TMS subgroups.

This makes the flexible-noise model a *finding* rather than a *fix*, which is a
much stronger position. Content:
- probit slopes by stake, per order, n = 73, baseline only;
- the fitted noise functions: second-presented option grows **2.69×** across
  the payoff range, first-presented only **1.25×**;
- five flexible forms all agree on the shape; Weber is flat and wrong.

**Also promote Supplementary Fig. 1 (the stake split) into Figure 3**, since it
is what motivates this section. The old Supplementary Table 1 / Fig 1 numbers
at lines 78–80 then move here.

**Add the Stevens / power-law framing.** The next-simplest form after Weber is a
power law. Cite Stevens (1957) for the *psychophysical* precedent — but be
careful: Stevens' law concerns the **mean** percept, and applying it to the
*noise* gives σ_log ∝ φ^(−a), the wrong sign for what we find. Say explicitly
that we fit a power law on the **noise**, which is the "generalized Weber"
form, and cite Prat-Carrabin & Woodford (2022) for the magnitude-dependent-noise
precedent.

---

## 5. Results §5 — the cognitive model (lines 74–81, Fig. 4)

**Delete and rewrite the whole subsection.** Current title: *"A flexible noise
model localizes the cTBS effect to shared perceptual noise"* — the conclusion in
the title is no longer what we find.

What must go:
- **"5-parameter B-spline"** → the reported model is a **2-parameter power law**
  (splines become a robustness check: the shape is the same).
- **"shared perceptual noise alone"**, and the whole
  perceptual-vs-memory decomposition (the 2.0 / 59.9 / 64.3 / 69.9 nat
  comparisons) — that model family is superseded. The new family is indexed by
  *which presented option* carries the effect, not by perceptual vs memory.
- **"sixteen model variants"** → the grid is now 155 fitted models, of which
  114 converge. Quote a converged-only ladder.
- The claim that *"models in which the cTBS effect was confined to a single
  presentation position fit substantially worse"* is now **reversed** —
  single-position models are among the best.

What replaces it: noise on the second-presented option rises with payoff as a
power law; cTBS raises it, most at small payoffs. Table from converged models
only.

**Supplementary Table 1 (line 233–339) is entirely superseded** — regenerate
from `notes/data/loo_anchor/`, filtered to converged traces.

---

## 6. Results §6 — the mechanism (lines 83–87, Fig. 5)

**Rewrite.** Old Figure 5 (5 columns × 2 rows, decision-space heatmaps,
leverage) is retired; the new Figure 5 is a 3×3 page
(`notes/figures/fig5_<model>.pdf`).

Specific claims to drop:
- **"cTBS added a roughly constant absolute amount of noise (about 0.2 CHF)"** —
  wrong. In natural space fitted noise grows **supra**-linearly (ν ∝ x^1.28 for
  the second-presented option) and the cTBS effect is directly
  magnitude-specific.
- **"the first-presented option already carries working memory noise, so added
  noise compresses it most strongly. In safe-first trials, both factors act on
  the same option"** — this is the old mechanism. The new model puts the effect
  on the **second**-presented option.
- **"leverage"** as a named quantity — the panel is gone.

What replaces it: the two channels of the decision variable (perceived
risky/safe ratio vs decision SD) as a function of safe payoff, per order. They
**cross** at ≈17 CHF (risky second): decision noise dominates at small payoffs
(+10.7% at 7 CHF vs +2.5% for the ratio), the perceived-ratio shift dominates
at large ones (+8.1% vs −2.1% at 28 CHF).

**Numbers must come from the draw-integrated tables**
(`notes/data/anchor_mechanism.<label>.tsv`), never from per-subject medians or
a uniform ratio grid — both flip the sign of the risky-first effect
(`notes/analyses/aggregation_check.md`).

---

## 7. A new paragraph you do not yet have: what the model does *not* explain

Every candidate model under-produces the observed effect. Predicted ΔP(risky)
when the risky option came second is 0.017–0.030 against an observed **0.053**,
and every model gets the *slope* change too small (predicted −0.02 against
observed −0.10, ppp 0.93–0.99).

Say this explicitly rather than letting a reviewer find it. It is a genuine
limitation and reads as honesty, not weakness.

---

## 8. Discussion (lines 88–100)

Mostly survives. Three targeted edits:

1. **¶3** (line 91, "To achieve a more detailed mechanistic understanding…")
   — rewrite the second half. "spline functions", "scalar invariance…strictly
   linear increase", "cTBS raised it by an approximately constant absolute
   amount", and the safe-option-loses-more mechanism all change.
2. **¶2** (line 90): *"choices became less consistent and shifted toward
   risk-seeking, but only when the safe option was presented first"* — the
   empirical claim is fine; check the mechanism clause that follows it.
3. **Add** the Weber-failure result to the contributions list in ¶2 and the
   conclusion: it is a standalone psychophysical finding in 73 participants,
   independent of the stimulation.

Untouched: the Coutlee/Panidi comparison, the lesion paragraph, the
risk-elicitation paragraph, the arousal/stress paragraph, the conclusion.

---

## 9. Methods (lines 101–228)

- **Cognitive model subsection**: rewrite. New parameterisation (free
  parameters *are* noise SDs at named payoff anchors), power-law form, the
  placement grammar, priors (`PRIOR_SPEC v1-2026-08-28`), sampler settings.
- **Add a convergence-reporting sentence**: r̂ ≤ 1.01 and ESS ≥ 400 on
  group-level parameters, and say how many of the fitted grid met it.
- **Add** the baseline-session Weber analysis (n = 73, session 1, probit with
  subject dummies, stake median split).
- Unchanged: participants, fMRI acquisition, fMRIPrep, nPRF fitting, decoding,
  TMS protocol and neuronavigation.

---

## 10. Figures — final set

| # | File | Status |
|---|---|---|
| 1 | unchanged | — |
| 2 | unchanged | — |
| 3 | `notes/figures/fig3_probit.pdf` | rebuilt; caption to rewrite |
| **4** | `notes/figures/fig4_weber.pdf` | **new**; caption to write |
| 5 | `notes/figures/fig5_<model>.pdf` | rebuilt; **model not yet chosen** |
| S1 | promoted into Fig. 3 | — |
| S2 | PPC gallery | rebuild, converged only |
| S3 | model comparison | rebuild, converged only |

---

## Suggested drafting order

1. §4b (Weber fails) — fully determined, biggest addition, and it sets up
   everything after it.
2. §7 (what the model does not explain) — determined.
3. §1 Abstract, §8 Discussion edits — need only the direction of the result.
4. §5, §6, §9 — **after** the model decision in §0.
