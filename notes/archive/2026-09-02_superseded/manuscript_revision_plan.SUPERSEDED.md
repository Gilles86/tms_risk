# Manuscript revision plan (v10 -> v11)

Written 2026-09-01. Source text: `notes/paper/TMS_paper_v10.docx` (extracted to
plain text for auditing). Target model: **`log-power-n1n2`** — anchor
parameterisation, log space, power noise form (free parameters are sigma at 7
and 112 CHF), cTBS given its own noise function for the first- and
second-presented option. bauer `93c2e9a`, prior spec `v1-2026-08-28`.

---

## 1. What actually changed, scientifically

Three claims in v10 do not survive the anchor refit. Everything else does.

| v10 claim | Status | Replacement |
|---|---|---|
| Noise grows **sublinearly** with payoff in natural space | **Reversed** | sigma_log RISES with payoff, so natural-space noise is **supra-linear**: nu ~ x^1.33 for the second-presented option, x^1.05 (Weber-like) for the first |
| cTBS added an **approximately constant absolute amount** of noise (~0.2 CHF), hence largest in relative terms at small payoffs | **Gone as a mechanism** | The effect is directly magnitude-specific: **+29.7% at 7 CHF**, falling monotonically to −9.4% at 112 CHF, credible only over **7–11.5 CHF** |
| cTBS acts on **shared perceptual noise**; a memory effect adds nothing | **Not supported as stated** | The credible effect sits on the **second-presented option** (n2). n1 never reaches credibility (+5.5% to −6.6%). `perc`/`percmem` now fit 4–5 nats WORSE than `n1n2` |

Claims that **survive unchanged**:

* flexible noise beats Weber — `log-weber-n1n2` is **35.6 nats** worse
* every cTBS model beats its null — `log-power-nullind` **−94.5**, `log-power-null` **−114.1**
* the relative perturbation is largest at the smallest payoffs
* order specificity emerges from the model rather than being fitted (`n1` alone −30.6, `n2` alone −25.3, both together is best of the three)
* all fMRI results (Figs 1–2), the psychophysical results (Fig 3), and the brain–behaviour correlation

### The elephant: the prior-SD effect

The global best model over 127 log-space variants is **`log-power-n1n2psd`**
(−4124.2), which lets cTBS change the **width of the prior** as well as the
noise. `log-power-n1n2` is **30.3 nats** behind it. `pmusd` (+24.5) and `psd`
(+22.4) are also well ahead of the noise-only model.

`notes/PLAN.md` §5 currently rules prior-width out of the paper. That decision
now costs 30 nats of ELPD and has to be either defended in the text or revisited.
See open decision **D1**.

---

## 2. Text edits, section by section

### Abstract (line 7-16 of the extract)
* "raised the noise on the **perceptual representation** of payoffs" — "perceptual" is a claim about the perc/mem decomposition that the refit does not support. Change to the position-based statement, or resolve **D2** first.
* "this **underestimated the smaller safe options more than the larger risky ones**" — must be re-derived. In `n1n2` the noise lands on the second-presented option, which on risky-second trials is the RISKY one. The model still predicts the right sign for the perceived ratio (Fig 4f/g: +2 to +4% for risky-second), but the verbal chain in the abstract is the old one. **Rewrite after D2.**
* "in relative terms most strongly for smaller magnitudes" — keep, and update the number.

### Results, "A flexible noise model localizes the cTBS effect to shared perceptual noise"
* **Section title changes** — it names the conclusion that changed.
* Paragraph introducing the flexible model: "we introduced a 5-parameter B-spline function that maps magnitudes to noisiness **in natural space**" -> the anchor parameterisation in log space. Two free parameters per channel, which ARE the noise SD at 7 and 112 CHF, log-interpolated.
* Model-comparison paragraph: every number changes.
  - "three best-fitting models were all flexible-noise models" — recheck against the 127-model ladder.
  - "best flexible beat best Weber by 34.8 nats (dSE 17.3)" -> **35.6** (recompute dSE).
  - "best model beat the null by 114.1 nats (dSE 15.8)" -> **114.1** for `log-power-null`, **94.5** for `nullind`. Say which null.
  - "adding memory noise costs nothing (2.0, dSE 3.9), removing perceptual costs 59.9 (dSE 12.2)" -> **delete**; replace with the n1/n2 decomposition (n1-only −30.6, n2-only −25.3 relative to both).
  - "restricted to one presentation position fit substantially worse (64.3, 69.9)" -> **30.6 and 25.3**.
  - "around 12% at 7 CHF against roughly half that over most of the range" -> **+29.7% at 7 CHF, crossing zero near 51 CHF, −9.4% at 112 CHF; credible only 7–11.5 CHF**.
  - "a Weber-law version estimated essentially no cTBS effect on perceptual noise" — recheck against `log-weber-n1n2`.
* The **stake-median-split psychophysics paragraph** (slopes 3.02->2.33 etc.) is independent of the PMC refit and can stand — but the numbers come from a probit whose random-effects structure is unsettled (**D3**).
* The paragraph deriving why the effect lands on safe-first trials ("total decision noise is dominated by the first-presented option... added noise matters most when it acts on the first option") **contradicts the new fit**, which puts the credible effect on the second option. Rewrite.

### Results, "Understanding the influence of cTBS on neurocognitive representations"
This whole section is the Figure 5 walkthrough (perceived value lost, decision-space maps, leverage, distortion x leverage). **Figure 5 is being retired.** Options:
* compress to two paragraphs built on the new Fig 4f/g (perceived ratio and decision noise, the two ingredients of the decision variable), or
* keep the leverage/decision-space argument as a supplementary figure and cite it.
Either way the "safe option lost more perceived value" chain must be re-derived under `n1n2`.

### Discussion
* "representational noise grows **sublinearly** with payoff magnitude, and cTBS raised it by an **approximately constant absolute amount**" -> reverse both halves.
* "this reduced the perceived value of the (smaller) safe option more than that of the risky one" -> re-derive.
* Everything about targeting, nPRF, comparison to Coutlee/Panidi, and the future-directions paragraph stands.

### Methods
* Replace the B-spline / natural-space model description with the anchor parameterisation: forms (weber/affine/power/spl3), placements, what the free parameters mean, the log link.
* State the **prior spec** (`v1-2026-08-28`), the sampler, and the **bauer commit** the traces are stamped with. `notes/PLAN.md` §6 has the clone-and-PYTHONPATH protocol.
* Add the choice-rule note (raw-evidence vs KLW-consistent `diff_sd`) and say which was used.

---

## 3. Figures

| Figure | Action |
|---|---|
| 1, 2 | **Unchanged** |
| 3 | Unchanged pending **D3** (which probit fit the panels and the reported slopes come from) |
| **4** | **Replaced** by `notes/figures/fig4_big_log-power-n1n2.pdf` — full page (7.25 x 8.6 in). **a,b** the two noise terms; **c,d** the cTBS effect on each, with a black bar where the 95% CrI excludes zero; **e** where the priors sit; **f,g** the two ingredients of the decision variable; **h** the psychometric functions, three stake terciles x two presentation orders, IPS vs vertex; **i** model comparison. Script: `tms_risk/behavior/scripts/plot_fig4_big.py`. Also rendered for `log-power-n1n2psd` (the D1 sensitivity analysis) and `log-weber-n1n2` (the qualitative Weber failure) |
| **5** | **Deleted.** Mechanism absorbed into 4f/g; decision-space maps optionally demoted to supplementary |
| Supp Fig 1 (stake split) | Keep; regenerate against the chosen probit (**D3**) |
| Supp Table 1 (16 variants) | **Replace** — the ladder is now 127 log-space variants. Needs a curated table, not a dump |
| New supplementary | The spline models (`spl3`), showing the power form is not doing the work; the natural-space arm (**D4**) |

### Why panel h is the full psychometric function
The earlier version plotted P(chose risky) against safe payoff — two panels, one
number per cell. The bias statistics it summarises can be matched by a model
that gets the SLOPE wrong, and the effect had to be read off an axis. The
tercile-split psychometric curve (`ppc_anchor.stake3rung.*`, new output of
`extract_anchor_ppc`) shows slope and shift together, matches the split the
published Figure 4A used, and makes the claim legible without reading an axis:
**red sits above green in all three risky-second panels and in none of the
risky-first panels.** RMSE 0.040 over the 72 cells, 94.4% inside the 95% band.

**No error bars on the observed points**, deliberately. The band is a posterior
PREDICTIVE interval: choices are simulated for the same participants using their
own per-subject parameters, so it already carries the trial-level sampling noise
the observed proportion is subject to. Measured on the 72 cells, the band
half-width is **0.75x** 1.96*SEM (narrower in 70 of them) yet covers **94.4%**
of the observed points -- so the band is calibrated and the SEM is simply the
wrong scale for the comparison: it measures spread ACROSS subjects, which the
model conditions on rather than predicts. Plotting both made the fit look less
determined than it is. The caption must say the dots are observed proportions
and the band is the 95% posterior predictive interval.

### Deliberately NOT in Figure 4
A probit slope / risk-neutral-probability PPC was built and dropped: **the
model's choice function is not a probit in log(frac)**, because nu depends on
payoff and the risky payoff is frac x n_safe. The closed form linearises at each
cell's mean payoffs, which costs up to 3.7 percentage points and a 14% slope
error, concentrated at risky-second / high payoff — exactly the contrast the
paper claims. Full detail in `notes/PLAN.md` §3b. Code survives behind
`--with_probit`.

---

## 4. Decisions taken 2026-09-01

**D1 — prior-SD effect: RESOLVED as a nuisance term, not a mechanism.**
`log-power-n1n2psd` wins by 30.3 nats and fits visibly better
(RMSE 0.0230 -> 0.0212; 95%-band coverage 0.90 -> 0.95; per-subject coverage
97.9% -> 98.6%), and it roughly doubles the predicted cTBS effect on
risky-second choices (+0.016 -> +0.030 against an observed ~+0.053), which
turns the failing posterior predictive check into a passing one
(`dp_second_mean` ppp 0.01 -> 0.09; `slope_second_ctbs` ppp 0.98 -> 0.93).

**But the group-level cTBS effect on prior width is exactly zero:**

| prior | vertex − IPS, log scale | p(>0) |
|---|---|---|
| risky | −0.001 [−0.065, +0.065] | 0.482 |
| safe  | +0.034 [−0.050, +0.143] | 0.785 |

**Verified, not inferred**: `psd` moves only the prior SD (the prior *mean* is
intercept-only in that fit; `pmu`/`pmusd` are the separate mean-varying
variants). The between-subject SD of the stimulation contrast on prior width is
credibly non-zero — risky **0.330 [0.231, 0.456]**, safe **0.241 [0.050,
0.417]** — and the risky one is nearly as large as the between-subject SD of the
intercept (0.397). So participants differ in prior width between sessions by
roughly a factor of 1.4, while the group mean of that difference is zero.

The 30 nats therefore buy **per-participant session-specific prior width**, not
a systematic effect of stimulation. Caveat for the text: each participant has
only two sessions, so within a participant stimulation and session are perfectly
confounded. Counterbalancing keeps the group effect clean, but the per-subject
variance necessarily mixes genuine heterogeneity in cTBS response with ordinary
session-to-session drift — which is itself an argument for reporting it as
nuisance variance rather than as a finding. Decision: keep
`log-power-n1n2` as the primary, interpretable model; report `n1n2psd` as a
sensitivity analysis with those two numbers stated. This is the defensible
sentence D1 needed, and it is true.

**D2 — drop the perceptual/memory decomposition.** Justified, but NOT by
"an effect on n2 and not n1": that contrast is not itself credible
(n1 @7 CHF +0.012 [−0.029, +0.052], p=0.726; n2 @7 CHF +0.029 [+0.002, +0.057],
p=0.980 — overlapping). The real reasons:
1. `percmem` estimates the **memory** effect at zero (−0.0015 [−0.036, +0.031],
   p=0.462) and so collapses onto `perc`, which raises both options equally
   (+0.023 vs +0.025 at 7 CHF). It is 0.5 nats from `log-power-perc` — the extra
   parameter buys nothing.
2. `n1n2` is the parameterisation that lets the data speak about position, and
   it fits 4-5 nats better.
**Caveat to respect in the text:** `log-power-n1n2` beats `log-power-n2` by
25.3 nats, so we must NOT write that cTBS affected only the second option in the
model-structure sense. The claim is about the fitted magnitude profile.

**D3 — use the published random-intercept probit** (`.ri`). Repoint Fig 3B and
Supp Fig 1; reported slope becomes Δ −0.666 [−1.043, −0.286].

**D4 — drop the natural-space arm** (mention in Methods only).

**D5 — report the amplitude gap.** Draft wording in the chat log; it is now a
smaller gap given D1.

**Noise form — present splines, report the power law.** In the `n1n2` family:

| form | k | ELPD | vs power |
|---|---|---|---|
| power | 2 | −4154.5 | 0.0 |
| spl3 | 3 | −4154.8 | −0.3 |
| affine | 2 | −4155.2 | −0.7 |
| spl5 | 5 | −4163.0 | −8.5 |
| genweber | 2 | −4175.2 | −20.7 |
| weber | 1 | −4190.1 | −35.6 |

Magnitude-dependence is required (Weber costs 35.6 nats) but two parameters
suffice: the power law matches the 3-knot spline and beats the 5-knot one. Same
ordering holds in the `n1n2psd` family. This is a clean Occam argument and the
power law is the form the literature already uses.

## 4. Open decisions — these are yours

**D1. The prior-SD effect.** `log-power-n1n2psd` beats the noise-only model by
30.3 nats. Options: (a) keep noise-only as primary and report the psd result in
supplementary with a stated reason for not featuring it; (b) promote it to
primary and rewrite the mechanism as noise + prior width; (c) present both as a
sensitivity analysis. You have previously said you do not want to publish on a
prior-width effect — (a) is consistent with that but needs a defensible sentence,
because 30 nats is not nothing.

**D2. `n1n2` vs `perc`/`percmem`.** Only 4–5 nats apart. `n1n2` is
position-based and is what the current Figure 4 shows; `perc`/`percmem` is the
perceptual-vs-memory decomposition the PMC framing and all of v10's prose rest
on. Choosing `n1n2` means rewriting the localisation claim in the abstract,
results, and discussion. Choosing `percmem` keeps continuity but needs the
credibility of its cTBS effect checked first.

**D3. Which probit is the target for Fig 3B and Supp Fig 1** — random-intercept
(the published structure, `probit_stake_group_posterior.ri.tsv`, Δslope −0.666
[−1.043, −0.286]) or full random effects (−0.398 [−0.880, +0.113])? The figures
currently use the second; the published structure is the first. This changes
reported numbers.

**D4. The natural-space arm.** v10's model lived in natural space; the refit is
log space. Supplementary, or drop entirely?

**D5. Do we state the amplitude gap in Results?** The model reproduces the
pattern of the cTBS effect on choice but about a fifth of its size. Everything
testable has been ruled out (bug, priors, prior width, role scale, order
interactions, choice rule, spline count); per-subject effects range −0.42 to
+0.37 with only 20/35 positive. Reporting it is honest and pre-empts a reviewer;
not reporting it is defensible only if Fig 4h/i make it visible anyway (they do).

---

## 6. Audit corrections (2026-09-02)

Two audits were run against v10 and this plan. What they changed:

### The primary model is now `log-power-n2psd`
Log space, power noise form, cTBS on the SECOND-presented option's noise plus the
width of the magnitude priors. Everything numbered above against
`log-power-n1n2` is superseded.

Why it is defensible, in the order the Results should make the case:
1. **ELPD/Occam** — within 2.8 ± 4.9 nats of the best model with fewer cTBS
   parameters; 122 nats better than the null; 18 better than Weber.
2. **Predictive adequacy** — passes 7 of 8 predictive checks on the Figure-3B
   quantities.
3. **Theory** — it instantiates the hypothesis; the prior-width-only model is its
   null, not its rival.

### The noise term cannot be justified by ELPD, only by the slope check
`log-power-psd` (prior width, NO noise effect) is 5.1 ± 4.4 nats behind — ELPD
cannot separate them. What separates them is the cTBS effect on the psychometric
SLOPE: `psd` predicts −0.281 and FAILS (ppp 0.025); `n2psd` predicts −0.482 and
passes (ppp 0.145) against an observed −0.702. The mechanism is that noise enters
the slope twice (raises `diff_sd`, lowers `w`) while prior width enters only
through `w`, so the two are near-degenerate for the bias but not for the slope.

**This supersedes §3's "Deliberately NOT in Figure 4".** That section rejected a
probit PPC because the CLOSED FORM linearises. The check used here does not use
the closed form: it simulates choices from the posterior and fits the same ML
probit to simulated and real data, so no approximation enters. `probit_ppc_ml.py`.

### v10 claims that must go, beyond those in §2
* **Introduction**, not only Results: *"the smaller, safe payoff lost more
  perceived value than the larger, risky one, particularly when it had to be held
  in working memory as the first-presented option"* — dead under the n2 placement.
* **"the order specificity emerges from the model rather than being fitted
  directly"** — DELETE. The primary model now confines the effect to one position
  by construction. This is a real loss and must not be papered over.
* **"The model reproduces the observed (lack of) change in the proportion of
  risky choices in both conditions across all stake sizes"** — false. ΔRNP
  observed +0.049, predicted +0.021, ppp 0.995, and EVERY model fails it.

### Verified, contrary to the audit
The published brain–behaviour correlation **reproduces exactly**: r = 0.533,
p = 0.001, Spearman ρ = 0.592 (v10 reports 0.53 / 0.59) — from mask
`NPCr2cm-cluster`, selection `cvr2pos`, measure `d_amp_median`. The r = 0.41 I
reported earlier was a different column of `bb_link_master.tsv`. Caveat for the
text: this is the strongest of 75 mask x selection x measure combinations, though
the mask is the pre-specified stimulation-site one.

### Open, and genuinely unresolved
1. **Selective use of ELPD.** The plan keeps ELPD as evidence against Weber and
   the null while discounting it for `psd`. Adopt one stance and state it:
   ELPD ranks, predictive checks adjudicate.
2. **Multiplicity.** The noise term rests on one check out of eight. Either
   pre-specify it as the primary check or state the multiplicity.
3. **The model's noise shift does not track the neural measure** (r = +0.07,
   Spearman +0.16; sweep over 420 pairings gives 18 hits against 21 expected by
   chance). The brain-behaviour section uses a model-free measure. Report the
   null explicitly rather than omitting it.
4. **No figure supports** the slope PPC, the failing RNP check, or the
   per-subject prior-width spread. At least the first belongs in the supplement.

---

## 7. Weber's law: state the violation explicitly (2026-09-02)

A result currently absent from the manuscript, and worth a short paragraph of
its own in the Results, right before the cTBS effect:

**Weber's law holds for the first-presented option and fails for the second.**
Weber's law (scalar variability) is exactly constant sigma in log space. Under
the primary model `log-power-n2psd`, over the 7-112 CHF range:

| option | sigma(7) | sigma(112) | ratio | exponent b | natural space |
|---|---|---|---|---|---|
| first-presented | 0.215 | 0.237 | **1.10x** | +0.036 | nu ~ x^1.04 |
| second-presented (vertex) | 0.106 | 0.228 | **2.15x** | +0.276 | nu ~ x^1.28 |

Formally: replacing the power law with Weber noise costs 18 nats (see the
comparison figure); within the `n1n2` placement it cost 35.6.

### The right law to name, and the one NOT to name

The minimal generalisation of Weber is a **power law on the NOISE**,
sigma(x) ~ x^b, which nests Weber at b = 0 and costs one parameter. This is the
standard "generalized Weber's law" form.

**Do not invoke Stevens' law.** Stevens is a power law for the PERCEIVED
MAGNITUDE, psi = phi^a. If noise were constant in that space then
sigma_phi ~ phi^(1-a) and **sigma_log ~ phi^(-a)** -- DECREASING with magnitude
for any positive exponent. We observe sigma_log INCREASING (b = +0.28), which
would require a = -0.28. Writing "Stevens" here would be sign-wrong and a
psychophysics reviewer would catch it.

Note also that our departure runs OPPOSITE to the classic "near-miss to Weber's
law" in loudness (McGill & Goldberg 1968), where the exponent sits slightly
BELOW proportional. Ours sits above. Say so rather than letting a reader assume
the familiar direction.

### References to cite (verify details before submission)
* Weber-Fechner / logarithmic magnitude coding: **Dehaene (2003)**, *TiCS*,
  "The neural basis of the Weber-Fechner law".
* Scalar variability in numerosity: **Whalen, Gallistel & Gelman (1999)**,
  *Psychological Science*; **Gibbon (1977)**, *Psychological Review* (scalar
  timing).
* Departures from Weber: **McGill & Goldberg (1968)**, *Perception &
  Psychophysics* (the near-miss).
* Stevens' power law, if mentioned only to be set aside: **Stevens (1957)**,
  *Psychological Review*, "On the psychophysical law".
* Flexible magnitude-dependent noise in economic choice: **Prat-Carrabin &
  Woodford (2022)** -- already cited in v10.

### The interpretive payoff
The shape difference recovers the perceptual/memory distinction WITHOUT
parameterising it, which matters now that `percmem` is dropped (D2). The
first-presented option carries high (0.215) and magnitude-INDEPENDENT noise --
the signature of a working-memory component. The second carries low (0.106) and
strongly magnitude-DEPENDENT noise -- the signature of a perceptual one. cTBS
acts on the latter, at small magnitudes, which is where the stimulated
population is tuned (nPRF preferred numerosities IQR [6, 10] against a presented
IQR [13, 30]; Fig 2B). That is the same argument v10 made, reached from the
noise functions rather than from a decomposition the data do not support.

---

## 8. NEW Figure 4: "Weber's law fails, and only for the second option"

Decided 2026-09-02. The Weber comparison currently has no visual support
anywhere (the first audit's MUST-FIX 1), and it is doing real work in the
argument. It deserves its own figure, placed between the psychophysics and the
cTBS result. Renumbering:

| # | content | status |
|---|---|---|
| 1 | paradigm, nPRF targeting | unchanged |
| 2 | nPRF parameters, decoding | unchanged |
| 3 | psychophysics: what cTBS did to choices | unchanged (D3: `.ri` probit) |
| **4** | **NEW — noise grows with magnitude, only for the 2nd option** | to build |
| **5** | the cTBS effect on that noise, mechanism, and choice PPC | = current `fig4_big_log-power-n2psd.pdf` |

No Figure 6. The old Figure 5 stays retired.

### Why this ordering works
It separates "what kind of observer is this?" from "what did cTBS do to it?".
v10 interleaved them, which is why the model-comparison material kept crowding
the cTBS panels. It also front-loads a result that needs no model at all.

### Panel plan

**a. The model-free demonstration.** Probit slope by stake and presentation
order, vertex sessions only, from the published hierarchical probit. **Weber's
law predicts the slope does not depend on stake.** Observed:

| order | slope, low stake | high stake | drop | p(>0) |
|---|---|---|---|---|
| Risky **first** | 2.525 | 2.523 | **+0.002** [−0.471, +0.473] | 0.503 |
| Risky **second** | 2.927 | 2.071 | **+0.856** [+0.392, +1.335] | **1.000** |

Exactly Weber for the first-presented option; a 29% decline for the second.
Draw the Weber prediction as a flat reference through the low-stake value.
Source: `notes/data/probit_stake_group_posterior.ri.tsv`.

**b. The fitted noise functions**, vertex only, both options, log-log, with a
horizontal Weber reference. σ(7)→σ(112): first-presented 0.215→0.237 (1.10x),
second-presented 0.106→0.228 (**2.15x**). Same dissociation, now from the model.
This is panels a+b of the current Figure 4, stripped of the cTBS contrast.

**c. Weber against the data.** The Weber model's choice PPC beside the power
model's, or their residuals against stake. Rows already exist in
`supp_ppc_gallery`; lift the two relevant ones.

**d. Noise-form ladder** within the chosen placement: weber / affine / power /
spl3 / spl5, ELPD with paired SE. The Occam point: magnitude-dependence is
required (Weber costs 18-36 nats depending on placement) but **two parameters
suffice** — the power law matches the 3-knot spline (0.3 nats) and beats the
5-knot one (8.5). Blocked on the `log-weber-n2psd` / `log-spl3-n2psd` /
`log-affine-n2psd` fits (job 5457817, running).

### The text this figure supports
Goes where v10's "we refined the model further" paragraph sits, and replaces the
hand-wave with a measurement. It also carries §7's Weber/Stevens framing, and it
gives the perceptual-vs-memory reading (high flat noise on the first option,
low steep noise on the second) a figure to point at now that `percmem` is gone.

---

## 9. The bridge from the earlier papers (2026-09-02)

The earlier applications of the PMC model (de Hollander et al. 2024a;
Barretto-Garcia et al. 2023) assumed scalar invariance and never tested it. The
paper needs to say why it is being relaxed now, BEFORE any model is fitted, or
the whole model-class change reads as post hoc. Two independent motivations, in
this order:

### 1. Weber is rejected in the BASELINE session, with no stimulation involved
73 participants, 8662 trials, session 1 only. Probit with subject dummies:

| order | slope, low stake | high stake | drop | |
|---|---|---|---|---|
| Risky second | 2.623 | 1.652 | **−37%** | z = −7.75, **p < 0.0001** |
| Risky first | 2.599 | 2.026 | −22% | |
| order x stake x slope | | | | z = +2.23, **p = 0.026** |

Weber predicts no stake dependence at all. This uses the full cohort including
the 38 participants who were never stimulated — data otherwise unused.

**Correction to §7 and §8.** Those sections said Weber "essentially holds for
the first-presented option" (drop +0.002 in the vertex sessions). That was
n = 35. At n = 73 the first option violates Weber too, by 22%. It is a
difference of DEGREE, not kind: both options depart from scalar invariance, the
second significantly more. The fitted σ ratio for the first option (1.10x over
7-112 CHF) is mild, not null. **Do not write the clean dissociation**; §8 panel
a must show the baseline result, not only the vertex sessions.

### 2. The stimulated populations are tuned below the presented range
Already in v10: nPRF preferred-numerosity IQR [6, 10] against a presented IQR
[13, 30] (Fig 2B). A cTBS effect on those populations should be
magnitude-specific rather than uniform. This motivation is independent of (1)
and is about where the perturbation lands, not about the baseline observer.

### Draft paragraph
> Our earlier applications of this model (de Hollander et al., 2024a;
> Barretto-Garcia et al., 2023) assumed scalar invariance — noise proportional
> to magnitude — which is the standard idealisation and which we did not test.
> Two observations, neither of which depends on the cognitive model, indicate
> that it should be relaxed here. First, the psychophysics violates it: in the
> baseline session, before any stimulation, choice consistency declined by 37%
> from low- to high-stake trials when the risky option was presented second
> (p < 0.0001), and the decline was larger for the second-presented option than
> the first (p = 0.026). Second, the cortical populations we targeted are tuned
> to numerosities well below the presented range (IQR [6, 10] versus [13, 30];
> Fig. 2B), so any cTBS effect on their representations should be
> magnitude-specific rather than uniform. We therefore extended the model to
> estimate the relationship between magnitude and noise, rather than assuming
> it.

Placement: replaces v10's "we refined the model further, following the
recommended procedures" paragraph, which currently motivates the change by
appeal to Gelman and to model misfit rather than by evidence.
