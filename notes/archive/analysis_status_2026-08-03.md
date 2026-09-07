# Where we are — 2026-08-03

Short status across everything asked for today. Detail lives in
`notes/reanalysis_handoff.md`, `notes/encoding_model_choice.md`,
`notes/tuning_width_by_preference.md`.

## Done, and settled

**1. m1 vs m2 → use m1.** Settled on out-of-sample encoding fit alone.
m1 beats m0 by +0.0244 cvR² (34/35 subjects, p < 0.0001) — per-session amplitude earns
its keep. m2 is *worse* than m1 (−0.0055, p = 0.010, only 9/35 favour m2) — per-session
tuning does not. m2's Δamplitude is also entangled with its own nuisance parameters
(Δbaseline r = −0.50, Δsd r = +0.38, Δmu r = −0.40, all within-subject), and correlates
with m1's estimate of the same quantity at only r = +0.20.
→ `notes/encoding_model_choice.md`

**2. The brain–behaviour correlation should come out of the paper.** It is r = −0.43
under m2 but r = −0.08 under m1, and under m2 it does not survive partialling out
Δbaseline (p = 0.064). Reporting it would mean choosing the encoding model by which one
gives the result.

**3. "cTBS mostly affects low numerosities" is not supported as a group-level empirical
claim.** Three independent routes fail:
- *Model*: group-level P[Δν(7) > Δν(28)] = 0.35–0.43 absolute, 0.73–0.76 relative.
- *Neural*: amplitude loss does not depend on a voxel's preferred numerosity
  (within-subject mean rho = −0.040, p = 0.46).
- *Behaviour*: on the clean magnitude axis (n_safe, orthogonal to ratio at r = 0.024)
  the effect is flat — +0.042 vs +0.038, p = 0.92. The apparent localisation in
  n_risky bins is a ratio confound (n_risky correlates with ratio at r = 0.589). What
  *is* marginally localised is the ratio/indifference axis (p = 0.048 unadjusted).
→ `notes/reanalysis_handoff.md` §7.5–7.6

**4. What survives as backing for the low-numerosity framing** is a targeting fact, not
a behavioural result: tuning mass is concentrated at small numerosities. Density-weighted
precision falls 7.3× from n = 7 to n = 112; mass at n ≤ 20 is 0.662 vs 0.338 above,
p = 8.4e-06. Preferred-numerosity median ~9.7 against a presented IQR of [13, 30].

**5. Tuning width vs preference — the answer flips with units.** In log units (the
parameter the model fits) low-preference nPRFs are *wider*, i.e. less sharply tuned:
per-subject slope −0.245 in NPC12r (p = 0.049), −0.369 in the 2 cm stimulation ROI
(p = 0.0045), absent when all numerosity ROIs are pooled (p = 0.24). In linear units the
opposite, trivially (rho +0.40 to +0.54). Strongest at the stimulation site.
→ `notes/tuning_width_by_preference.md`

**6. No correlation between decoding / predicted uncertainty and behaviour.** Against
the behavioural localisation slope, n = 35, all null: baseline decoding error r = −0.121
(p = 0.49), cTBS change in decoding error r = −0.021 (p = 0.91), baseline predicted
uncertainty r = +0.003 (p = 0.98), cTBS change in predicted uncertainty r = +0.100
(p = 0.57). Same for `flexible2nf`. **But treat this null as weak** — see the next point.

## Does not reproduce / blocked

**The decoder is collapsed, and this blocks the whole decoding-precision analysis.**
Mean decoded value on vertex sessions is ~59–62 for *every* stimulus from 7 to 111
(bias +31 at s = 28, −51 at s = 111). So E(s) traces |60 − s|, the bounded grid, not the
neural code, and any log-log slope from it is meaningless (I get +0.11 to −0.12 against
Weber's 1.0 and the behavioural 0.45). Extending the evaluation grid will not fix it: if
the likelihood is near-flat, a wider grid just moves the centre. The prerequisite is a
decoder that carries information — a voxel-selection / noise-model problem. This also
means point 6's null may simply be two noise variables failing to correlate.

**The paper's preferred-numerosity IQR of [6, 10] does not reproduce.** I get
[7.10, 15.22] in NPC12r at cvR² > 0; it only approaches [6, 10] at cvR² > 0.10 in the
2 cm ROI ([5.15, 11.73]). The median (~8.7–9.7) is robust everywhere. Provenance of the
published [6, 10] needs to be found before it is re-quoted. The *presented* IQR
[13, 30] reproduces exactly.

## Not done — needs GPU, and the offered boxes cannot host it

**`sciencecloud_gpu…gpu4` are not usable for this work as provisioned.** Checked
directly: `tensorflow` and `braincoder` are both **missing** from `tms_risk_gpu`, and
the 22 GB derivatives tree holds only `cogmodels.*` behavioural traces — no
`encoding_model2.*`, no `glm_stim1.*`, no `ips_masks`. Using them means building a
TF+braincoder env *and* shipping tens of GB of imaging derivatives. The natural home is
**sciencecluster** (`tms_risk_cuda`, `/shares/zne.uzh/gdehol/ds-tmsrisk`), where the
existing m0/m1/m2 fits were produced and `modeling/slurm_jobs/` already has wrappers.

Two items are queued behind that:

1. **The m3 encoding variant** (mu fixed across sessions, amplitude and dispersion
   free) — the model matching "cTBS reduces gain without moving tuning preference".
   Needs 35 subjects × (1 main fit + 6 CV folds). Code change: a `model_label == 3`
   branch in `fit_regression_nprf.py::get_model`/`get_grid` plus a matching `fixed_pars`
   branch in both fit scripts. Hours as a SLURM array.
2. **The full simulation-decoding study** (extended evaluation grid, the
   density/width/gain decomposition, the IPS-vs-vertex attribution swaps, mean posterior
   SD, three cvR² cutoffs). Ω is *not* stored anywhere — `decode.py` and
   `fisher_information.py` refit it and discard it — so every variant needs a
   `ResidualFitter` run first. Multi-day as specified. **Worth deciding whether to run
   it at all until the decoder collapse is resolved**, since right now it would produce
   more collapsed decoders.

## Suggested next step

Fix the decoder before spending GPU time on the simulation study. The collapse is the
single blocking finding: it invalidates the E(s) analysis, weakens the
decoding↔behaviour null, and would silently propagate into every variant in the queue.
