# Supplementary figures for v12

Four figures, in the order you proposed. Everything else I built during the
model search is working material, not a supplement.

## S1 — Stake × cTBS × order (the model-free probit)  ✓ exists

`plot_supp_fig1_stake.py` → `supp_fig1_stake.pdf`. Indifference points as
risk-neutral probability, low vs high stake, by presentation order.
Choice-rule-independent, so nothing about the refit touches it. This is what
the draft already cites.

## S2 — The ELPD ladder  ✓ exists, needs a rerun

`plot_elpd_ladder.py` → `supp_elpd_ladder.pdf`. Panel a: where the cTBS effect
acts. Panel b: what shape the noise function takes. Bars are the PAIRED
difference against the reported model with the SE of that difference; rungs
that miss the convergence gate carry their own r̂ and ESS rather than a
verdict, and only genuinely unusable fits are greyed.

Rerun once the ladder refits land, so every bar sits on one prior spec.

## S3 — All the PPCs, compact  ✗ to build

One panel per targeted statistic is too much; the compact form is the one the
design audit picked: **every targeted statistic on a single standardised axis**,
each observed value positioned inside its own predictive interval, with the
posterior predictive p in a column at the right. Eight rows, one figure,
no per-panel axes. Built by extending `plot_fig4_big --ppc stats` into a
standalone script.

Add a second panel with the individual-participant check
(`plot_ppc_subject.py`): observed against predicted for all 420
participant × cell proportions, with the coverage number. That is the panel
that answers "does it work for individuals or only for the group", and it is
the strongest single piece of evidence in the PPC set (r = 0.93, 97% coverage).

## S4 — The noise functions  ✓ built, waiting on fits

`plot_supp_noise_functions.py` → `supp_noise_functions.pdf`. Panel a: every
fitted ν(x) on one axis, vertex condition, both options. Panel b: the cTBS
effect each form implies, with the reported form's credible band behind it.

Currently four forms (Weber, generalised Weber, affine, power). The smooth
spline ladder (cspl3/5/7, job 5707201) drops in when it lands and is the point
of the figure: **power → cspl3 → cspl5 → cspl7 changes only the number of
anchors**, because all four use the log link. The piecewise-linear spl3/spl5
change the link at the same time, so a difference there confounds smoothness
with resolution — they go in the table as a robustness row, not in this figure.

What panel b already shows with four forms: every payoff-dependent form gives
Δν ≈ +0.045 at 7 CHF falling to ≈ 0 by 56 CHF, and they sit on top of one
another inside the reported form's credible band. Weber, having no payoff
dependence to give, returns a flat +0.007. The answer does not depend on the
form; it depends on the form being allowed to vary with payoff at all.

## Not supplements

Everything under `notes/figures/panelF/`, the `fig5_*` variants, `psy3_*`,
`ppc_safe_*`, `ppc_stake_*`, `supp_ppc_gallery*`, `supp_ppc_psychometric_*`,
`supp_weber_misfit`, `supp_model_comparison`, `supp_subject_params`. These were
built to choose the reported model and the panel-f design. Keep them in the
repo, cite none of them.
