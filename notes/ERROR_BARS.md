# What every error bar in this paper means

One rule, three cases. If a figure deviates it is listed at the bottom with a reason.

| What is plotted | Interval | Written as |
|---|---|---|
| **Raw data** — choice proportions, per-subject measurements | **±1 SEM across subjects** | "Error bars show ±1 SEM across the 35 participants" |
| **A parameter, or anything derived from a posterior** — noise functions, prior means, ν₁ − ν₂, probit slope/RNP | **95% credible interval** | "Shaded regions show the 95% credible interval" |
| **A posterior predictive of an observable** — the model's predicted choice proportion in a PPC | **95% posterior predictive interval** | "The band is the 95% posterior predictive interval" |

Never write "confidence interval" — nothing here is frequentist.

## The three rules behind the rule

1. **SEM is between-subject and paired where the contrast is within-subject.** For a
   cTBS effect, compute IPS − vertex *per subject first*, then take the SEM of that
   difference across subjects. Taking SEMs of the two conditions separately and
   comparing them throws away the pairing and inflates the bar.
   `extract_behavior_grid.py` does it the paired way.

2. **A credible interval on a derived quantity must be propagated through the draws,
   never assembled from marginals.** The interval for ν₁ − ν₂ is
   `quantile(nu1_draws - nu2_draws)`, not anything computed from the intervals of ν₁
   and ν₂. Same for ratios and products. `extract_pmc_parameters.py` follows this;
   `validate_source_data.py` checks that stored contrasts equal the difference of the
   stored means.

3. **Aggregate before summarising, not after.** For a PPC, compute the statistic the
   data represent (e.g. mean over subjects) *within each draw*, then take quantiles
   across draws. Reversing the order collapses the predictive uncertainty. Related and
   more dangerous: evaluating a nonlinear link at the *mean* parameters is not the mean
   of the link — see the "mean-parameter trap" in the scientific-figures skill. That
   error made a good probit fit look bad (RMSE 0.069 vs 0.019).

## Figure by figure

| Figure | Element | Interval | Source |
|---|---|---|---|
| Fig 3a,b | probit prediction band | 95% CrI | `probit_{stim}_lo/hi` |
| Fig 3a,b | observed proportions | **no bars** — PPC convention puts the uncertainty on the model | `ppc_fig3a` |
| Fig 3c,d | Δ RNP, Δ slope | 95% CrI, paired within draw | `localnoise_group_posterior` |
| Fig 4a | ELPD differences | **dSE** — ArviZ's standard error of the ELPD *difference*, not a CrI | `table1_all16` |
| Fig 4b,c,d | noise functions, ν₁ − ν₂, relative effect | 95% CrI | `pmcpars_curves`, `pmcpars_relative` |
| Fig 5a–c | decision-space maps | point estimates, **no interval shown** | `decision_space` |
| Fig 5d | model line | point estimate | `decision_space` |
| Fig 5d | observed points | ±1 SEM, paired within subject, n = 35 | `behavior_effect_by_safe` |
| PPC figures | model band | 95% posterior predictive interval | `ppc_fig3a` |
| PPC figures | observed points | ±1 SEM across subjects | `ppc_fig3a` |
| Spline ladder | noise functions and contrast | 95% CrI | `pmcpars_curves` |
| Percept distortion | Δ perceived value | 95% CrI | `pmc_percepts_by_order` |

## Deviations, and why

- **Fig 4a uses dSE, not a credible interval.** ELPD differences come with ArviZ's
  `dse`, the standard error of the *difference*, which accounts for the correlation
  between models evaluated on the same data. It is larger than a naive difference of
  the two `se` values and is the right quantity for comparing models. Say "standard
  error of the difference" in the caption, not "credible interval".
- **Fig 5a–c show no uncertainty.** They are 2D maps; a per-pixel interval cannot be
  drawn without a second set of maps. The uncertainty is carried by Fig 5d, which shows
  the same effect in 1D with intervals, and by Fig 4.
- **Ratio bins are Vincentized**, i.e. formed within each participant (and within each
  safe payoff where the figure splits on it), so every participant contributes to every
  bin and n = 35 everywhere. This matters: binning on the pooled ratio instead leaves
  cells covering only 22-32 subjects, because the risky payoffs a participant sees
  depend on their own responses. Vincent averaging averages **both** coordinates — the
  plotted x is the across-subject mean of each participant's own bin mean — so the axis
  stays in ratio units and can carry real ticks. `bin(risky/safe)` in `utils/data.py`
  is built this way, and so is the safe-payoff split in `plot_ppc_fig3a.py`.

## Caption boilerplate

> Points are observed choice proportions, averaged within participant and then across
> the 35 participants; error bars are ±1 SEM of the within-participant IPS − vertex
> difference. Lines and shaded bands are the model's posterior predictive median and
> 95% interval. Credible intervals are highest-density posterior intervals.
