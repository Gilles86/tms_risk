# Supplementary Table 1. Model comparison

Expected log predictive density (ELPD, leave-one-out; Vehtari et al., 2017) for every candidate model that met the convergence criterion (r̂ ≤ 1.01 and effective sample size ≥ 400 on all group-level parameters). Every model holds the magnitude priors fixed across stimulation sessions. ΔELPD is the PAIRED difference against the model reported in the main text, with the standard error of that difference (dSE) in brackets; positive values favour the alternative. `p_loo` is the effective number of parameters. `PPC` is the number of the seven targeted posterior predictive statistics that fall inside the model's own 95% predictive interval.

| Noise function | cTBS affects | ELPD | ΔELPD (dSE) | p_loo | PPC | r̂ | ESS |
|---|---|---:|---:|---:|:---:|---:|---:|
| **Power** | Noise on both options | -4154.5 | reference | 223 | 5/7 | 1.000 | 1727 |
| Power | Perceptual + memory noise | -4158.9 | -4.4 (7.5) | 194 | 5/7 | 1.000 | 2549 |
| Power | Perceptual noise only | -4159.4 | -5.0 (8.1) | 186 | 4/7 | 1.000 | 3346 |
| Spline, 5 knots | Noise on both options | -4163.0 | -8.5 (9.7) | 293 | 6/7 | 1.000 | 1237 |
| Power | Noise on 1st-presented option | -4185.1 | -30.6 (6.4) | 204 | 6/7 | 1.010 | 1204 |
| Weber, constant ν | Noise on both options | -4190.1 | -35.6 (8.8) | 182 | 5/7 | 1.000 | 3450 |
| Power | Memory noise only | -4211.6 | -57.2 (9.8) | 175 | 6/7 | 1.000 | 3543 |
| Power | No cTBS effect | -4248.9 | -94.5 (12.4) | 156 | 5/7 | 1.000 | 2327 |
| Spline, 3 knots | No cTBS effect | -4251.2 | -96.8 (12.9) | 171 | 4/7 | 1.010 | 1952 |
| Weber, constant ν | No cTBS effect | -4260.2 | -105.7 (13.9) | 134 | 4/7 | 1.000 | 2901 |

**Excluded for non-convergence.** These models were fitted but did not meet the criterion above, so their ELPD is not interpretable and they are not ranked: `log-power-n2` (r̂ = 1.10), `log-spl3-n1n2` (r̂ = 1.07), `log-genweber-n1n2` (r̂ = 1.03).
