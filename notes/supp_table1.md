# Supplementary Table 1. Model comparison

Expected log predictive density (ELPD, leave-one-out; Vehtari et al., 2017) for every candidate model that met the convergence criterion (r̂ ≤ 1.01 and effective sample size ≥ 400 on all group-level parameters). Every model holds the magnitude priors fixed across stimulation sessions. ΔELPD is the PAIRED difference against the model reported in the main text, with the standard error of that difference (dSE) in brackets; positive values favour the alternative. `p_loo` is the effective number of parameters. `PPC` is the number of the seven targeted posterior predictive statistics that fall inside the model's own 95% predictive interval.

| Noise function | cTBS affects | ELPD | ΔELPD (dSE) | p_loo | PPC | r̂ | ESS |
|---|---|---:|---:|---:|:---:|---:|---:|
| Power | Perceptual + memory noise | -4153.9 | +1.3 (1.5) | 183 | 6/7 | 1.000 | 4873 |
| **Power** | Perceptual noise only | -4155.2 | reference | 177 | 5/7 | 1.000 | 5179 |
| Spline, 5 knots | Noise on both options | -4170.5 | -15.3 (12.1) | 283 | 6/7 | 1.000 | 2822 |
| Power | Noise on 1st-presented option | -4173.1 | -17.9 (10.4) | 204 | -- | 1.010 | 1585 |
| Weber, constant ν | Noise on both options | -4184.4 | -29.2 (12.0) | 179 | -- | 1.000 | 9083 |
| Power | Memory noise only | -4195.0 | -39.8 (9.5) | 168 | -- | 1.000 | 3778 |
| Power | No cTBS effect | -4250.6 | -95.5 (14.1) | 154 | -- | 1.000 | 1419 |
| Spline, 3 knots | No cTBS effect | -4254.5 | -99.3 (15.0) | 174 | 5/7 | 1.010 | 1235 |
| Weber, constant ν | No cTBS effect | -4262.7 | -107.5 (15.0) | 135 | -- | 1.000 | 5350 |

**Excluded for non-convergence.** These models were fitted but did not meet the criterion above, so their ELPD is not interpretable and they are not ranked: `log-power-n1n2.mapjitter.klw` (r̂ = 1.12), `log-power-n2.mapjitter.klw` (r̂ = 1.05), `log-spl3-n1n2.mapjitter.klw` (r̂ = 1.06), `log-genweber-n1n2.klw` (r̂ = 1.14).
