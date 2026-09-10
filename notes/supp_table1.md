# Supplementary Table 1. Model comparison

Expected log predictive density (ELPD, leave-one-out; Vehtari et al., 2017) for every candidate model that met the convergence criterion (r̂ ≤ 1.01 and effective sample size ≥ 400 on all group-level parameters). The upper block varies WHAT the perturbation is allowed to move with the noise function held at the power law; the lower block varies HOW FLEXIBLE the noise function is with the placement held at the reported one. ΔELPD is the PAIRED difference against the model reported in the main text, with the standard error of that difference (dSE) in brackets; positive values favour the alternative. `p_loo` is the effective number of parameters. `Targeted` is the number of the eight targeted posterior predictive statistics that fall inside the model's own 95% predictive interval; `Grid` is the same count over the 34 cells the design itself fixes (5 safe payoffs x 2 presentation orders x 2 stimulation arms, across four views of the same choices). Eleven models score 8/8 on the targeted statistics, so that column is necessary but not diagnostic; `Grid` is the one that separates them.

| Noise function | cTBS affects | ELPD | ΔELPD (dSE) | p_loo | Targeted | Grid | r̂ | ESS |
|---|---|---:|---:|---:|:---:|:---:|---:|---:|
| Power | Perceptual + memory noise, prior means | -4148.1 | +0.0 (1.1) | 193 | 8/8 | 32/34 | 1.000 | 11368 |
| **Power** | Perceptual noise + prior means | -4148.1 | reference | 188 | 8/8 | 32/34 | 1.000 | 11016 |
| Power | Perceptual + memory noise | -4153.9 | -5.8 (4.1) | 183 | 7/8 | 30/34 | 1.000 | 4873 |
| Power | Perceptual noise only | -4155.2 | -7.1 (4.1) | 177 | 6/8 | 29/34 | 1.000 | 5179 |
| Power | Prior means only, no noise change | -4173.8 | -25.7 (6.7) | 170 | 8/8 | 31/34 | 1.000 | 12851 |
| Power | Memory noise only | -4195.0 | -46.9 (10.1) | 168 | 7/8 | 28/34 | 1.000 | 3778 |
| Power | No cTBS effect | -4259.5 | -111.4 (13.7) | 139 | 6/8 | 27/34 | 1.000 | 2422 |
| Power perc. + Weber mem. | Perceptual + memory noise, prior means | -4150.4 | -2.3 (2.8) | 191 | 8/8 | 32/34 | 1.000 | 10898 |
| Power perc. + Weber mem. | Perceptual noise + prior means | -4149.7 | -1.6 (2.7) | 186 | 7/8 | 32/34 | 1.000 | 10556 |
| Power perc. + Weber mem. | Perceptual + memory noise | -4156.1 | -8.0 (4.9) | 182 | 6/8 | 29/34 | 1.000 | 7293 |
| Power perc. + Weber mem. | Perceptual noise only | -4155.0 | -6.9 (4.9) | 175 | 5/8 | 29/34 | 1.000 | 7842 |
| Power perc. + Weber mem. | No cTBS effect | -4259.0 | -110.9 (13.9) | 137 | 6/8 | 27/34 | 1.000 | 11001 |
| Spline, 3 anchors | Perceptual noise + prior means | -4143.1 | +5.0 (5.6) | 215 | 8/8 | 32/34 | 1.000 | 8698 |
| Spline, 4 anchors | Perceptual noise + prior means | -4145.4 | +2.7 (7.7) | 235 | 8/8 | 32/34 | 1.000 | 10303 |
| Spline, 5 anchors | Perceptual noise + prior means | -4158.1 | -10.0 (8.3) | 239 | 8/8 | 32/34 | 1.000 | 10162 |
| Spline, 6 anchors | Perceptual noise + prior means | -4165.1 | -17.0 (9.6) | 248 | 8/8 | 32/34 | 1.000 | 16199 |
| Spline, 7 anchors | Perceptual noise + prior means | -4174.1 | -26.0 (10.1) | 259 | 8/8 | 32/34 | 1.000 | 15352 |
| Smooth spline, 3 anchors | Perceptual noise + prior means | -4146.3 | +1.8 (6.6) | 221 | 8/8 | 32/34 | 1.000 | 6519 |
| Smooth spline, 5 anchors | Perceptual noise + prior means | -4153.0 | -4.9 (9.8) | 248 | 8/8 | 32/34 | 1.000 | 9946 |
| Smooth spline, 7 anchors | Perceptual noise + prior means | -4171.0 | -22.9 (11.1) | 261 | 8/8 | 32/34 | 1.000 | 13089 |
