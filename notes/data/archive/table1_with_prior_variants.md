| Model | ELPD (LOO) | Eff. no. params | Diff. in ELPD | SE | dSE | r&#770; | ESS |
|---|---:|---:|---:|---:|---:|---:|---:|
| flexible2_noisefix_perception_prior | -4154.5 | 251.0 | -0.00 | 47.6 | 0.0 | 1.010 | 420 |
| Flexible PMC (cTBS on perceptual noise only) | -4157.7 | 247.6 | -3.16 | 47.6 | 2.6 | 1.000 | 745 |
| Flexible PMC (cTBS on perceptual and memory noise) ⚠ | -4159.7 | 266.3 | -5.20 | 47.7 | 4.7 | 1.010 | 353 |
| flexible2_noisefix_prior | -4179.2 | 218.9 | -24.68 | 47.4 | 7.4 | 1.000 | 779 |
| Flexible PMC (cTBS on memory noise only) ⚠ | -4217.5 | 250.5 | -63.03 | 47.4 | 12.4 | 1.010 | 386 |
| Flexible PMC null model ⚠ | -4273.2 | 192.5 | -118.72 | 46.8 | 14.6 | 1.020 | 109 |

⚠ failed the convergence gate (r̂ ≤ 1.01, ESS ≥ 400): Flexible PMC (cTBS on perceptual and memory noise); Flexible PMC (cTBS on memory noise only); Flexible PMC null model.
