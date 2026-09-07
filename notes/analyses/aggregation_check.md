# Which aggregation the mechanism panels should use

**2026-09-03.** Triggered by the observation that the prototype's predicted-ΔP
panel disagreed with the model's own posterior predictive. It did. The cause is
now identified, and it is **not** the algebra.

Figure: `notes/figures/diag_agg_log-power-n1n2.pdf` (and `…log-power-n2.pdf`).
Script: `tms_risk/behavior/scripts/extract_anchor_agg_variants.py` +
`plot_agg_diag.py`.

## The question

Figure 5 draws the same object three different ways and they were never
reconciled:

| Panel | Quantity | Trials | Posterior | Across subjects |
|---|---|---|---|---|
| f/g (mechanism) | perceived-ratio shift, decision-SD ratio | real | **per-subject median, plugged in** | mean |
| h/i (PPC) | P(chose risky) | real | **integrated (mean over draws)** | mean |
| prototype panel | predicted ΔP | **uniform ratio grid** | median | mean |

Four axes vary at once. `extract_anchor_agg_variants.py` crosses all of them on
one trace and scores each against the model's own simulated PPC.

## Result — the algebra is right

Evaluated on the **real trials**, **integrating over draws**, averaging over
**subjects**, the closed form reproduces the PPC essentially exactly
(24 order × stake × rung cells):

| Estimator | r vs PPC (n1n2) | max abs. dev. | r vs PPC (n2) |
|---|---|---|---|
| **Integrate over draws, mean** | **0.991** | 0.0036 | **0.994** |
| Integrate over draws, median | 0.989 | 0.0043 | 0.995 |
| Plug in posterior **median** | 0.977 | 0.0065 | 0.986 |
| Plug in posterior **mean** | 0.959 | 0.0074 | 0.979 |

So the reconstruction was never broken. Ranked by how much each choice moves
the answer:

1. **Uniform grid vs real trials — the largest, and it changes a conclusion.**
   The design's risky/safe ratios are a per-subject calibrated ladder, not
   uniform on [1.2, 3.5]. Averaging over a uniform grid re-weights the payoff
   range: mean ΔP for **risky first flips sign, −0.0005 (trials) → +0.0041
   (grid)**, and risky second is compressed 0.0170 → 0.0138 and given a
   spurious downward slope in safe payoff (see panel b, dotted vs solid).
2. **Median vs mean ACROSS SUBJECTS — nearly as large, and also flips a sign.**
   Risky first −0.0013 (mean) → **+0.0045** (median); risky second 0.0140 →
   0.0084, a 40% compression. The subject distribution is right-skewed. The
   mean is the only defensible choice here: the observed data and the PPC are
   both subject means, so a median would not be comparable to either.
3. **Plug-in vs integrate over draws — real, second-order.** Every mechanism
   quantity is nonlinear in the parameters (`w = σ²/(σ² + ν²)`), so the value
   at the median parameter is not the mean value. Flips the sign of the small
   risky-first effect (−0.0013 → +0.0014) and compresses risky second by 21%
   (0.0140 → 0.0111).
4. **Mean vs median OVER DRAWS — negligible.** ≤0.0005 anywhere; r changes in
   the third decimal. Not a factor and not worth a sentence in the paper.

Worth noting: plugging in the **median** beats plugging in the **mean**
(r = 0.977 vs 0.959). These are log-scale regression coefficients whose
posteriors are right-skewed once exponentiated, so the mean overshoots.

## What follows

- **Never average a mechanism quantity over a uniform ratio grid.** Evaluate on
  the trials actually presented. `extract_anchor_decision_function.py` builds a
  grid, and its output is a contour map of the decision rule — not a prediction
  for this experiment, and must not be read as "what cTBS did".
- **Panels f/g should integrate over draws** rather than read per-subject
  medians out of `anchor_priors_subject.tsv` / `anchor_curves_subject.tsv`
  (both written with `np.median`). The 21% compression is survivable; the
  risky-first sign flip is not, since order-specificity is the paper's claim.
- **Always average across subjects with the mean**, never the median.
- Mean vs median over draws needs no decision.

## Reproduce

```bash
# on sciencecluster, where the anchor traces live
PYTHONPATH=/scratch/gdehol/bauer_anchor srun -c2 --mem 24G --time 20 \
  --account=zne.uzh $HOME/data/conda/envs/tms_risk_cpu/bin/python \
  -m tms_risk.behavior.scripts.extract_anchor_agg_variants log-power-n1n2 \
  --trace_dir /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor \
  --out_tsv agg_variants.log-power-n1n2.tsv
```
