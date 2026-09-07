# Audit: is the ΔP map in Figure 5 too small?

**Date** 2026-08-31 · **Model** `log-power-n1n2` · **Trace**
`/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor/model-log-power-n1n2_trace.netcdf`
(35 subjects, 4 chains × 3000 draws) · **Figure**
`/Users/gdehol/git/tms_risk/notes/figures/fig5_paper_anchor.pdf` · **Script under audit**
`/Users/gdehol/git/tms_risk/tms_risk/behavior/scripts/extract_anchor_decision_map.py`

## Verdict

**`extract_anchor_decision_map.py` computes ΔP correctly. There is no bug in panel d or
panel e.** The apparent 5× shortfall is a *product-of-maxima* artefact: panel b and panel c
peak in opposite corners of the (safe payoff × ratio) plane, so multiplying their maxima
overstates the largest achievable cellwise product by 4×. Evaluated cell by cell the chain
rule *holds*, and at its peak ΔP is 1.2× **larger** than the chain-rule product, not 5×
smaller.

There is one real defect, and it is a **labelling/units defect in panel c, not an arithmetic
error**: `leverage` is dP per unit **objective** log-ratio, while `ratio_shift` is a shift in
the **perceived** (posterior) log-ratio. The two panels are therefore not in
chain-rule-compatible units, and multiplying them *understates* the bias term by 1/w_R =
1.20–1.42×. Fix below; ΔP itself is unaffected.

## 0. The closed form is verified against the PyMC graph

Two independent gates, both passed.

* **Reproduction.** A standalone re-derivation of `p_vertex`, `p_ips`, `ratio_shift` and
  `leverage` from the same posterior reproduces the shipped
  `notes/data/decision_map/decision_map.log-power-n1n2.tsv` to floating-point
  (max |Δ| = 2.2e-16 on `ratio_shift`, exactly 0 on the other four).
* **Cross-check against simulated choices.** The same algebra evaluated on the **actual
  8335 trials**, compared with `notes/data/ppc_anchor/ppc_anchor.stakerung.log-power-n1n2.tsv`
  (which comes from the PyMC graph via `pm.compute_deterministics`, then simulated choices):

  | | value |
  |---|---|
  | grand-mean P(risky): analytic vs observed | 0.5584 vs 0.5598 (gap −0.0014, tripwire 0.02) |
  | per-cell P vs PPC, 24 (order × stake × rung) cells | max abs diff 0.0037 (ips) / 0.0061 (vertex) |
  | **ΔP: analytic counterfactual vs PPC** | **r = 0.9913, rms 0.0019, max 0.0036** |

  Cell-level ΔP (risky second, low stake), PPC vs analytic:
  `+0.0232/+0.0207, +0.0212/+0.0218, +0.0154/+0.0161, +0.0103/+0.0107, +0.0064/+0.0079,
  +0.0017/+0.0037`. The residual is fully explained by the ips/vertex cells containing
  different trials (the counterfactual re-evaluates the *same* trials under both conditions)
  plus 400-draw Monte-Carlo noise.

  This also confirms the condition coding is right: `sigma_at`/`prior_par` treat the
  intercept as **ips** and add the regressor for **vertex** (patsy Treatment coding,
  `ips` alphabetically first). A flipped coding would give r ≈ −1 here.

## 1. The chain-rule check — where it holds, where it breaks

Straight from the shipped TSV, `pred = leverage × log(ratio_shift)`:

| order | r(ΔP, pred) | max\|pred\| | max\|ΔP\| | **max(lev) × max(log shift)** |
|---|---|---|---|---|
| Risky first | 0.875 | 0.0366 | 0.0486 | **0.1035** |
| Risky second | 0.629 | 0.0427 | 0.0532 | **0.1690** |

**This is the whole "5×".** The PI's 1.1 × log(1.08) ≈ 0.085 pairs the maximum of panel c
with the maximum of panel b, but those live in opposite corners:

| order | where leverage peaks | shift there | where shift peaks | leverage there |
|---|---|---|---|---|
| Risky first | 0.982 at n_safe=7, ratio 1.71 | 0.982 (log −0.018) | 1.111 at n_safe=28, ratio 3.50 | 0.344 |
| Risky second | 1.151 at n_safe=7, ratio 1.79 | 1.032 (log +0.032) | 1.158 at n_safe=28, ratio 3.50 | 0.291 |

Leverage is largest where choice is near indifference — low ratio, low stake — and the
perceived-ratio shift is largest at high ratio and high stake, where the psychometric curve
has already saturated. The product of the maxima is unreachable. Compare like with like and
the map is if anything *more* responsive than the chain rule predicts: at the peak cell,
ΔP = +0.0532 vs a cellwise product of +0.0311 (risky second, n_safe=7, ratio 1.37).

## 2. Which quantity is inconsistent — panel c's units

`leverage = φ(index) · w_R / diff_sd` (line 122) is `dP / d log(**objective** ratio)`: the
`w_R` factor is the shrinkage of the percept toward the prior, i.e. how much a change in the
*presented* payoff moves the *posterior*. Verified by finite-differencing `p_vertex` along the
ratio axis — the formula tracks the true derivative to within 1–14% (median ratio
fd/lev = 0.86–0.99). **Panel c is a correct map of sensitivity to the objective ratio.**

But `ratio_shift` is `exp((post_R − post_S)_ips − (post_R − post_S)_vertex)` — a shift already
expressed in **posterior** units. The correct first-order bias term is therefore

    dP_bias ≈ φ(index) · Δlog(perceived ratio) / diff_sd  =  leverage × Δlog / w_R

so panel b × panel c is short by a factor w_R. Median w_R = 0.705 (risky first) / 0.834
(risky second) → the naive product understates by **1.42× / 1.20×**. With the correction:

| order | r(ΔP, lev·Δlog) | slope | **r(ΔP, φ·Δlog/diff_sd)** | **slope** |
|---|---|---|---|---|
| Risky first | 0.875 | 0.89 | **0.907** | **1.18** |
| Risky second | 0.629 | 0.96 | **0.908** | **1.01** |

The corrected chain rule tracks ΔP with slope ≈ 1 in both orders. Note this correction makes
the *expected* ΔP larger, so it cannot be the source of a shortfall — it deepens the puzzle
that section 1 dissolves.

## 3. Bias vs scaling decomposition (hypothesis (b))

Per subject × draw, with `num = post_R − post_S + log 0.55` and `den = diff_sd`:

* bias only: `Φ(num_i/den_v) − Φ(num_v/den_v)`
* scale only: `Φ(num_v/den_i) − Φ(num_v/den_v)`
* remainder: second-order curvature of Φ (exact identity per element)

cTBS does move the decision SD: `den_ips/den_vertex` median **1.053** (risky first) /
**1.043** (risky second), range 0.949–1.137. Peak cells:

| order | cell | ΔP | bias | scale | 2nd order |
|---|---|---|---|---|---|
| Risky first | n_safe=28, ratio 3.50 (peak \|ΔP\|) | +0.0486 | +0.0389 | −0.0062 | +0.0151 |
| Risky first | n_safe=7, ratio 1.71 (max leverage) | −0.0113 | −0.0121 | +0.0049 | −0.0040 |
| Risky second | n_safe=7, ratio 1.37 (peak \|ΔP\|) | +0.0532 | +0.0545 | +0.0118 | −0.0130 |
| Risky second | n_safe=28, ratio 3.50 (max shift) | +0.0518 | +0.0399 | −0.0007 | +0.0120 |

Grid means (what panel e averages):

| order | bias | scale | 2nd order | ΔP |
|---|---|---|---|---|
| Risky first (n_safe 7 → 28) | −0.0134 → +0.0202 | +0.0023 → −0.0039 | −0.0010 → +0.0035 | −0.0119 → +0.0200 |
| Risky second (n_safe 7 → 28) | +0.0179 → +0.0118 | −0.0019 → +0.0019 | +0.0004 → +0.0002 | +0.0165 → +0.0139 |

**Hypothesis (b) is real but is not the reason panel e is small.** The scaling term is
median 29–40% of the bias term in magnitude and is opposite-signed in 32–45% of cells,
because widening `diff_sd` pushes P *toward* 0.5 — up where p_vertex < 0.5, down where
p_vertex > 0.5. That is exactly why ΔP flips sign along the ratio axis (risky second,
n_safe=7, ratio 2.24: bias −0.002, scale −0.015, ΔP −0.004, despite a shift of 1.022). But
**averaged over the ratio grid the scaling term is −0.004…+0.002, an order of magnitude
below the bias term (+0.009…+0.020)** — it redistributes ΔP across the ratio axis rather
than cancelling it.

Second-order curvature is *not* negligible: median |2nd order| 0.007–0.008, up to 0.016, i.e.
comparable to the bias term itself in mid-range cells. That is a further reason a strictly
first-order chain rule cannot be expected to hold cell by cell.

Also note (a) from the brief: leverage *is* evaluated at the vertex index while the shift
moves it, and with log-shifts up to 0.147 the index moves by up to ~0.4 SD — that is the
second-order term above, and it is signed *toward* larger ΔP at the high-ratio corner.

## 4. Panel e's averaging — no hidden concentrated effect

Mean ΔP per safe payoff over the full plotted grid (1.2–3.5) vs restricted to the ladder rungs
actually presented (1.52–3.24):

| order | n_safe | grid 1.2–3.5 | ladder 1.52–3.24 |
|---|---|---|---|
| Risky first | 7 / 10 / 14 / 20 / 28 | −0.0119 / −0.0033 / +0.0050 / +0.0136 / +0.0200 | −0.0181 / −0.0064 / +0.0046 / +0.0151 / +0.0234 |
| Risky second | 7 / 10 / 14 / 20 / 28 | +0.0165 / +0.0130 / +0.0122 / +0.0125 / +0.0139 | **+0.0069 / +0.0056 / +0.0074 / +0.0098 / +0.0139** |

Restricting to the presented range makes risky-second **smaller**, not larger: the model's
largest risky-second ΔP sits at ratios 1.2–1.5, *below the lowest ladder rung* (1.52). Panel d's
hot corner in the risky-second row is therefore extrapolation beyond the paradigm and should be
read as such. For risky-first the restriction slightly amplifies the effect.

Ground truth on the real paradigm (analytic ΔP on the observed trials, matching the PPC to
0.002): risky second **+0.0135** (low stake) / **+0.0147** (high stake); risky first **−0.0086**
/ **+0.0051**. So panel e's ≈ +0.015 is the honest number for the experiment as run — the peak
of 0.05 is a single corner of the plane, not a typical trial.

## Recommended fix (not applied)

`tms_risk/behavior/scripts/extract_anchor_decision_map.py`, **line 122**:

```python
lev = norm.pdf(v['index']) * v['wR'] / v['diff_sd']
```

If the figure's argument is meant to be multiplicative (b × c ≈ d), drop the `w_R`:

```python
lev = norm.pdf(v['index']) / v['diff_sd']          # dP / d log(PERCEIVED ratio)
```

and relabel panel c `dP / d log(perceived ratio)` in
`tms_risk/behavior/scripts/plot_fig5_paper_anchor.py` (the `'Choice sensitivity\ndP / d log(ratio)'`
colorbar label) and the docstring line "leverage dP(risky) / d log(ratio) at vertex".
If panel c is deliberately sensitivity to the *objective* ratio, keep line 122 and instead
stop the caption from inviting the multiplication. **Either way panel d and panel e are correct
and need no change.**

Two secondary notes for whoever regenerates this:

* The TSV ships only the **marginal median** of each quantity, so any cell-by-cell arithmetic
  across columns mixes marginals. Element-level chain-rule residual: median 0.0075 / 0.0058
  (risky first / second); the same residual from marginal-median arithmetic: 0.0055 / 0.0101.
  Real but not dominant.
* `p_vertex` is needed to interpret every cell of panels b–d and is already in the TSV but not
  plotted. Overlaying the p_vertex = 0.5 contour on panel d would make the sign flip legible
  and would pre-empt exactly this question.

### Reproduction

Scratch scripts (not in the repo):
`/private/tmp/claude-1763273667/-Users-gdehol-git-tms-risk/7a4cd6b2-f9f6-450d-bc19-228b19df69f7/scratchpad/{audit_dp.py,trial_check.py,chain_check.py,decomp.py,lev_check.py,ppc_cmp.py,final.py,second.py}`.
The two cluster-side steps:

```bash
ssh sciencecluster 'srun -c2 --mem 32G --time 30 --account=zne.uzh bash -lc \
  "PYTHONPATH=/scratch/gdehol/bauer_anchor $HOME/data/conda/envs/tms_risk_cpu/bin/python \
   /scratch/gdehol/audit_dp.py \
   /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor log-power-n1n2 \
   /scratch/gdehol/audit_dp_map2.tsv"'
ssh sciencecluster 'cd $HOME/git/tms_risk && srun -c2 --mem 48G --time 40 --account=zne.uzh bash -lc \
  "PYTHONPATH=/scratch/gdehol/bauer_anchor $HOME/data/conda/envs/tms_risk_cpu/bin/python \
   /scratch/gdehol/trial_check.py \
   /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor log-power-n1n2 \
   /shares/zne.uzh/gdehol/ds-tmsrisk /scratch/gdehol/trialchk"'
```

---

# Audit 2 (2026-09-01): panel G vs panel H in `story_anchor.pdf`

**Model** `log-spl3-percmem` · **Trace**
`/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor/model-log-spl3-percmem_trace.netcdf`
(35 subjects, 4 chains) · **Figure**
`/Users/gdehol/git/tms_risk/notes/figures/story_anchor.pdf` · **Scripts**
`tms_risk/behavior/scripts/extract_anchor_decision_space.py` (panel G) and
`extract_anchor_ppc.py` → `ppc_anchor.delta_stake.*` (panel H), drawn by
`tms_risk/behavior/scripts/plot_story_anchor.py`.

## Verdict

**Panel G is not plotted the right way, and the G-vs-H gap is entirely an averaging
artefact. Neither extraction script has an arithmetic bug.**

The "+9.5% perceived value" in panel G and the "+0.02 ΔP" in panel H are both computed as
*mean over subjects of f(θ_s)*, but they sit on **opposite sides of the Jensen gap**:

| quantity, cell (Risky second, n_safe = 28) | mean over subjects of f(θ_s) — **as plotted** | f(mean over subjects of θ_s) — "average subject" | ratio |
|---|---|---|---|
| `rel_risky` (panel G) | **+9.49 %** | +5.48 % | 1.73 (plotted value **too big**) |
| `dp` (panel H) | **+0.0234** | +0.0764 | 0.31 (plotted value **too small**) |

**1.73 × 3.26 = 5.64× — that is the PI's 5×, to two significant figures.** For the average
subject the chain rule works: leverage 1.176 × log(1.0625) = **+0.0712** predicted against
**+0.0764** actual, agreeing to 7%. For the average over subjects it fails by 3.4×
(0.817 × log(1.1012) = +0.0788 against +0.0234). The PI's hypothesis is confirmed exactly.

## Gate

A per-subject re-derivation that imports the repo script's own `tap()` reproduces the shipped
`notes/data/decision_space/decision_space.log-spl3-percmem.tsv` exactly
(max |Δ| = 0 on `rel_risky` / `rel_safe` / `ratio_shift`, 1.7e-10 on `dp` — the residual is my
`Φ(num/s)` reconstruction versus the graph's own `p`). The graph's internal check
`max |Φ(m/s) − p| = 4.1e-09` also passes.

## 1. G and H are on different cells — but that is *not* the problem

Panel G is indexed by `n_safe` ∈ {7,10,14,20,28}; panel H by per-subject stake tercile
((n_safe+n_risky)/2, means 12.7 / 23.0 / 42.2 CHF). They are not the same partition. **But it
does not matter, because `decision_space` already carries `dp` on panel G's own cells**, and
the two agree:

| | ΔP |
|---|---|
| G's own cell (Risky second, n_safe = 28), `dp` from `decision_space` | **+0.0234** |
| H's cell (Risky second, top stake tercile, 42 CHF), `model` from `delta_stake` | **+0.0233** |

Across all six of H's cells the simulated-choice ΔP and the closed-form ΔP agree to
max 0.0044. **The gap is internal to panel G's own row of the TSV, not between the panels.**
Like-for-like, panel G's cells read (Risky second, n_safe 7→28):
`rel_risky` +0.22 / +0.34 / +1.75 / +4.84 / +9.49 % against `dp` +0.0058 / +0.0059 / +0.0123 /
+0.0208 / +0.0234.

## 2. Both averaging orders, per cell

Posterior medians, all ten cells, `mean_of_f` (as plotted) vs `f_of_mean`:

| order | n_safe | rel_risky (a) | rel_risky (b) | dp (a) | dp (b) | lev (a) | lev (b) |
|---|---|---|---|---|---|---|---|
| Risky first | 7 / 14 / 28 | −0.07 / +1.25 / +7.62 | −0.34 / +0.56 / +4.46 | −0.0026 / +0.0011 / +0.0082 | −0.0182 / +0.0149 / +0.0473 | 0.93 / 1.04 / 0.85 | 1.28 / 1.35 / 1.12 |
| Risky second | 7 / 14 / 28 | +0.22 / +1.75 / +9.49 | +0.04 / +0.99 / +5.48 | +0.0058 / +0.0123 / +0.0234 | +0.0009 / +0.0218 / +0.0764 | 0.93 / 0.98 / 0.82 | 1.37 / 1.26 / 1.18 |

The Jensen gap runs the *same direction everywhere*: `rel_risky` and `ratio_shift` are convex
functions of the log shift, so mean-of-f **exceeds** f-of-mean; `dp` and `lev` are concave in
the relevant range, so mean-of-f **falls short**. It grows with `n_safe`, from ~1% at
n_safe = 7 to 73% (rel_risky) and 3.3× (dp) at n_safe = 28.

## 3. Reconciling G with H arithmetically — the ladder

Focus cell (Risky second, n_safe = 28), each step exact, all averaged over subjects last:

| step | value | × previous |
|---|---|---|
| 0 naive: log(mean ratio_shift) × φ(0)/mean(diff_sd) — **the PI's envelope** | **+0.0724** | — |
| 1 use the real mean leverage: log(mean shift) × mean(lev) | +0.0503 | 0.70 |
| 2 **per-subject product: mean_s(lev_s · Δlog_s)** | **+0.0223** | **0.44** |
| 3 exact bias only: mean_s[Φ(num_i/s_v) − Φ(num_v/s_v)] | +0.0252 | 1.13 |
| 4 + scaling and second order | — | — |
| 5 **exact ΔP = mean_s[Φ(idx_i) − Φ(idx_v)]** | **+0.0238** | — |

**Step 1 → 2 is the whole story: −0.0281, a factor of 0.44, from the across-subject covariance
`cov_s(lev_s, Δlog_s)` (corr = −0.496).** The subjects with the largest perceived-ratio shift
are precisely the subjects whose choices are least sensitive to it. Everything else is small:
within-subject covariance across ladder rungs +0.0004, curvature +0.0029, scaling −0.0071 and
second order +0.0057 (which largely cancel; mean s_i/s_v = 0.998 here, so unlike audit 1 the
decision SD barely moves in this model).

The effective group slope is `mean(Δp)/mean(Δlog) = 0.388` against a mean per-subject leverage
of 0.821 — **a factor 2.1 lost purely to averaging.** Across all ten cells the naive/exact ratio
runs 1.1–3.0 for risky-second and 2.5–18 for risky-first (where mean ΔP is near zero).

**So panel G and panel H are consistent with each other and with the model. The chain rule is
what fails, and it fails because it is applied to two separately-averaged group quantities.**

## 4. `rel_risky` is a mean of a heavily skewed per-subject percentage — report the median

Two distinct problems, both in the same number.

**(i) The exp() convexity.** `rel_risky = mean_s[100·(exp(Δpost_s) − 1)]`. The percentage of the
mean log change is `100·expm1(mean_s Δpost_s)` = **+5.51 %**, against the plotted mean of
percentages **+9.52 %**. Same for `ratio_shift`: exp(mean Δlog) = 1.063 against
mean(exp Δlog) = 1.103.

**(ii) A five-subject tail.** At the focus cell, over 35 subjects:

| statistic | value |
|---|---|
| **mean (plotted)** | **+9.52 %** |
| **median** | **−0.38 %** |
| SD / skew | 31.9 % / +2.84 |
| range | −27.0 % … **+147.7 %** |
| top-1 subject's share of the sum | **44 %** |
| top-3 share | **86 %** |
| top-5 share | **114 %** (the other 30 sum to *negative*) |
| subjects above the plotted mean | **5 / 35** |
| subjects within ±8 % | 24 / 35 |
| mean after dropping the top 5 | **−1.50 %** |

**The median subject shows no change in perceived risky value at all.** The figure should show
the median (`median_of_f` is already computable from the same arrays: +0.05 % at this cell), or
plot Δ log perceived value with the per-subject scatter overlaid. The mean is defensible as a
group estimand, but not as the quantity a reader chains into panel H.

**Who the five are:** they are the *noisiest* participants. Split by the subject's own decision
SD `s_v`:

| quartile of s_v | s_v | leverage | Δlog | rel_risky | ΔP | p_vertex |
|---|---|---|---|---|---|---|
| Q1 (sharpest) | 0.240 | 0.959 | −0.020 | −5.9 % | −0.038 | 0.711 |
| Q2 | 0.315 | 0.980 | −0.000 | +1.1 % | +0.009 | 0.568 |
| Q3 | 0.358 | 0.788 | −0.042 | −1.7 % | −0.026 | 0.582 |
| **Q4 (noisiest)** | **0.440** | **0.555** | **+0.295** | **+43.3 %** | **+0.146** | 0.378 |

`corr(s_v, rel_risky) = +0.65`, `corr(s_v, Δlog) = +0.61`, `corr(lev, Δlog) = −0.50`. The
mechanism is mechanical: a noisier subject has more prior shrinkage, so a given cTBS increase in
noise moves their percept much further — and the same noise makes their psychometric function
flat, so the moved percept barely changes their choices. Panel G's headline is therefore set by
the participants whose choices carry the least information.

## 5. What makes panel G misleading as drawn

1. **It plots the mean of a quantity with skew 2.84 whose median is ~0.** 30 of 35 subjects sit
   below the plotted point.
2. **Its error bars are posterior uncertainty on the group mean, not between-subject spread.**
   The band [+4.6 %, +15.0 %] reads as a precise, well-identified effect; the between-subject SD
   is 31.9 %.
3. **The exp() is applied per subject and then averaged**, inflating +5.5 % to +9.5 % before any
   outlier effect.
4. **Its neighbour panel invites a chain-rule reading that the averaging destroys.** G and H are
   both "mean over subjects", but of functions with opposite curvature, so their ratio is
   inflated 5.6×. A reader doing the arithmetic the PI did will always get ~5×.
5. **The two panels are also on different cell partitions** (n_safe vs stake tercile). They
   happen to coincide numerically here, but nothing guarantees that.

**Suggested fixes (not applied).** In `plot_story_anchor.py` panel G (lines ~185–199), plot
`median_of_f` and overlay the per-subject points; or plot Δ log perceived value. In
`extract_anchor_decision_space.py`, the aggregation at lines 145–153 already has the per-subject
array in hand — emitting `median` and a between-subject SD alongside `mean` would cost nothing
and would let the figure show both. If the multiplicative G→H argument is to be kept, both
panels must be computed for the **same** representative subject (the `f_of_mean` column above),
where the chain rule agrees to 7%.

### Reproduction

Scratch scripts:
`/private/tmp/claude-1763273667/-Users-gdehol-git-tms-risk/7a4cd6b2-f9f6-450d-bc19-228b19df69f7/scratchpad/{space_persubject.py,gate2.py,q1.py,q23.py,q24.py,q5.py}`.

```bash
ssh sciencecluster 'cd $HOME/git/tms_risk && srun -c4 --mem 64G --time 60 --account=zne.uzh bash -lc \
  "PYTHONPATH=/scratch/gdehol/bauer_anchor $HOME/data/conda/envs/tms_risk_cpu/bin/python \
   /scratch/gdehol/space_persubject.py log-spl3-percmem /shares/zne.uzh/gdehol/ds-tmsrisk \
   /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor /scratch/gdehol/psub"'
```
