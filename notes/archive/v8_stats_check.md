# Stats check: the v8 Results against the code that produced them

Audited 2026-07-29/30 against `notes/paper/TMS_paper_v8_with_CR_comments.pdf`.
Every number was traced to a notebook cell and re-run. All 12 now reproduce from
disk (§4 covers the recovery needed to get there).

**Three things to fix in the manuscript**, none of which changes a conclusion:

1. `p = 0.004` for the r = 0.76 correlation comes from a *different analysis* — §2
2. The trial-order labels in the two split correlations are swapped — §3
3. `r(34)` should be `r(33)` in all five correlations (N = 35 ⇒ df = N − 2)

Plus: both the preferred-numerosity pair (17.8 / 14.8) and the dispersion pair
(0.9 / 0.77) are correct numbers assigned to the **wrong conditions** — §5.

---

## 1. Verification table

| # | Quantity | Source cell | Code returns | Manuscript | |
|---|---|---|---|---|---|
| 1 | nPRF amplitude | `analyze_encoding_model.ipynb` · `pairwise_tests` loop | median vertex **1.3015** → ips **1.0416**; t = **1.9924**, p₂ = 0.0544 ⇒ **p₁ = 0.0272** | 1.30→1.04, t(34)=1.99, p=0.027 | ✅ |
| 2 | Preferred numerosity | same cell (`mu`) | t = **1.0069**, p₂ = **0.3211**; natural-space **ips 17.79 / vertex 14.75** | 17.8→14.8, t(34)=1.00, p=0.32 | ⚠️ §5 |
| 3 | nPRF dispersion | same cell (`sd`) | t = **1.2780**, p₂ = **0.2099**; **ips 0.9083, vertex 0.7733** | 0.9→0.77, t(34)=1.27, p=0.21 | ⚠️ §5 |
| 4 | Explained variance | same cell (`r2`) | median vertex **0.0681** → ips **0.0493**; t = **2.0644**, p₂ = 0.0467 ⇒ **p₁ = 0.0233** | 6.8%→4.9%, t(34)=2.06, p=0.023 | ✅ |
| 5 | Prop. voxels cvR² > 0 | same nb · `pg.ttest(..., 'less')` | mean vertex **0.1113** → ips **0.0752**; t = **1.9893**, **p₁ = 0.0274** | 11.1%→7.5%, t(34)=1.99, p=0.027 | ✅ |
| 6 | Decoding accuracy | `analyze_decoding.ipynb` · `rm_anova` | vertex **0.141897**, ips **0.092318**; **F(1,34)=4.9921, p=0.0321** | r=0.142 vs 0.092, F=4.99, p=0.032 | ✅ |
| 7 | Decoding × order | same nb · 2-way `rm_anova` | **F(1,34)=0.8613, p=0.3599** | F=0.86, p=0.360 | ✅ |
| 8 | Indifference × consistency | `correlation_preference_noise.ipynb` · cell 3 | **r = 0.7561**, **p = 1.497 × 10⁻⁷** | r(34)=0.76, **p=0.004** | ❌ §2 |
| 9 | Δconsistency × Δrisk attitude | same nb · final cell (`'less'`) | **r = −0.5107590**, **p₁ = 0.0008588** | r(34)=−0.51, p<0.001 | ✅ |
| 10 | …risky **second** | same cell, by `risky_first` | `risky_first=True` (= risky **FIRST**): r = **−0.5944470**, p₁ = **0.0000831** | r(34)=−0.59, p<0.001 | ⚠️ §3 |
| 11 | …risky **first** | same cell | `risky_first=False` (= risky **SECOND**): r = **−0.3285178**, p₁ = **0.0270015** | r(34)=−0.32, p=0.027 | ⚠️ §3 |
| 12 | Δamplitude × Δnoise (7/10/14) | `neurobehavioral_correlates.ipynb` · cell 4 | **r = −0.379198**, **p₁ = 0.012335** (sign flips to +0.38 under the paper's "decrease × increase" framing) | r(34)=0.38, p=0.012 | ✅ |

**Tails.** Items 1–4 come from a cell that runs `alternative='two-sided'`; the paper
halves it for items 1 and 4. Item 11's 0.027 is likewise the halved 0.0540029. Only
item 5 is natively one-sided. Defensible given the directional hypotheses, but the
code as written does not compute those one-sided values.

**Items 1 and 5 are not a copy-paste duplication.** Different dependent variables,
different voxel sets (item 5 is unmasked), coincidentally close: t = **1.9924** /
p₁ = **0.0272** vs t = **1.9893** / p₁ = **0.0274**.

**Item 12 description.** The paper says "the three smallest safe payoffs (7/10/14)";
the code averages the contiguous range x = 7…14 of `perceptual_noise_sd`.

---

## 2. Where `p = 0.004` comes from

`p = 0.004` is impossible for r = 0.76 at N = 35 (t(33) = 6.65, p ≈ 1.5 × 10⁻⁷).
Checked against the same data: spearman, kendall, bicor, percbend, shepherd,
skipped, a 20 000-shuffle permutation test, both one-sided tails, within-condition
subsets and `pg.rm_corr` — all between 4 × 10⁻⁸ and 0.011, none near 0.004.

The one thing that matches is the **MAP point-estimate version of the same
correlation** (`derivatives/map_models/psychometric_simple.csv`, γ = 1/(2√ν)):
**r = 0.4716, p = 0.004234**. So the *r* is the hierarchical Bayesian fit and the
*p* is the MAP fit. Report one pair or the other — the Bayesian one is what the
rest of the paper uses:

> …correlated highly with choice consistency (*r*(33) = 0.76, *p* < 0.001).

---

## 3. The trial-order labels are swapped

`risky_first` is `p1 == 0.55` (`utils/data.py:267`) — **True means the risky option
came first**, and every mapping in the repo agrees.

| code | r | manuscript says |
|---|---|---|
| risky presented **first** | −0.5944 | "presented **second**" |
| risky presented **second** | −0.3285 | "presented **first**" |

Both coefficients are right; each is attached to the wrong condition. This
propagates to the summary sentence three lines later ("the effect only occurs when
safe options are presented first"), which inverts too. Per the code the stronger
coupling is on **risky-first** trials.

---

## 4. Degrees of freedom, and the recovery

**df.** N = 35 ⇒ df = 33; the paper prints `r(34)`. The reported p-values only
reconcile at df = 33 (item 11: 0.0270 at df 33 vs 0.0252 at df 34; item 12: 0.0123
vs 0.0111), which confirms the analyses ran on n = 35 and only the printed df is
wrong. The t-test and ANOVA df of 34 are correct.

**Recovery.** Three commits on `cleanup/ddm-port` had made items 1–5 and 12
un-rerunnable; as of 2026-07-30 they reproduce again.

- `ead9b9c` stripped the notebook outputs documenting items 1–5 — recover with
  `git show ead9b9c^:tms_risk/encoding_model/notebooks/analyze_encoding_model.ipynb`.
- `ba58cb1` swapped `get_prf_parameters_volume(...)` for
  `get_prf_parameters(model_label=1, ...)`. **Under model 1 only `amplitude` varies
  per session** — `mu`/`sd`/`r2`/`cvr2` are session-invariant by construction, so
  items 2–5 return `t = NaN`. Use `model_label=2` for genuine per-session `mu`/`sd`.
- `derivatives/encoding_models/prf_parameters_thr.tsv` was overwritten 2026-05-22
  by the refactored notebook, which flipped item 12 from r = −0.38 to r = +0.11.
  The pre-overwrite copy is backed up alongside it as
  `prf_parameters_thr.model-1_20260522.bak.tsv`.

The only genuinely missing input was the log-space **cvR² maps**; all 350 parameter
files and 105 cached ROI masks survived locally. The cvR² maps came from the
department SMB archive (not the cluster, which was down):

```
/Volumes/g_econ_department$/projects/2022/dehollander_moisa_ruff_ipsriskydecisionmaking/
    data/ds-tmsrisk/derivatives/encoding_model.cv.denoise.smoothed/
```

68 of 70 were there; the two it lacked (`sub-45/ses-2`, `sub-72/ses-2`) were the two
that had survived locally. Pulled with `rsync --ignore-existing`.

Restored re-run vs. the git-recovered saved outputs — all across-subject medians
match **exactly**, t-statistics to the 3rd–4th decimal, every rounded value unchanged:

| | restored | saved |
|---|---|---|
| amplitude | t = 1.992815, p₁ = 0.027179 | t = 1.992434, p₁ = 0.027201 |
| `mu` | t = 1.008787, p₂ = 0.320202 | t = 1.006935, p₂ = 0.321079 |
| `sd` | t = 1.275565, p₂ = 0.210756 | t = 1.278041, p₂ = 0.209892 |
| `r2` | t = 2.064647, p₁ = 0.023325 | t = 2.064371, p₁ = 0.023340 |
| prop. cvR² > 0 | t = 1.988019, p₁ = 0.027455 | t = 1.989332, p₁ = 0.027379 |
| Δamp × Δnoise | **r = −0.379198, p₁ = 0.012335** | r = −0.379173, p₁ = 0.012340 |

Two traps worth remembering:

- **`get_volume_mask()` loads the session's fMRIPrep EPI brain mask
  unconditionally**, before the `ips_masks` cache check, so it fails for sessions
  2/3 whose masks were pruned. On a cache hit it *returns* the cached ROI mask and
  discards `base_mask`, so reading
  `ips_masks/sub-XX/func/ses-N/sub-XX_space-T1w_desc-<ROI>_mask.nii.gz` directly is
  exactly equivalent.
- **`get_sd_curve()` under current bauer does not reproduce the published noise
  curves.** Both bauer versions build a real spline basis, but the knot anchoring
  moved: `ecc6454` rebuilt it per call as `bs(x, degree=3, df=5,
  include_intercept=True, lower_bound=min_n, upper_bound=max_n)` over min/max of
  both `n1` and `n2`, while HEAD fixes `design_info` at construction against the
  paradigm column. On identical inputs that is r = **−0.3792** (ecc6454 formula,
  matching the paper) vs r = **−0.3167** (HEAD's `get_sd_curve`). Rebuild the basis
  with patsy and dot it with the trace's five spline coefficients.

---

## 5. Items 2 and 3: descriptives

Both t-statistics reproduce exactly; the descriptives do not.

**Item 3.** 0.9 / 0.77 are real — the across-subject medians of `sd` — but **0.9083
is IPS and 0.7733 is Vertex**. Since t = +1.278 (ips − vertex), cTBS went with a
non-significant *increase* in dispersion. Written "from 0.9 to 0.77" in a paragraph
whose convention is vertex → parietal, it reads as a decrease.

**Item 2.** Resolved: 17.8 / 14.8 is the across-subject **median of the per-subject
mean of exp(mu)** — i.e. preferred numerosity averaged per subject in natural
units, then a median across subjects. From the restored data: **IPS = 17.79,
Vertex = 14.75**. Correct numbers, same inversion as item 3 — the paragraph's
convention is vertex → parietal, so it should read "from 14.8 to 17.8".

Note the paper pairs a natural-space descriptive with a log-space test: t = 1.007
(p = 0.32) is the paired test on log `mu`, whereas the same test on natural-space
mu gives t = 1.541, p = 0.133. Testing in the space the model is fit in is the
right call, but the Methods should say so, since the reader sees natural units.
