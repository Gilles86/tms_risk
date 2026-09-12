# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Analysis code for the combined cTBS-TMS + 3T fMRI (Philips Achieva) study **"Risk Attitudes Causally Rely on Parietal Magnitude Representations"** (de Hollander, Moisa & Ruff). The paper draft is at `notes/paper/TMS_paper_v9.pdf` (audit + open items: `notes/v9_plan.md`). The pipeline targets numerosity-tuned right parietal cortex with cTBS (vertex control vs. parietal) and measures effects on (a) nPRF responses, (b) trial-by-trial decoding accuracy, (c) psychophysical choice consistency / risk-neutral probability, and (d) parameters of the Perceptual-and-Memory-based Choice (PMC) model and its **Flexible PMC** extension (B-spline noise function over magnitude).

Three layered analyses sit on top of the same BIDS dataset (`/data/ds-tmsrisk` locally; `/shares/zne.uzh/gdehol/ds-tmsrisk` on the cluster):

1. `tms_risk/modeling/` — nPRF fits + Bayesian decoding (uses `braincoder`, TensorFlow). Compute-heavy, runs on cluster GPU/CPU.
2. `tms_risk/glm/` — GLMsingle single-trial betas feeding the encoding model.
3. `tms_risk/behavior/` — PMC / Flexible PMC / probit models (uses `bauer` + PyMC/bambi). Hierarchical Bayesian; runs on CPU.

`tms_risk/tms_targeting/`, `prepare/`, `registration/`, `surface/` support the imaging pipeline; `visualize/` holds plotting helpers. Exploratory behavioral notebooks live in `tms_risk/behavior/notebooks/archive/`.

## Repo-specific architecture

**Single source of truth for data access**: `tms_risk/utils/data.py`. Every notebook and script should go through `get_all_behavior(...)`, `get_subjects(...)`, or instantiate `Subject(subject_id, bids_folder)` (defined at line 154). Subject methods accept the cross-cutting flags `denoise`, `smoothed`, `natural_space`, `mask`, `n_voxels` that select between GLMsingle variants — touching the same flag in every analysis script is intentional.

**Outliers are hardcoded** to subjects `[22, 49]` in `tms_risk/utils/data.py` (rationale in `tms_risk/behavior/notebooks/archive/outliers.ipynb`). Pass `exclude_outliers=False` to `get_subjects` / `get_all_behavior` to keep them.

**Model dispatch by string label**: `tms_risk/behavior/fit_model.py::build_model(model_label, df)` maps short labels (`"flexible2.6a"`, `"5c"`, `"session1_full"`, …) to `bauer.models` constructors with specific regressor formulas and `prior_estimate` / `memory_model` options. The submit scripts (`submit_all_flexible_models.sh`, `submit_all_weber_models.sh`, `fit_model.sh`) in `tms_risk/behavior/slurm_jobs/` iterate over these labels for SLURM arrays. When adding a model variant, add a new label rather than mutating an existing one — downstream notebooks load traces by label from `<bids>/derivatives/cogmodels/model-<label>_trace.netcdf`.

**`libs/` contains git submodules** (`bauer/` on `refactor`, `braincoder/` on `main`). The cognitive models in the paper (`FlexibleNoiseRiskRegressionModel`) live in `bauer.models.risky_choice`; DDM and Race-diffusion variants of the same model exist in `bauer.models.{ddm,race}` (`DDMFlexibleNoiseRiskRegressionModel`, `RaceDiffusionFlexibleNoiseRiskRegressionModel`).

**Conda environments**: three of them, all installing the same `pymc=5.17` / `bambi=0.13` / `arviz=0.20` stack used to fit the paper's models.

| Env name | YML | Use case |
|----------|-----|----------|
| `tms_risk` | `environment_apple_silicon.yml` (top-level) | Local Mac dev (Metal-accelerated TF) |
| `tms_risk_cpu` | `create_env/environment_cpu.yml` | Cluster CPU jobs (PMC fits, plotting) |
| `tms_risk_cuda` | `create_env/environment_cuda.yml` | Cluster GPU jobs (nPRF fits, decoding) |

## Common commands

```bash
# Local install (after creating env)
pip install -e .

# Fit one cognitive model locally
python -m tms_risk.behavior.fit_model flexible2.6 --bids_folder /data/ds-tmsrisk

# Submit full flexible-model sweep on cluster
cd ~/git/tms_risk/tms_risk/behavior/slurm_jobs && bash submit_all_flexible_models.sh

# Build cluster envs (CUDA env must run on a GPU node)
sbatch create_env/create_cpu_env.sh
sbatch create_env/create_gpu_env.sh
```

There are no automated tests; correctness is checked via posterior predictive notebooks under `tms_risk/behavior/notebooks/` (e.g. `analyze_flexible_model_ppcs.ipynb`, `figure4.ipynb`).

## Big model fits: the ScienceCloud VM

**Run every long cognitive-model fit on `ssh sciencecloud`, not locally.** It is a
plain Ubuntu VM (15 cores, 58 GB RAM) with **no SLURM** — launch with `nohup`, not
`sbatch`.

| What | Where |
|---|---|
| BIDS data | `/data/ds-tmsrisk` — already present, all 917 event TSVs |
| Repo | `/data/git/tms_risk` (bauer submodule at `libs/bauer`) |
| Conda | `/data/miniforge3`, env `tms_risk_behavior` |
| Env spec | `create_env/environment_cloud_behavior.yml` |
| Logs | `/data/logs/` |

The env is **behaviour-only** — no tensorflow, braincoder or fmriprep deps, so it
solves in seconds. It still needs `nibabel`/`nilearn` (imported at the top of
`tms_risk.utils.data`) and `seaborn`/`matplotlib-base<3.11` (imported at the top of
`bauer.utils`, and arviz 0.20 breaks on matplotlib ≥ 3.11). `bauer` is deliberately
**not** pip-installed: these fits are version-sensitive (see below), so scripts put a
pinned checkout on `sys.path` themselves.

```bash
# ship local edits (macOS rsync chokes on multiple --exclude + --delete-excluded)
tar czf - --exclude __pycache__ tms_risk create_env \
  | ssh sciencecloud 'tar xzf - -C /data/git/tms_risk'

# launch; ~4 cores per model, so 3-4 models fit concurrently
ssh sciencecloud 'cd /data/git/tms_risk && nohup \
  /data/miniforge3/envs/tms_risk_behavior/bin/python -m tms_risk.behavior.fit_model \
  flexible2.6 --bids_folder /data/ds-tmsrisk > /data/logs/flexible2.6.log 2>&1 &'
```

A 5000-tune + 5000-draw × 4-chain Flexible PMC fit takes roughly 3–5 h on the CPU VM
(pymc, 3 cores per model, 4 models in parallel).

### The T4 GPU nodes

Four further VMs (SSH aliases `sciencecloud_gpu` … `sciencecloud_gpu4`, each a Tesla
T4, 8 cores, 31 GB) run the same fits through the **numpyro/JAX** backend
(`fit_pmc_noisefix … --backend numpyro`). Same layout: `/data/git/tms_risk`,
`/data/ds-tmsrisk`, `/data/logs`, conda env `tms_risk_gpu` under `/data/miniforge3`.
**Never write their IPs into this repo — it is public.** Addresses, disk layout and
the conda/CUDA specifics live in the private `~/.claude/reference/sciencecloud-vms.md`;
the SSH aliases are enough for anything written down here. One GPU means one model at a
time per box, but a
5000+5000 × 4-chain `flexible1` fit takes ~1.5–2 h there versus ~15 h+ on the CPU VM,
so a four-model nested family finishes overnight. `libs/bauer` on that node is
checked out at the commit the traces are stamped with.

Overnight outputs land in `derivatives/cogmodels.overnight/`, kept apart from both
`cogmodels/` (published) and `cogmodels.noisefix/`.

### Extract remotely, plot locally

The traces are ~1.2 GB each with `log_likelihood`, so never rsync them. Run the
extraction on whichever node holds them and pull the TSVs. Every extraction script
takes `--trace_dir` (read from somewhere other than `derivatives/cogmodels`) and
`--tag` (name the outputs something short):

```bash
python -m tms_risk.behavior.scripts.summarize_traces \
    --trace_dir /data/ds-tmsrisk/derivatives/cogmodels.overnight \
    --pattern 'flexible1_noisefix*' --loo --out_tsv /data/summary_flexible1.tsv
```

`summarize_traces` is the morning check: one row per trace with max r̂, min ESS,
divergences, and the `tms_risk_bauer_commit` / `tms_risk_family` /
`tms_risk_constrained` stamps, plus an ArviZ LOO comparison across every trace that
carries a `log_likelihood` group. It gates on r̂ ≤ 1.01 and ESS ≥ 400 over the
group-level parameters only (per-subject offsets would swamp the table).

Then `plot_pmc_explained` rebuilds the whole mechanism figure from the pulled TSVs
alone — no trace, no bauer, no GPU. See `notes/pmc_refit_results.md`.

## Interactive surface viewers (the paper's web companion)

`~/git/tms_risk_viewers` is a static pycortex site: group maps on fsaverage, one
viewer per participant on their own flattened surface, a gallery, and the Figure-2b
amplitudes. Rebuild it with (details and inputs in `notes/PROVENANCE.md`):

```bash
~/mambaforge/envs/tms_risk/bin/python -m tms_risk.surface.sample_model1_to_surface      # NPZ maps
~/mambaforge/envs/pycortex2/bin/python -m tms_risk.visualize.import_flatmaps            # once
~/mambaforge/envs/pycortex2/bin/python -m tms_risk.visualize.make_static_viewers --out_dir ~/git/tms_risk_viewers
```

- pycortex subjects are `tms.sub-XX`, imported from the **local**
  `derivatives/freesurfer` recon; the cluster's `fmriprep/sourcedata/freesurfer` is a
  different recon (sub-45 differs) and has no `surf/`. The flatmaps came from
  `surface/slurm_jobs/autoflatten.sh` on sciencecluster with those surfaces uploaded;
  its `autoflatten_xla_fix/sitecustomize.py` is required (jaxlib 0.11 aborts on an XLA
  flag autoflatten sets, surfacing only as `BrokenProcessPool`).
- fsaverage's pycortex cache is shared with other projects: `build_group` sets it
  aside and restores it, because a custom `overlay_file` needs `recache=True`.
- Chrome caches the bundle's `*_[inflated]_*.svg`/`.ctm` across rebuilds; check a
  rebuild on a fresh port or a `Cache-Control: no-store` server.

## The choice rule is KLW-consistent, and there is only one model set

Since 2026-09-08 `fit_anchor.py` fits the **KLW-consistent decision noise by
default**: the decision variable is the noisy posterior mean, so its SD is
`w·nu` with `w = sd_prior^2/(sd_prior^2 + nu^2)`, and the comparison is
normalised by `sqrt((w1 nu1)^2 + (w2 nu2)^2)` (`bauer/core.py:169-197`, via
`posterior_mean_sd` at `bauer/utils/bayes.py:26-47`). bauer's historical rule
shrinks the numerator by `w` and leaves the denominator raw, which makes the
prior width change the psychometric SLOPE as a pure artefact of the
normalisation — so `nu` does not denote the same quantity in two models with
different prior widths, and the model set is not comparable within itself.

`--raw_choice_noise` still fits the old rule, but it prints a `RuntimeWarning`
banner and tags its output `.rawnoise`. **Never mix `.rawnoise` into an ELPD
ladder, a Supp table or a figure.** KLW outputs keep the historical `.klw`
suffix so nothing fitted before the switch had to be renamed.

Everything fitted under the old rule was moved aside, not deleted:

| What | Where it went |
|---|---|
| 157 traces (143 GB) | `<bids>/derivatives/cogmodels.anchor.rawnoise/` |
| 992 derived TSV/NPY | `notes/data/archive_rawnoise/` |
| 36 figures | `notes/figures/archive_rawnoise/` |

## Cognitive-model traces are bauer-version-sensitive

**A stored PMC trace only means something against the bauer commit that produced
it, and mismatches fail silently.** Loading `model-flexible2.6_trace.netcdf` into a
graph built by current bauer raises no error — every free parameter is present, the
design matrices match `constant_data` exactly, subject-level parameters recompute to
machine precision — yet predicted choice probabilities are off by +0.12 (grand mean
0.65 vs an observed 0.53).

The published `flexible2.6` fits (2024-11-05) need **`bauer@ecc6454`**. Two things
changed afterwards, neither renaming a parameter:

1. `b66c806` (2026-04-03) changed the `'payoff'` branch of
   `_get_choice_predictions` from `diff_sd = sqrt(ν1² + ν2²)` to a
   posterior-variance- and probability-scaled form.
2. At the fitting commit, `_get_trialwise_evidence_sd` used `labels1` (the *memory*
   splines) for **both** spline terms of `n1_evidence_sd`, so the perceptual noise
   function never entered the first-presented option — contrary to the Methods'
   ν₁ = ν_perceptual + ν_memory. Fixed in the same `b66c806`.

So the noise-composition bug cannot be fixed by checkout alone without also changing
the choice rule. **Because of (1), ν does not denote the same quantity on the two
sides of `b66c806`, and the two fits reach different scientific conclusions**: the
published trace puts the cTBS noise increase on the second-presented option at low
payoffs only, the HEAD refit spreads it evenly over both options and all payoffs
(table in `notes/pmc_refit_results.md`). Both fit the choices and both reproduce the
order-specific behavioural effect. Decide which side the paper reports; do not let
the checkout decide.

### Never pin bauer by checking it out on sciencecluster

`~/git/tms_risk/libs/bauer` on the cluster is pip-installed **editable into
fourteen conda envs**, and not only this project's: `retsupp_neuropythy`,
`retsupp_snake`, `retsupp.old.20260511`, `soglio_cuda`, `value_capture`,
`tf-cpu`, `tf2-cpu`, `tf2-gpu`, `tf2-gpu.bak` all resolve `import bauer` to it,
alongside the five `tms_risk_*` envs. `git checkout <commit>` there changes the
library for every one of them at once, including jobs already running, and
nothing raises — results just quietly change.

To pin a version, clone and use PYTHONPATH (it precedes site-packages, so it
wins over the editable install without touching it):

```bash
git clone ~/git/tms_risk/libs/bauer /scratch/gdehol/bauer_<tag>
git -C /scratch/gdehol/bauer_<tag> checkout <commit>
PYTHONPATH=/scratch/gdehol/bauer_<tag> python -m tms_risk.behavior.fit_model ...
```

The canonical state of the shared checkout is **4cd98a4**; a `post-checkout`
hook and an untracked `SHARED_CHECKOUT_README.md` in that directory warn about
this, but neither can prevent it. Note 4cd98a4 does **not** contain
`LogFlexibleNoiseRiskRegressionModel`, so the log-space `lfx2-*` grid cannot be
refit from the shared checkout at all — those traces were produced by
`dd6feab` and need an isolated clone.

The sciencecloud GPU boxes have no bauer installed in `tms_risk_gpu`, so there
`import bauer` is resolved purely by path — which makes them the cleaner place
to run version-pinned fits.

Two ways forward, both supported by
`tms_risk/behavior/scripts/fit_pmc_noisefix.py`:

- `--variant head` — fit fresh against current bauer. **Preferred for new fits**: a
  fresh posterior is internally consistent with whatever code produced it, so the
  version-pinning problem simply does not arise. Verified that `make_dm` still
  returns a proper 6-column cubic B-spline basis at `spline_order=6`.
- `--variant noisefix` — `ecc6454` + `notes/patches/bauer-ecc6454-noisefix.patch`,
  changing *only* the noise composition. Use to attribute a change to that bug alone.

Outputs go to `derivatives/cogmodels.noisefix/model-<label>.<variant>_trace.netcdf`,
kept away from `derivatives/cogmodels/`, and the bauer commit is stamped into
`trace.posterior.attrs['tms_risk_bauer_commit']`. **Stamp the commit on anything you
fit from now on.** To *re-evaluate* an old trace, pin the commit and gate on a
posterior-predictive grand-mean check — `decompose_pmc_channels.py` aborts if the gap
exceeds 0.02, so a wrong pin cannot pass unnoticed.

## Conventions worth knowing

- **Never use a maximum-likelihood estimator.** Every model in this repo,
  including a throwaway probit fitted "just to check", is **hierarchical
  Bayesian with partial pooling**, and every uncertainty shown is a posterior
  credible interval or a posterior predictive interval — never a bootstrap CI,
  never an s.e.m. bar on an observed point. Two reasons, both load-bearing:
  pooling subjects within a cell FLATTENS the psychometric function because
  participants sit at different indifference points (the exact artefact under
  test), and only a posterior gives a predictive band against which a point
  outside the band means real misfit. Use
  `behavior/scripts/fit_observed_probit_hier.py` (TMS cohort, cells include
  stimulation) or `fit_baseline_probit_hier.py` (session 1, n = 73, cells are
  order × stake), never `statsmodels.GLM`. Report contrasts computed **per
  draw**, not as differences of summaries.

- **Reliability of per-participant estimates: use the draw-pair estimator, not
  variance subtraction.** `behavior/scripts/anchor_subject_reliability.py`
  reports all three. The classical
  `(Var_s[E_d] - median_s Var_d) / Var_s[E_d]` correction subtracts the
  within-subject variance from the spread of the POSTERIOR MEANS, which partial
  pooling has already shrunk — it double-counts shrinkage, and returns exactly
  0.00 for most parameters here. `reliability_draw` / `rank_stability`, the
  correlation between two independent posterior draws of the whole subject
  vector, cannot go negative and equals the reliability directly. On a synthetic
  case with true reliability 0.5 the variance estimator gives 0.00 and the
  draw-pair estimator 0.39 (the sampled between-SD's own value). Run it with
  `--trace_dir` on the cluster; without a trace only the biased form is
  available.

- Scripts take a positional `subject` arg and `--bids_folder` kwarg, and are submitted as SLURM arrays from `*/slurm_jobs/`. Each analysis submodule keeps its SLURM wrappers in its own `slurm_jobs/` subfolder.
- Cognitive model traces are written to `<bids>/derivatives/cogmodels/model-<label>_trace.netcdf` as ArviZ NetCDF. The directory name `cogmodels` is historical (kept on disk so existing traces remain loadable).
- The `tms_keys.yml` and `all_subjects.yml` resource files under `tms_risk/data/` are the authoritative subject lists; `get_tms_subjects()` reads `tms_keys.yml`, `get_all_subject_ids()` reads `all_subjects.yml`. **For cluster sweeps of PRF-based analyses (decode / fisher / mc_decode), use the intersection of `tms_keys.yml` with `derivatives/encoding_model2.model-1.smoothed/` on disk — 35 subjects** (excludes sub-22 and sub-49 who lack PRF fits). Hard-coded in `tms_risk/modeling/slurm_jobs/submit_{fisher_information,mc_decode}.sh`.
- **The dataset describes itself** (2026-08-26): `<bids>/sub-XX/sub-XX_sessions.tsv`
  carries a `stimulation` column (`baseline` for ses-1, `ips`/`vertex` for
  ses-2/3), and `<bids>/{README,sessions.json,participants.json,task-task_events.json}`
  document the design, that column, `participants.tsv` and the `_events.tsv`
  columns. `tms_keys.yml` stays authoritative for the stimulation assignment; the
  session tables are a one-way export. All of it is installed by `python -m
  tms_risk.prepare.write_bids_metadata` (`--check` verifies the dataset is still
  in sync, `--dry_run` previews); the four root files are version-controlled
  templates in `tms_risk/data/bids_metadata/`, so edit them there, never in the
  dataset. Read the tables back with `get_sessions_info(bids_folder)` or
  `get_tms_conditions(bids_folder)` — the latter returns exactly the YAML dict
  when given a folder, and the packaged YAML when called with no argument.
  Note `participants.tsv`'s `tms_subject` column is `true` for only 35 subjects —
  it already excludes the two outliers (22, 49) that `tms_keys.yml` still lists,
  so it is not a clean "was this subject stimulated" flag; the session table is.
  The README's "Known quirks" section lists every remaining `bids-validator`
  error (no `duration` in `_events.tsv`, Philips `_physio.log`, a stray
  FreeSurfer tree in `sub-32/`, …) — all pre-existing, none introduced by this.
- The `slurm_jobs/fit_model.sh` wrapper activates `tms_risk_cpu`, uses `--account=zne.uzh`, and invokes the script via `python -m tms_risk.behavior.fit_model`.
- **Canonical PRF / encoding model**: `model_label=1` of `Subject.get_prf_parameters` — the "amplitude varies per session" regression variant from `modeling/fit_regression_nprf.py` (the paper's Fig 2 fits). All decode / fisher / mc_decode scripts use this. Files live at `derivatives/encoding_model2.model-1.smoothed[.cv]/sub-XX/`.
- **Color semantics across ALL paper figures** (codified 2026-08-20): red/green is
  reserved for stimulation (see next bullet); **presentation order never gets a hue** —
  encode it by row/panel position with a text label wherever possible, and when both
  orders must share one panel, use **risky second = near-black (`.15`) filled markers,
  risky first = light gray (`.62`) open markers** (as in `plot_fig3c_localization.py`).
  Blue/orange are reserved for model contrasts (flexible vs Weber in Fig 4D; tuning vs
  magnitude models in the Fig 2 bottom row). NOTE: Fig 3B's dark/light densities encode
  *significance*, not order — do not read or extend them as an order palette.
- **Canonical IPS / Vertex palette**: **IPS (stimulated) = `#d62728` red, Vertex (sham) = `#2ca02c` green** — red marks the active/experimental arm. Use this in *every* IPS-vs-vertex figure. NOTE: `tms_risk.behavior.utils.stimulation_palette = sns.color_palette()[2:4]` is the tuple `(green, red)`; do **not** blindly zip it with alphabetical `['ips','vertex']` (that gives the inverted IPS=green). Map explicitly: `{'ips': stimulation_palette[1], 'vertex': stimulation_palette[0]}` or hardcode the hexes above.
- **"Expected uncertainty" vs "decoded SD"**: when plotting decoding-accuracy curves, prefer `mean_abs_error` (the realised mean |decoded − true| from `model.get_expected_uncertainty`) over `sqrt(var_E)`. The former is the actual simulate-and-decode-back error you'd see on a fresh trial; the latter is just the decoder's self-reported posterior width and tends to underestimate the realised error.
- **`risky_first` means the RISKY option came FIRST.** Defined at `tms_risk/utils/data.py:267` as `p1 == 0.55`; every mapping in the repo is `{True: 'Risky first', False: 'Risky second'}`. Do not re-label it when writing figures or prose.

## Where the paper's numbers come from

**`notes/PROVENANCE.md` is the index**: one row per published figure, table and
reported statistic, with the script that produces it, the TSV it reads, and whether
the draft's number is still current. Update it whenever you add or supersede an
analysis — it is the file a future reader will open first.


One cell per reported statistic. `notes/v8_stats_check.md` holds a
line-by-line audit of the v8 Results against these cells.

| Reported statistic | Notebook · cell |
|---|---|
| nPRF amplitude / preferred numerosity (`mu`) / dispersion (`sd`) / explained variance (`r2`) | `modeling/notebooks/analyze_encoding_model.ipynb` · the `pingouin.pairwise_tests` loop (runs **two-sided**; the paper halves it for the directional claims) |
| Proportion of voxels with cvR² > 0 | same notebook · the `pg.ttest(..., alternative='less')` cell (the only genuinely one-sided test there) |
| Decoding accuracy + decoding × order interaction | `modeling/notebooks/analyze_decoding.ipynb` · the two `rm_anova` cells |
| Indifference point × choice consistency; Δconsistency × Δrisk attitude (overall + split by `risky_first`) | `behavior/notebooks/correlation_preference_noise.ipynb` · cell 3 and the final cell |
| Δ nPRF amplitude × Δ cognitive noise | `behavior/notebooks/neurobehavioral_correlates.ipynb` · cell 4 |
| ELPD table / Flexible PMC curves | `behavior/notebooks/comprehensive_model_comparison.ipynb`, `behavior/notebooks/figure4.ipynb` |

**Five traps when regenerating any of the above:**

1. **Model 1 makes only `amplitude` session-specific.** `mu`, `sd`, `r2`, `cvr2`
   are shared across sessions by construction, so an IPS-vs-vertex test on them
   under `get_prf_parameters(model_label=1, ...)` returns `t = NaN`. The paper's
   Fig-2 parameter tests came from the older `get_prf_parameters_volume(...)`
   (log-space `encoding_model.denoise.smoothed` tree), which commit `ba58cb1`
   removed. Use `model_label=2` if you need genuine per-session `mu`/`sd`.
2. **Outputs of the paper-canonical notebooks were stripped** by commit
   `ead9b9c`. Recover them with
   `git show ead9b9c^:tms_risk/encoding_model/notebooks/<nb>.ipynb`.
3. **`derivatives/encoding_models/prf_parameters_thr.tsv` is a cached
   intermediate**, written by `analyze_encoding_model.ipynb` and read by
   `neurobehavioral_correlates.ipynb`. It was overwritten on 2026-05-22 by the
   refactored notebook, which flipped the brain–behaviour correlation from
   r = −0.38 to r = +0.11. The post-refactor version is preserved alongside it as
   `prf_parameters_thr.model-1_20260522.bak.tsv`; check provenance before trusting
   either.
4. **`get_volume_mask()` needs the session's fMRIPrep EPI brain mask**, which it
   loads before the `ips_masks` cache check — so it raises for sessions 2/3, whose
   fMRIPrep masks are no longer on local disk. On a cache hit it *returns* the
   cached ROI mask and discards `base_mask`, so reading
   `derivatives/ips_masks/sub-XX/func/ses-N/sub-XX_space-T1w_desc-<ROI>_mask.nii.gz`
   directly is exactly equivalent.
5. **`get_sd_curve()` under current bauer does not reproduce the published noise
   curves** — a concrete instance of the version-sensitivity documented above.
   Both versions build a genuine spline basis (`FlexibleNoiseRiskRegressionModel`
   inherits `make_dm` from `FlexibleNoiseComparisonModel`; the 2-column
   `[1, x_norm]` `make_dm` in `risky_choice.py` belongs to `AffineNoiseRiskModel`
   and is irrelevant here). What changed is **knot anchoring**: `ecc6454` built the
   basis per call as `bs(x, degree=3, df=spline_order, include_intercept=True,
   lower_bound=min_n, upper_bound=max_n)` over `min/max` of *both* `n1` and `n2`,
   while HEAD fixes `design_info` at construction time anchored to the paradigm
   column for that variable. Measured on the brain–behaviour correlation: the
   `ecc6454` formula gives **r = −0.3792** (matching the paper's saved −0.379173 to
   five decimals), HEAD's `get_sd_curve` gives **r = −0.3167** on identical inputs.
   To re-evaluate a published trace, rebuild the basis with patsy using the
   `ecc6454` formula and dot it with the five spline coefficients from the trace —
   that sidesteps `build_model()`, which can't be used against the pinned commit
   anyway since `fit_model.py` passes the renamed `spline_order=`.

The **department SMB archive** is the recovery source of record for pruned
derivatives — it held the log-space cvR² maps when the cluster was unreachable:
`/Volumes/g_econ_department$/projects/2022/dehollander_moisa_ruff_ipsriskydecisionmaking/data/ds-tmsrisk/derivatives/`.
