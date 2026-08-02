# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Analysis code for the combined cTBS-TMS + 7T fMRI study **"Risk Attitudes Causally Rely on Parietal Magnitude Representations"** (de Hollander, Moisa & Ruff). The paper draft is at `notes/paper/TMS paper -v7.pdf`. The pipeline targets numerosity-tuned right parietal cortex with cTBS (vertex control vs. parietal) and measures effects on (a) nPRF responses, (b) trial-by-trial decoding accuracy, (c) psychophysical choice consistency / risk-neutral probability, and (d) parameters of the Perceptual-and-Memory-based Choice (PMC) model and its **Flexible PMC** extension (B-spline noise function over magnitude).

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

### The T4 GPU node

A second VM at `ubuntu@172.23.206.84` (Tesla T4, 8 cores, 31 GB) runs the same fits
through the **numpyro/JAX** backend (`fit_pmc_noisefix … --backend numpyro`). Same
layout: `/data/git/tms_risk`, `/data/ds-tmsrisk`, `/data/logs`, conda env
`tms_risk_gpu` under `/data/miniforge3`. One GPU means one model at a time, but a
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

- Scripts take a positional `subject` arg and `--bids_folder` kwarg, and are submitted as SLURM arrays from `*/slurm_jobs/`. Each analysis submodule keeps its SLURM wrappers in its own `slurm_jobs/` subfolder.
- Cognitive model traces are written to `<bids>/derivatives/cogmodels/model-<label>_trace.netcdf` as ArviZ NetCDF. The directory name `cogmodels` is historical (kept on disk so existing traces remain loadable).
- The `tms_keys.yml` and `all_subjects.yml` resource files under `tms_risk/data/` are the authoritative subject lists; `get_tms_subjects()` reads `tms_keys.yml`, `get_all_subject_ids()` reads `all_subjects.yml`. **For cluster sweeps of PRF-based analyses (decode / fisher / mc_decode), use the intersection of `tms_keys.yml` with `derivatives/encoding_model2.model-1.smoothed/` on disk — 35 subjects** (excludes sub-22 and sub-49 who lack PRF fits). Hard-coded in `tms_risk/modeling/slurm_jobs/submit_{fisher_information,mc_decode}.sh`.
- The `slurm_jobs/fit_model.sh` wrapper activates `tms_risk_cpu`, uses `--account=zne.uzh`, and invokes the script via `python -m tms_risk.behavior.fit_model`.
- **Canonical PRF / encoding model**: `model_label=1` of `Subject.get_prf_parameters` — the "amplitude varies per session" regression variant from `modeling/fit_regression_nprf.py` (the paper's Fig 2 fits). All decode / fisher / mc_decode scripts use this. Files live at `derivatives/encoding_model2.model-1.smoothed[.cv]/sub-XX/`.
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
