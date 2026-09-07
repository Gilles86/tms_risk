# TMS Risk

Combined cTBS-TMS + 7T fMRI study of how parietal magnitude
representations causally shape risk preferences (de Hollander, Moisa
& Ruff). Manuscript draft: `notes/paper/TMS_paper_v8_with_CR_comments.pdf`.

> **Looking for the code behind a specific figure, table or statistic?**
> [`notes/PROVENANCE.md`](notes/PROVENANCE.md) has one row per published item:
> which script produces it, what it reads, what it writes, and whether the
> number in the current draft is still current.

The pipeline targets numerosity-tuned right parietal cortex with cTBS
(vertex control vs. parietal) and measures effects on (a) nPRF
responses, (b) trial-by-trial decoding accuracy, (c) psychophysical
choice consistency / risk-neutral probability, and (d) parameters of
the Perceptual-and-Memory-based Choice (PMC) model and its **Flexible
PMC** extension.

## Quick start

### Local development (Apple Silicon)

```bash
conda env create -f environment_apple_silicon.yml
conda activate tms_risk
```

### Cluster (SLURM)

```bash
sbatch create_env/create_cpu_env.sh   # tms_risk_cpu
sbatch create_env/create_gpu_env.sh   # tms_risk_cuda — must be on GPU node
```

See [`create_env/README.md`](create_env/README.md) for env details.

## Repo layout

```
tms_risk/
├── CLAUDE.md                       # developer-facing recipes & gotchas
├── README.md                       # this file
├── environment_apple_silicon.yml   # local Mac dev env
├── create_env/                     # cluster env builds (sbatch wrappers)
├── experiment/                     # PsychoPy task code
├── libs/                           # git submodules (braincoder, bauer)
├── notes/                          # paper + working notes
└── tms_risk/                       # main analysis package
    ├── utils/data.py               # Subject class — single source of truth
    ├── prepare/                    # raw → BIDS conversions
    ├── glm/                        # GLMsingle single-trial betas
    ├── modeling/                   # nPRF + decoding (braincoder, TensorFlow)
    ├── behavior/                   # PMC / Flexible PMC / probit (bauer, PyMC)
    ├── tms_targeting/              # individualized cTBS site selection
    ├── visualize/                  # plotting helpers
    └── ...
```

Each analysis submodule has its own `slurm_jobs/` subfolder with the
SLURM wrappers for the python scripts next to it.

## Reproducing the paper

Step-by-step guide in [`REPRODUCING.md`](REPRODUCING.md): pipeline
stages, model labels for Table 1, and a per-figure notebook map.

### Which notebook produced which reported number

Every statistic in the Results maps to one notebook cell:

| Reported in the paper | Notebook |
|---|---|
| nPRF amplitude / preferred numerosity / dispersion / explained variance / proportion cvR² > 0 (Fig. 2A–B) | `tms_risk/modeling/notebooks/analyze_encoding_model.ipynb` |
| Decoding accuracy + decoding × presentation-order ANOVA (Fig. 2C) | `tms_risk/modeling/notebooks/analyze_decoding.ipynb` |
| Psychometric curves, slope (consistency) and RNP (Fig. 3) | `tms_risk/notebooks/figure2.ipynb` (name is historical) |
| Indifference point × consistency; ΔConsistency × Δrisk attitude, overall and split by trial order | `tms_risk/behavior/notebooks/correlation_preference_noise.ipynb` |
| Flexible PMC noise curves (Fig. 4) | `tms_risk/behavior/notebooks/figure4.ipynb` |
| ELPD model comparison (Table 1) | `tms_risk/behavior/notebooks/comprehensive_model_comparison.ipynb` |
| Δamplitude × Δcognitive-noise brain–behaviour link | `tms_risk/behavior/notebooks/neurobehavioral_correlates.ipynb` |

A line-by-line audit of the v8 Results against these cells — including
three discrepancies and the reproducibility caveats introduced by the
`cleanup/ddm-port` refactor — is in
[`notes/v8_stats_check.md`](notes/v8_stats_check.md).

## Citation

Manuscript in preparation. See `notes/paper/TMS paper -v7.pdf`.
