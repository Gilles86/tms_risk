# TMS Risk

Combined cTBS-TMS + 7T fMRI study of how parietal magnitude
representations causally shape risk preferences (de Hollander, Moisa
& Ruff). Manuscript draft: `notes/paper/TMS paper -v7.pdf`.

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

## Citation

Manuscript in preparation. See `notes/paper/TMS paper -v7.pdf`.
