# TMS Risk Project

Neural and behavioral analysis of TMS effects on risk decision-making.

## Quick Start

### Environment Setup

See [environments/README.md](environments/README.md) for detailed conda environment setup instructions.

**Local development (Mac Apple Silicon):**
```bash
conda env create -f environments/environment_apple_silicon.yml
conda activate tms_risk
```

**Local development (Intel Mac/Linux CPU):**
```bash
conda env create -f environments/environment.yml
conda activate tms_risk
```

**On compute cluster:**
```bash
cd ~/git/tms_risk/environments
sbatch build_env.sh cpu
```

### Running Analysis

See individual module READMEs:
- `experiment/` - Task implementation
- `tms_risk/cogmodels/` - Cognitive modeling
- `tms_risk/encoding_model/` - Neural encoding models
- `tms_risk/encoding_model/cluster_scripts/` - SLURM batch scripts

## Project Structure

```
tms_risk/
├── environments/          # Conda environment specifications
├── experiment/           # Experimental task code
├── tms_risk/            # Main analysis package
│   ├── cogmodels/       # Cognitive model fitting
│   ├── encoding_model/  # Neural encoding models
│   ├── utils/           # Data utilities
│   └── ...
└── libs/                # External dependencies
    ├── braincoder/      # Neural encoding framework
    └── bauer/           # Utility library
```

## Citation

[Add citation information when available]
