# Conda Environments

This directory contains conda environment specifications for the TMS risk project.

## Available Environments

- **environment.yml** - Standard CPU environment (Linux/Mac x86_64)
  - Uses Intel MKL for optimized CPU computation
  - TensorFlow 2.18 CPU version
  - Environment name: `tms_risk`

- **environment_apple_silicon.yml** - Apple Silicon (M1/M2/M3) optimized
  - Uses Apple Metal for GPU acceleration
  - TensorFlow 2.18 with Metal plugin
  - Environment name: `tms_risk`

- **environment_gpu.yml** - NVIDIA GPU environment
  - CUDA-enabled TensorFlow
  - Environment name: `tms_risk_gpu`

- **environment_cuda.yml** - Alternative CUDA configuration
  - Environment name: `tms_risk_cuda`

## Local Setup

### Mac (Apple Silicon)
```bash
conda env create -f environments/environment_apple_silicon.yml
conda activate tms_risk
```

### Mac (Intel) / Linux (CPU only)
```bash
conda env create -f environments/environment.yml
conda activate tms_risk
```

### Linux with NVIDIA GPU
```bash
conda env create -f environments/environment_gpu.yml
conda activate tms_risk_gpu
```

## Cluster Setup (SLURM)

On the compute cluster, use the provided SLURM script to build environments:

### Build CPU environment (default, recommended for most jobs)
```bash
cd ~/git/tms_risk/environments
sbatch build_env.sh cpu
```

### Build GPU environment (requires GPU node!)
```bash
sbatch --gres=gpu:1 build_env.sh gpu
```

### Build CUDA environment (requires GPU node!)
```bash
sbatch --gres=gpu:1 build_env.sh cuda
```

**Important:** GPU and CUDA environments MUST be built on GPU nodes. The build script will fail if you try to build them on CPU-only nodes.

### Monitor build progress
```bash
tail -f ~/logs/build_env_*.txt
```

The build script will:
1. Remove any existing environment with the same name
2. Create a fresh environment from the yml file
3. Verify Python and TensorFlow versions
4. Test device availability

Build typically takes 10-30 minutes depending on cluster load.

## Environment Names

| YML File | Conda Environment Name | Use Case |
|----------|----------------------|----------|
| environment.yml | `tms_risk` | Standard CPU jobs, local development |
| environment_apple_silicon.yml | `tms_risk` | Mac M1/M2/M3 development |
| environment_gpu.yml | `tms_risk_gpu` | NVIDIA GPU compute |
| environment_cuda.yml | `tms_risk_cuda` | Alternative CUDA setup |

## Updating Environments

After modifying any yml file, rebuild the environment:

**Locally:**
```bash
conda env update -f environments/environment.yml --prune
```

**On cluster:**
```bash
sbatch environments/build_env.sh cpu  # or gpu/cuda
```

## Troubleshooting

### "Environment already exists"
The build script automatically removes existing environments. If you encounter issues:
```bash
conda env remove -n tms_risk
```

### Conda activation fails
Make sure `init_conda.sh` exists in your home directory on the cluster.

### Package conflicts
If you get dependency conflicts, try:
1. Update conda: `conda update -n base conda`
2. Clear cache: `conda clean --all`
3. Rebuild from scratch using the build script
