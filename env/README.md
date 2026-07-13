# Residual Environment Snapshot

These files capture the source machine's `residual` Python environment for reproducing the DexMG chunk-residual experiments on another machine.

Files:

- `residual_pip_freeze.txt`: pip requirements exported from `/mnt/mnt/data/envs/residual`, with the PyTorch CUDA 12.8 wheel index added.
- `residual_env.yml`: conda environment export with the local `prefix:` removed.

Notes:

- The private `openpi-client` editable dependency from `kai0_new4090` is omitted. It is only needed for LIBERO/pi0 experiments, not the DexMG five-task runs in `outputs_chunk`.
- `robosuite` is listed as `robosuite==1.5.1`; for exact source-machine behavior, install the project `deps/robosuite` source tree as described in `docs/reproduce-on-new-machine.md`.

Typical use:

```bash
conda create -n residual python=3.10 -y
conda activate residual
pip install -r env/residual_pip_freeze.txt
```

Alternative:

```bash
conda env create -f env/residual_env.yml
conda activate residual
```
