**HPC3 VLM workflow cheat sheet**
# quick glossary

* **Slurm**: the cluster job scheduler. You talk to it with `sinfo`, `squeue`, `sbatch`, `srun`, etc.
* **Partition**: a Slurm queue (e.g., `gpuqs`, `gpuqm`, `gpuql`). Each has different time limits.
* **Node**: a physical machine in a partition (e.g., `cs001`, `g7`).
* **TRES**: *Trackable RESources* (cpu, mem, gpu, etc.) that Slurm accounts/limits.
* **GRES**: *Generic RESources* (GPUs, etc.) when configured. On HPC3, GRES appears unset; we target nodes directly instead.

---

# 1) what hardware is available?

**Partitions and node states**

```bash
sinfo                                 # overview
sinfo -p gpuqm -o "%20N %10t %10G"    # nodes in gpuqm; GRES shows as (null) here
sinfo -N -p gpuqm -o "%N %t" | awk '$2=="idle"{print $1}'   # idle nodes in gpuqm
```

**Per-node resource info**

```bash
scontrol show node cs001 | egrep -i "CfgTRES|AllocTRES|Gres|Feature"
# CfgTRES shows CPU count and RAM per node
```

**Check GPU model on a node**

```bash
srun -p gpuqm -w cs001 -c 1 --mem=1G -t 00:02:00 --pty bash -lc 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

**Your current GPU inside an allocation**

```bash
nvidia-smi
```

---

# 2) how do i pick a specific gpu node?

Since `GRES` isn’t exposed, pick an **idle GPU node** and target it with `-w`:

```bash
# list idle nodes
sinfo -N -p gpuqm -o "%N %t" | awk '$2=="idle"{print $1}'

# interactive shell on a specific node
srun -p gpuqm -w cs001 -c 4 --mem=16G -t 00:30:00 --pty /bin/bash
```

> `cs001–cs003` have **A100 40GB** (great for 7B VLMs).
> `g3–g…` are typically **P100 12GB**.

---

# 3) what params/config should i set?

Common flags (work for both `srun` and `sbatch`):

* `-p <partition>`: queue (`gpuqs/gpuqm/gpuql`)
* `-w <node>`: pick a node (e.g., `cs001`)
* `-c <N>`: CPU cores
* `--mem=<XG>`: RAM
* `-t DD-HH:MM:SS`: walltime
* `--pty`: allocate a pseudo-tty (interactive run)

Env vars for your run:

```bash
module load python3
source ~/cmpe295b/venv_hpc3/bin/activate

export CUDA_VISIBLE_DEVICES=0                # pick GPU index inside your node allocation
export HF_HOME=~/cmpe295b/hf_cache
export HF_HUB_OFFLINE=1                      # enforce offline HF
```

---

# 4) run code on a GPU (interactive)

```bash
srun -p gpuqm -w cs001 -c 8 --mem=32G -t 01:00:00 --pty /bin/bash
module load python3
source ~/cmpe295b/venv_hpc3/bin/activate
export CUDA_VISIBLE_DEVICES=0 HF_HOME=~/cmpe295b/hf_cache HF_HUB_OFFLINE=1

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

> In your Transformers code, you already use `device_map="auto"` and `torch_dtype=torch.float16`; the model loads onto the visible GPU.

---

# 5) run code via batch (`sbatch`) and check results

**Submit**

```bash
sbatch my_job.sbatch
```

**Monitor**

```bash
squeue -u $USER          # running/pending jobs
sacct -j <JOBID>         # finished job history
```

**Logs & outputs**

* `#SBATCH -o my_job.%j.out` → stdout log
* `#SBATCH -e my_job.%j.err` → stderr log

**Example minimal sbatch**

```bash
#!/bin/bash
#SBATCH -J demo
#SBATCH -p gpuqm
#SBATCH -w cs001
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o demo.%j.out
#SBATCH -e demo.%j.err

set -euo pipefail
module load python3
source ~/cmpe295b/venv_hpc3/bin/activate
export CUDA_VISIBLE_DEVICES=0 HF_HOME=~/cmpe295b/hf_cache HF_HUB_OFFLINE=1

python -u your_script.py
```

---

# 6) `srun` vs `sbatch` vs `--pty`

* **`srun ... --pty /bin/bash`**: *interactive* allocation; you see output live.
* **`sbatch script.sbatch`**: *queued batch*; runs unattended; output goes to `.out/.err`.
* **`--pty`**: allocates a pseudo-terminal (so programs behave like you’re in a real shell).

---

# 7) copying results back to your Mac (from your Mac terminal)

```bash
# login gateway path
scp 016197935@coe-hpc1.sjsu.edu:~/cmpe295b/sanity_one_<JOBID>.parquet ~/Downloads/
# or any other artifact:
scp 016197935@coe-hpc1.sjsu.edu:~/cmpe295b/vlm_train10_13845.parquet ~/Downloads/
```

---

# 8) CUDA, partitions, GRES, TRES in plain terms

* **CUDA**: NVIDIA’s GPU compute runtime. You’re using it through **PyTorch** (`torch.cuda.is_available()` → True) and Transformers.
* **Partition**: job queues with limits/time policies. On HPC3 you used `gpuqm` (7-day limit).
* **CUDA_VISIBLE_DEVICES=0**: inside your allocation, expose only GPU 0 to the process; makes multi-GPU nodes predictable.
* **TRES** (cpu, mem, gpu, billing): how Slurm tracks/limits resources.
* **GRES** (generic resources): commonly used for GPUs (e.g., `--gres=gpu:1`). On HPC3, GRES isn’t exposed; you successfully targeted nodes directly with `-w cs001`.

> No, you don’t “partition the GPU” yourself. Slurm grants you a node (or slice) and you choose which GPU index to use via `CUDA_VISIBLE_DEVICES`.

---

# 9) where are the dataset and model on disk?

* **Dataset snapshot (CARLA)**
  `~/cmpe295b/hf_cache/datasets/immanuelpeter__carla-autopilot-multimodal-dataset`
  Parquet shards under: `.../data/*.parquet`
  (You built smaller samples too, e.g., `carla_sample_200.parquet`.)

* **Model snapshot (Qwen2.5-VL-7B-Instruct)**
  `~/cmpe295b/hf_cache/transformers/Qwen__Qwen2.5-VL-7B-Instruct`

* **Your working dir & job files**
  `~/cmpe295b/`
  Contains: `prompt.txt`, `sanity_check.sbatch`, `vlm_train10_parquet.sbatch`, outputs like `vlm_train10_13845.parquet`, and logs `*.out/*.err`.

---

# 10) dependencies you used

* **PyTorch** (GPU compute) — 2.6.0+cu124 on cluster
* **Transformers** (model + processor)
* **huggingface_hub** (offline snapshot management)
* **datasets** (HF datasets library, though we mostly used Arrow/Parquet directly)
* **Pillow (PIL)** (image decoding)
* **PyArrow** (columnar I/O; reading Parquet; writing your result Parquet)

> **What is PyArrow?**
> A high-performance columnar data engine and format; we used it to:
>
> * read Parquet shards (`pq.ParquetFile(...)`)
> * build small result tables and write `.parquet` outputs quickly.

---

# 11) verifying that the model ran on GPU

* You printed:

  * `torch.cuda.is_available()` → `True`
  * `torch.cuda.get_device_name(0)` → `NVIDIA A100-PCIE-40GB`
  * Model prints *“Model on: cuda:0”*
* You requested a GPU node via `-p gpuqm -w cs001` and limited to GPU 0 via `CUDA_VISIBLE_DEVICES=0`.

---

# 12) small ready-to-run snippets

**instant sanity check with srun**

```bash
srun -p gpuqm -w cs001 -c 4 --mem=16G -t 00:10:00 --pty /bin/bash
module load python3
source ~/cmpe295b/venv_hpc3/bin/activate
export CUDA_VISIBLE_DEVICES=0 HF_HOME=~/cmpe295b/hf_cache HF_HUB_OFFLINE=1
python - <<'PY'
import torch; print("GPU:", torch.cuda.get_device_name(0))
PY
```

**tail job logs**

```bash
squeue -u $USER
tail -n +1 sanity_check.*.{out,err}
```

**measure space + sizes**

```bash
df -h $HOME
du -sh ~/cmpe295b/hf_cache/datasets/immanuelpeter__carla-autopilot-multimodal-dataset
du -sh ~/cmpe295b/hf_cache/transformers/Qwen__Qwen2.5-VL-7B-Instruct
```

---

# 13) “why did earlier one-image job show no PNG?”

Because the HF parquet stores images as `{bytes, path}` structs and sometimes only **basename** is present or bytes were absent. We fixed it by:

* decoding from `bytes` when present; otherwise
* searching the dataset tree by filename (`glob("**/<name>.png", recursive=True)`).

---

# 14) quick pointer: picking models for a 12GB P100 vs 40GB A100

* **P100 12GB**: prefer smaller or **AWQ/INT4** quantized models.
* **A100 40GB**: 7B VLMs (like Qwen2.5-VL-7B-Instruct) in **fp16** are fine.

(AWQ = Activation-aware Weight Quantization; reduces VRAM and speeds up inference with minimal quality loss.)

---

