#!/bin/bash
#SBATCH -J vlm_sanity
#SBATCH -p gpuqm
#SBATCH -w cs001                 # A100 node; remove or change if needed
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o sanity_check.%j.out
#SBATCH -e sanity_check.%j.err

set -euo pipefail

# env
module load python3
source ~/cmpe295b/venv_hpc3/bin/activate

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=~/cmpe295b/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

python -u - <<'PY'
import os, glob
from io import BytesIO
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# paths
cwd       = os.getcwd()
hf_root   = os.path.join(cwd, "hf_cache")
ds_root   = os.path.join(hf_root, "datasets", "immanuelpeter__carla-autopilot-multimodal-dataset")
parq_dir  = os.path.join(ds_root, "data")
model_dir = os.path.expanduser("~/cmpe295b/hf_cache/transformers/Qwen__Qwen2.5-VL-7B-Instruct")
prompt_fp = os.path.join(cwd, "prompt.txt")
out_path  = os.path.join(cwd, f"sanity_one_{os.environ.get('SLURM_JOB_ID','local')}.parquet")

# prompt
prompt = "Briefly describe the scene and the next safe driving action."
if os.path.exists(prompt_fp):
    with open(prompt_fp, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
print("[info] prompt:", prompt[:120].replace("\n"," ") + ("..." if len(prompt)>120 else ""))

# model (offline)
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_dir,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
print("Model on:", model.device)

# helper to strip heavy bytes before saving
def strip_image_bytes(row):
    if isinstance(row, dict) and "bytes" in row and "path" in row:
        return {"path": row.get("path")}
    return row

# pick first test shard
test_files = sorted(glob.glob(os.path.join(parq_dir, "test-*.parquet")))
if not test_files:
    raise SystemExit("No test-*.parquet shards found under data/")

pfile = test_files[0]
print("Reading:", os.path.basename(pfile))

pf = pq.ParquetFile(pfile)
selected = None
for batch in pf.iter_batches(columns=["image_front"], batch_size=1):
    row = batch.to_pydict()["image_front"][0]   # {'bytes':..., 'path':...}
    img = None
    if isinstance(row, dict) and row.get("bytes"):
        try:
            img = Image.open(BytesIO(row["bytes"])).convert("RGB")
        except Exception:
            img = None
    if img is None and isinstance(row, dict) and row.get("path"):
        # fall back to path search (handles basename-only cases)
        hits = glob.glob(os.path.join(ds_root, "**", row["path"]), recursive=True)
        if hits:
            try:
                img = Image.open(hits[0]).convert("RGB")
            except Exception:
                img = None
    if img is not None:
        selected = {"image_front": row}
        break

if selected is None:
    raise SystemExit("Could not decode any image from first test shard.")

# build chat + run once
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=[chat_text], images=[img], return_tensors="pt").to(model.device)

with torch.inference_mode():
    out_ids = model.generate(**inputs, max_new_tokens=64)
text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

# assemble row (no raw bytes) and write parquet
row_out = {"image_front": strip_image_bytes(selected["image_front"]), "vlm_output": text}
table = pa.Table.from_pylist([row_out])
pq.write_table(table, out_path)
print(f"[done] wrote 1 row to {out_path}")
PY
