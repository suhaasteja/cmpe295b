#!/bin/bash
#SBATCH -J vlm_train10pq
#SBATCH -p gpuqm
#SBATCH -w cs001                  # A100 node (change if needed)
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o vlm_train10pq.%j.out
#SBATCH -e vlm_train10pq.%j.err

set -euo pipefail

module load python3
source ~/cmpe295b/venv_hpc3/bin/activate

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=~/cmpe295b/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OUT_PQ="vlm_train10_${SLURM_JOB_ID}.parquet"

python - <<'PY'
import os, glob, sys
from io import BytesIO
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

workdir   = os.getcwd()
hf_root   = os.path.join(workdir, "hf_cache")
ds_root   = os.path.join(hf_root, "datasets", "immanuelpeter__carla-autopilot-multimodal-dataset")
p_dir     = os.path.join(ds_root, "data")
model_dir = os.path.expanduser("~/cmpe295b/hf_cache/transformers/Qwen__Qwen2.5-VL-7B-Instruct")
prompt_fp = os.path.join(workdir, "prompt.txt")
out_path  = os.path.join(workdir, os.environ.get("OUT_PQ", "vlm_train10.parquet"))

# Prompt
if os.path.exists(prompt_fp):
    with open(prompt_fp, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
else:
    prompt = "Briefly describe the scene and the next safe driving action."
print(f"[info] prompt: {prompt[:120]}{'...' if len(prompt)>120 else ''}")

# Model & processor (offline)
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_dir,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
print("[info] torch:", torch.__version__)
print("[info] cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[info] gpu:", torch.cuda.get_device_name(0))

# Helper: strip heavy image bytes; keep paths
def strip_image_bytes(row_dict):
    def _strip(v):
        # image struct: {'bytes': <binary or None>, 'path': <str or None>}
        if isinstance(v, dict) and "bytes" in v and "path" in v:
            return {"path": v.get("path")}
        return v
    return {k: _strip(v) for k, v in row_dict.items()}

# Iterate train shards and collect up to N rows
train_files = sorted(glob.glob(os.path.join(p_dir, "train-*.parquet")))
if not train_files:
    raise SystemExit("No train-*.parquet shards found.")

target = 10
rows_for_output = []

# Import here to avoid module import if not needed
import pyarrow as pa
from pyarrow import parquet as pq
from PIL import Image

picked = 0
for shard in train_files:
    if picked >= target:
        break
    print(f"[info] scanning {os.path.basename(shard)}")
    pf = pq.ParquetFile(shard)

    # Read 1-row batches to minimize memory
    for batch in pf.iter_batches(batch_size=1):
        if picked >= target:
            break

        bdict = batch.to_pydict()
        # Build a scalar row (each value is a singleton list)
        row = {k: (v[0] if isinstance(v, list) else v) for k, v in bdict.items()}

        # Decode one image from image_front bytes if available
        img = None
        img_struct = row.get("image_front")
        if isinstance(img_struct, dict) and img_struct.get("bytes"):
            try:
                img = Image.open(BytesIO(img_struct["bytes"])).convert("RGB")
            except Exception as e:
                print(f"[warn] image decode failed: {e}")
                continue
        elif isinstance(img_struct, dict) and img_struct.get("path"):
            # fallback: not expected here, but try resolving a file if any
            fname = img_struct["path"]
            hits = glob.glob(os.path.join(ds_root, "**", fname), recursive=True)
            if hits:
                try:
                    img = Image.open(hits[0]).convert("RGB")
                except Exception as e:
                    print(f"[warn] open via path failed: {e}")
                    continue
        else:
            # No usable image → skip
            continue

        # Qwen chat formatting
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]},
        ]
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[chat_text], images=[img], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=64)
        vlm_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        # Build output row: keep all metadata, drop heavy bytes, add vlm_output
        clean_row = strip_image_bytes(row)
        clean_row["vlm_output"] = vlm_text

        rows_for_output.append(clean_row)
        picked += 1
        print(f"[ok] {picked}/{target} -> {clean_row.get('image_front', {}).get('path')}")

# Write to Parquet
if rows_for_output:
    table = pa.Table.from_pylist(rows_for_output)
    pq.write_table(table, out_path)
    print(f"[done] wrote {len(rows_for_output)} rows to {out_path}")
else:
    print("[warn] no rows written")
PY
