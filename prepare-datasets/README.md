# prepare-datasets

Data preparation pipeline for **Perception-Judge** - a multimodal judge trained to mitigate perceptual judgment bias via perceptual perturbation and reward modeling (GRPO).

> **Paper:** *Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling (ICML 2026)*
> **Project page:** https://perception-judge.github.io/

---

## Overview

This pipeline generates **PPJD (Perceptually Perturbed Judgment Dataset)** — the core training dataset for Perception-Judge.

Starting from MMPR v1.2 (which provides `chosen` / `rejected` response pairs), we generate two perturbed rejected variants for each correct response:

- `r_c` - the verified correct (chosen) response
- `r_{rp}` - a perceptually perturbed response: visually incorrect perception, but reasoning logic preserved
- `r_{rp+r}` - a perceptually+reasoning perturbed response: both visual perception and reasoning corrupted

The desired judge ordering is: **`r_c ≻ r_{rp} ≻ r_{rp+r}`**

Perception-Judge is trained with GRPO using a batch ranking reward that enforces this graded ordering.

---

## File Structure

```
prepare-datasets/
├── client_async.py                           # Unified async VLM client (OpenAI / vLLM)
├── schema.py                                 # JSONL field auto-mapping
├── utils.py                                  # Logger, iter_jsonl, normalize_answer
├── processing_async.py                       # Core 3-step perturbation pipeline
├── prompts/
│   └── prompts.py                            # Prompts for verify / extract / rewrite
│
├── generate_synthetic_datasets_async.py      # [Entry] annotation files → PPJD quadruplets
├── generate_rejected_from_score.py           # [Entry] score.jsonl → PPJD (from scored data)
│
├── post_processing_filter_after_gen.py       # Remove PERCEPTION_ATTRS artifact
├── post_processing_merge_folders.py          # Merge two output folders
│
├── vlm_in_out_multi_gpu.py                  # Offline paraphrase via vLLM (tensor parallel)
├── vlm_in_out_merge.py                      # Merge multi-model paraphrase results
│
├── run_example.sh
└── requirements.txt
```

---

## Requirements

```bash
pip install httpx Pillow python-dotenv tqdm aiohttp

# Optional - only for vlm_in_out_multi_gpu.py
pip install vllm transformers
```

Set `OPENAI_API_KEY` in a `.env` file when using OpenAI models (GPT-4o / GPT-5).

---

## Step 0 - Dataset Setup (MMPR v1.2)

The pipeline uses **MMPR v1.2** (Wang et al., 2024) as the base dataset.

### Download annotations

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download MMPR v1.2 annotation JSONL files
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="OpenGVLab/MMPR-v1.2",
    repo_type="dataset",
    local_dir="data/MMPR-v1_2",
)
EOF
```

After downloading, the annotation files should be at:
```
data/MMPR-v1_2/annotations/annotations/*.jsonl
```

The pipeline uses files ending in `_correctness_rules.jsonl` (42 files, ~322k samples total).

### Download images

MMPR v1.2 spans multiple source datasets. Download each one and place images under a shared root:

```
data/MMPR-v1_2/images/images/    ← use this path as --images_root
├── ai2d/
├── chartqa/
├── coco/          # shared by okvqa, vqav2, vsr
├── dvqa/
├── GeomVerse/
├── gqa/
├── iconqa/
├── InfoVQA/
├── koniq/
├── M3CoT/
├── MapQA/
├── MAVIS-Function/
├── MAVIS-Geometry/
├── MathV360K/
├── RLAIF-V/
├── ScienceQA/
├── SROIE/
├── Super-CLEVR/
├── TabMWP/
├── TextVQA/
├── vg/            # used by tallyqa, vsr, _data_gen
├── wildvision/
└── ...
```

The filename prefix of each annotation JSONL (e.g., `tallyqa_`, `chartqa_`) determines which image subdirectory is used. See `processing_async.py:DATASET_PREFIX_MAP` for the full mapping.

> **Note:** Pass the path that directly contains the dataset subdirectories (`ai2d/`, `chartqa/`, etc.) as `--images_root`. With HuggingFace download, this is typically `data/MMPR-v1_2/images/images`.

---

## PPJD Generation (for GRPO training)

For each annotation sample, the pipeline runs three sequential VLM calls to generate PPJD quadruplets:

```
Step 0. Filter: skip if chosen has <think> tokens or length > 1600 chars
Step 1. Verify:  confirm r_c answers the question correctly (discard if wrong)
Step 2. Extract: identify visually grounded perception attributes from the image
Step 3. Perturb: rewrite r_c with wrong perception → 2 rejected variants (r_{rp}, r_{rp+r})
Step 4. Validate: confirm rejected answers differ from ground truth
```

Output is streamed and appended to JSONL files in real time. The PPJD data is later used to train Perception-Judge with GRPO (batch ranking reward: `r_c ≻ r_{rp} ≻ r_{rp+r}`).

### Generate PPJD

**GPT-4o / GPT-5** (requires `OPENAI_API_KEY` in `.env`):
```bash
python prepare-datasets/generate_synthetic_datasets_async.py \
    --annotations_dir data/MMPR-v1_2/annotations/annotations \
    --images_root     data/MMPR-v1_2/images/images \
    --out_dir         data/MMPR-v1_2/synthetic \
    --model           gpt-5 \
    --max_rows        200 \
    --batch_size      5 \
    --max_concurrent  10
```

**Local vLLM server:**
```bash
# Start server first
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --tensor-parallel-size 4 --port 8000

python prepare-datasets/generate_synthetic_datasets_async.py \
    --annotations_dir data/MMPR-v1_2/annotations/annotations \
    --images_root     data/MMPR-v1_2/images/images \
    --out_dir         data/MMPR-v1_2/synthetic \
    --model           Qwen/Qwen3-VL-30B-A3B-Instruct \
    --base_url        http://localhost:8000 \
    --max_rows        200
```

| Argument | Default | Description |
|---|---|---|
| `--annotations_dir` | required | Directory with MMPR annotation `*.jsonl` files |
| `--images_root` | required | Root directory of image files |
| `--out_dir` | required | Output root (timestamped sub-folder created automatically) |
| `--model` | `gpt-4o` | Model name. For non-OpenAI models, also set `--base_url` |
| `--base_url` | - | vLLM server URL (required for non-OpenAI models) |
| `--files` | all `*.jsonl` | Specific filenames to process |
| `--max_rows` | `50` | Max samples per file |
| `--batch_size` | `5` | Samples per concurrent batch |
| `--max_concurrent` | `10` | Max parallel API calls |
| `--skip_rows` | `0` | Skip first N rows of each file |

### Post-processing

**Remove PERCEPTION_ATTRS artifact** (occasionally appears in generated text):
```bash
python prepare-datasets/post_processing_filter_after_gen.py \
    --input_dir  data/MMPR-v1_2/synthetic/synthetic-gpt-5-YYYY-MMDD-HHMM-datasets \
    --output_dir data/MMPR-v1_2/synthetic/synthetic-gpt-5-YYYY-MMDD-HHMM-datasets_filtered
```

**Merge two generation runs:**
```bash
python prepare-datasets/post_processing_merge_folders.py \
    --folder1    data/MMPR-v1_2/synthetic/run_A \
    --folder2    data/MMPR-v1_2/synthetic/run_B \
    --output_dir data/MMPR-v1_2/synthetic/merged
```

### Offline paraphrase (optional)

Diversify responses by paraphrasing with a locally-loaded model (requires `vllm`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python prepare-datasets/vlm_in_out_multi_gpu.py \
    --input_dir            data/MMPR-v1_2/synthetic/..._filtered \
    --model                Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tensor-parallel-size 4
```

Merge variants from multiple models:
```bash
python prepare-datasets/vlm_in_out_merge.py \
    --base_dir    data/MMPR-v1_2/synthetic/..._filtered \
    --model_dirs  '{"qwen30b": "data/MMPR-v1_2/synthetic/..._filtered_qwen3_vl_30b"}' \
    --output_dir  data/MMPR-v1_2/synthetic/..._merged
```

### Alternative: PPJD from scored data

Use existing human-scored responses (score=5 as `r_c`) instead of MMPR annotation files:

```bash
python prepare-datasets/generate_rejected_from_score.py \
    --score_jsonl /path/to/score.jsonl \
    --images_root data/MMPR-v1_2/images/images \
    --out_dir     data/MMPR-v1_2/synthetic \
    --model       gpt-5
```

### PPJD output format

Each JSONL row is the original MMPR record with a `__normalized__` field appended:

```json
{
  "__normalized__": {
    "image_path": "/abs/path/to/image.jpg",
    "question": "...",
    "answer": "GT answer",
    "chosen": "r_c: correct visual reasoning → correct answer",
    "rejected_perception_1": "r_{rp}: wrong visual perception, reasoning preserved → wrong answer",
    "rejected_perception_answer_1": "wrong answer 1",
    "rejected_perception_2": "r_{rp+r}: wrong perception + degraded reasoning → wrong answer",
    "rejected_perception_answer_2": "wrong answer 2"
  },
  "...original MMPR annotation fields..."
}
```

---

## End-to-End Workflow

```
[MMPR v1.2 annotations]  ←  huggingface: wan-research/MMPR
         │
         ▼
generate_synthetic_datasets_async.py  (GPT-5 / vLLM)
         │
         ▼
post_processing_filter_after_gen.py
         │
         │  (optional)
         ▼
vlm_in_out_multi_gpu.py + vlm_in_out_merge.py
(paraphrase diversity with local VLM)
         │
         ▼
[PPJD quadruplets]  (r_c, r_{rp}, r_{rp+r})
         │
         ▼
GRPO training (verl framework)
batch ranking reward: r_c ≻ r_{rp} ≻ r_{rp+r}
         │
         ▼
Perception-Judge
```
