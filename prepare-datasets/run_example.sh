#!/bin/bash
# Example: generate synthetic preference data
# Run from the project root directory.

ANNOTATIONS_DIR="/path/to/annotations"   # directory with *.jsonl annotation files
IMAGES_ROOT="/path/to/images"            # root directory of image files
OUT_DIR="/path/to/output/synthetic"

# ── Option A: OpenAI (GPT-4o / GPT-5) ───────────────────────────────────────
# Requires OPENAI_API_KEY in .env or environment.
python prepare-datasets/generate_synthetic_datasets_async.py \
    --annotations_dir "$ANNOTATIONS_DIR" \
    --images_root     "$IMAGES_ROOT" \
    --out_dir         "$OUT_DIR" \
    --model           gpt-4o \
    --max_rows        50 \
    --batch_size      5 \
    --max_concurrent  10

# ── Option B: Local vLLM server ───────────────────────────────────────────────
# Start a vLLM server first, then point --base_url at it.
# python prepare-datasets/generate_synthetic_datasets_async.py \
#     --annotations_dir "$ANNOTATIONS_DIR" \
#     --images_root     "$IMAGES_ROOT" \
#     --out_dir         "$OUT_DIR" \
#     --model           Qwen/Qwen3-VL-30B-A3B-Instruct \
#     --base_url        http://localhost:8000 \
#     --max_rows        50 \
#     --batch_size      5 \
#     --max_concurrent  10

# ── Post-processing ──────────────────────────────────────────────────────────
# python prepare-datasets/post_processing_filter_after_gen.py \
#     --input_dir  "$OUT_DIR/synthetic-gpt-4o-2025-XXXX-XXXX" \
#     --output_dir "$OUT_DIR/synthetic-gpt-4o-2025-XXXX-XXXX_filtered"

# ── Optional: offline VLM paraphrase (requires vLLM) ─────────────────────────
# python prepare-datasets/vlm_in_out_multi_gpu.py \
#     --input_dir   "$OUT_DIR/synthetic-gpt-4o-2025-XXXX-XXXX_filtered" \
#     --model       Qwen/Qwen2.5-VL-7B-Instruct \
#     --tensor-parallel-size 2
