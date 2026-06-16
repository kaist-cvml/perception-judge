"""
Main entry point: generate synthetic preference data from annotation JSONL files.

For each annotation file the pipeline:
  1. Verifies the chosen response is correct
  2. Extracts visual perception attributes
  3. Generates 2 perception-perturbed rejected responses

Usage:
    # GPT-4o / GPT-5 (requires OPENAI_API_KEY in .env)
    python generate_synthetic_datasets_async.py \\
        --annotations_dir /path/to/annotations \\
        --images_root     /path/to/images \\
        --out_dir         /path/to/output \\
        --model           gpt-4o

    # Local vLLM server (Qwen or any OpenAI-compatible endpoint)
    python generate_synthetic_datasets_async.py \\
        --annotations_dir /path/to/annotations \\
        --images_root     /path/to/images \\
        --out_dir         /path/to/output \\
        --model           Qwen/Qwen3-VL-30B-A3B-Instruct \\
        --base_url        http://localhost:8000
"""

import argparse
import glob
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from processing_async import process_file_streaming_sync
from client_async import AsyncClient
from utils import Logger


def main():
    parser = argparse.ArgumentParser(description="Synthetic dataset generation (Async)")
    parser.add_argument("--annotations_dir", required=True,
                        help="Directory containing annotation JSONL files")
    parser.add_argument("--images_root", required=True,
                        help="Root directory of image files")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory (timestamped subdirectory will be created)")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Specific JSONL filenames to process (default: all *.jsonl in annotations_dir)")
    parser.add_argument("--max_rows", type=int, default=50,
                        help="Max samples to process per file (default: 50)")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--skip_rows", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model name. Use 'gpt-4o' or 'gpt-5' for OpenAI; "
                             "any model name for a local vLLM server (requires --base_url).")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for a vLLM-compatible API server (e.g. http://localhost:8000). "
                             "Required for non-OpenAI models.")
    args = parser.parse_args()

    # --- API setup ---
    api_key = None
    base_url = args.base_url

    openai_models = ["gpt-4o", "gpt-5"]
    if args.model in openai_models:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Set it in a .env file or as an environment variable.")
    else:
        if not base_url:
            raise ValueError(
                f"--base_url is required for model '{args.model}'. "
                "Provide the vLLM server URL, e.g. --base_url http://localhost:8000"
            )

    client = AsyncClient(
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_concurrent=args.max_concurrent,
    )
    Logger.info(f"Model: {args.model}", indent=0)

    # --- Output directory ---
    ts = datetime.now().strftime("%Y-%m%d-%H%M")
    model_tag = args.model.replace("/", "-")
    out_dir = os.path.join(os.path.dirname(os.path.normpath(args.out_dir)), f"synthetic-{model_tag}-{ts}")
    os.makedirs(out_dir, exist_ok=True)
    Logger.info(f"Output dir: {out_dir}", indent=0)

    # --- File list ---
    if args.files:
        files = [os.path.join(args.annotations_dir, f) for f in args.files]
    else:
        files = sorted(glob.glob(os.path.join(args.annotations_dir, "*.jsonl")))

    if not files:
        raise ValueError(f"No JSONL files found in {args.annotations_dir}")

    print(f"\n{'='*61}")
    print(f"Processing {len(files)} file(s)")
    print(f"{'='*61}")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    print(f"{'='*61}\n")

    total_success = 0
    total_processed = 0
    file_results = []

    for file_idx, file in enumerate(files, 1):
        print(f"\n{'+'*61}")
        print(f"| [{file_idx}/{len(files)}] {os.path.basename(file)}")
        print(f"{'+'*61}")

        base = os.path.basename(file).replace(".jsonl", "")
        out_path = os.path.join(out_dir, f"{base}_synthetic_perturbed.jsonl")

        success_count = process_file_streaming_sync(
            jsonl_path=file,
            images_root=args.images_root,
            client=client,
            output_path=out_path,
            max_rows=args.max_rows,
            batch_size=args.batch_size,
            skip_rows=args.skip_rows,
        )

        processed_count = args.max_rows if args.max_rows else success_count
        if success_count == 0:
            Logger.fail(f"No successful samples — skipping file")
            if os.path.exists(out_path):
                os.remove(out_path)
        else:
            Logger.success(f"Saved: {out_path} ({success_count} samples)")

        file_results.append((os.path.basename(file), success_count, processed_count))
        total_success += success_count
        total_processed += processed_count

    # --- Summary ---
    print(f"\n{'='*61}")
    print(f"All files done")
    print(f"{'='*61}")
    print(f"Output dir: {out_dir}\n")
    print(f"Total samples : {total_processed}")
    print(f"✅ Success     : {total_success} ({100*total_success/max(total_processed,1):.1f}%)")
    print(f"❌ Failed      : {total_processed - total_success}\n")
    print(f"Per-file breakdown:")
    for filename, success, processed in file_results:
        rate = 100 * success / processed if processed > 0 else 0
        icon = "✅" if rate > 50 else ("⚠️ " if rate > 20 else "❌")
        print(f"  {icon} {filename[:30]:30s}  {success:4d}/{processed:4d} ({rate:5.1f}%)")
    print(f"{'='*61}\n")


if __name__ == "__main__":
    main()
