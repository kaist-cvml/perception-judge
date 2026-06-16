"""
Generate rejected samples from a score JSONL file.

Reads a score.jsonl where each row has model responses with human scores.
Selects the response with human=5 as `chosen`, then uses a VLM to generate
2 perception-perturbed `rejected` responses.

Usage:
    # OpenAI
    python generate_rejected_from_score.py \\
        --score_jsonl /path/to/score.jsonl \\
        --images_root /path/to/images \\
        --out_dir     /path/to/output \\
        --model       gpt-4o

    # Local vLLM server
    python generate_rejected_from_score.py \\
        --score_jsonl /path/to/score.jsonl \\
        --images_root /path/to/images \\
        --out_dir     /path/to/output \\
        --model       Qwen/Qwen3-VL-30B-A3B-Instruct \\
        --base_url    http://localhost:8000
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client_async import AsyncClient
from utils import Logger, iter_jsonl
from processing_async import parse_rejecteds_response
from prompts.prompts import PROMPT_REWRITE_REJECTEDS


def group_by_id(score_jsonl_path: str) -> Dict[int, List[Dict]]:
    groups = defaultdict(list)
    for record in iter_jsonl(score_jsonl_path):
        qid = record.get("id")
        if qid is not None:
            groups[qid].append(record)
    return dict(groups)


def select_chosen(group: List[Dict]) -> Optional[Dict]:
    """Return a record dict with chosen set to the human=5 response, or None."""
    chosen_record = next((r for r in group if r.get("human") == "5"), None)
    if not chosen_record:
        return None
    return {
        "id":               chosen_record.get("id"),
        "score_id":         chosen_record.get("score_id"),
        "instruction":      chosen_record.get("instruction", ""),
        "image_path":       chosen_record.get("image_path", ""),
        "original_dataset": chosen_record.get("original_dataset", ""),
        "chosen":           chosen_record.get("answer", ""),
        "chosen_model":     chosen_record.get("name", "unknown"),
    }


async def generate_rejected(client, image_path, instruction, chosen, idx) -> Optional[Dict]:
    perception_attrs = ["visual detail", "object count", "color", "spatial relationship"]
    prompt = PROMPT_REWRITE_REJECTEDS.format(
        question=instruction,
        answer_gt="",
        chosen=chosen,
        perception_attrs_json=json.dumps(perception_attrs, ensure_ascii=False),
    )
    try:
        response = await client.call_api(prompt=prompt, image_path=image_path)
        return parse_rejecteds_response(response) if response else None
    except Exception as e:
        Logger.fail(f"[{idx}] Exception: {e}", indent=2)
        return None


async def process_all_async(client, records, images_root, output_path, batch_size=5) -> int:
    success_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start:batch_start + batch_size]
            Logger.info(f"Batch [{batch_start+1}-{batch_start+len(batch)}/{len(records)}]", indent=1)

            tasks = [
                generate_rejected(
                    client,
                    os.path.join(images_root, rec["image_path"]),
                    rec["instruction"],
                    rec["chosen"],
                    batch_start + i,
                )
                for i, rec in enumerate(batch)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                idx = batch_start + i
                rec = batch[i]
                if isinstance(result, Exception) or not result:
                    Logger.fail(f"[{idx}] Failed", indent=2)
                    continue

                output_rec = {
                    "id":               rec["id"],
                    "score_id":         rec["score_id"],
                    "instruction":      rec["instruction"],
                    "image_path":       rec["image_path"],
                    "original_dataset": rec["original_dataset"],
                    "chosen":           rec["chosen"],
                    "chosen_model":     rec["chosen_model"],
                    "rejected_1":       result.get("rejected_perception_answer_1", ""),
                    "rejected_1_text":  result.get("rejected_perception_1", ""),
                    "rejected_2":       result.get("rejected_perception_answer_2", ""),
                    "rejected_2_text":  result.get("rejected_perception_2", ""),
                }
                f.write(json.dumps(output_rec, ensure_ascii=False) + "\n")
                f.flush()
                success_count += 1
                Logger.success(f"[{idx}] Done", indent=2)

            await asyncio.sleep(0.5)
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate rejected samples from score.jsonl (human=5 as chosen)"
    )
    parser.add_argument("--score_jsonl",   required=True,
                        help="Path to score.jsonl file")
    parser.add_argument("--images_root",   required=True,
                        help="Root directory of image files")
    parser.add_argument("--out_dir",       required=True,
                        help="Output directory")
    parser.add_argument("--max_rows",      type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--batch_size",    type=int, default=5)
    parser.add_argument("--max_concurrent",type=int, default=10)
    parser.add_argument("--temperature",   type=float, default=0.7)
    parser.add_argument("--top_p",         type=float, default=0.9)
    parser.add_argument("--model",         type=str, default="gpt-4o",
                        choices=["gpt-4o", "gpt-5"])
    parser.add_argument("--base_url",      type=str, default=None,
                        help="Base URL for a vLLM-compatible API server. "
                             "Required for non-OpenAI models.")
    args = parser.parse_args()

    # --- API setup ---
    api_key = None
    base_url = args.base_url

    if args.model in ["gpt-4o", "gpt-5"]:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Set it in a .env file.")
    else:
        if not base_url:
            raise ValueError(f"--base_url is required for model '{args.model}'.")

    client = AsyncClient(
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_concurrent=args.max_concurrent,
    )

    ts = datetime.now().strftime("%Y-%m%d-%H%M")
    model_tag = args.model.split("/")[-1]
    out_dir = os.path.join(args.out_dir, f"rejected-from-score-{model_tag}-{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    Logger.info(f"Model       : {args.model}", indent=0)
    Logger.info(f"Score file  : {args.score_jsonl}", indent=0)
    Logger.info(f"Images root : {args.images_root}", indent=0)
    Logger.info(f"Output dir  : {out_dir}", indent=0)
    print(f"{'='*60}\n")

    Logger.section("Loading score.jsonl")
    groups = group_by_id(args.score_jsonl)
    Logger.success(f"Loaded {len(groups)} question groups", indent=1)

    Logger.section("Selecting chosen (human=5)")
    valid_records = [sel for g in groups.values() if (sel := select_chosen(g))]
    skipped = len(groups) - len(valid_records)
    Logger.success(f"Groups with human=5: {len(valid_records)}", indent=1)
    Logger.info(f"Skipped (no human=5): {skipped}", indent=1)

    if args.max_rows:
        valid_records = valid_records[:args.max_rows]
        Logger.info(f"Limited to {len(valid_records)} samples", indent=1)

    Logger.section(f"Generating rejected samples ({len(valid_records)} total)")
    output_path = os.path.join(out_dir, "rejected_pairs.jsonl")

    success_count = asyncio.run(
        process_all_async(
            client=client,
            records=valid_records,
            images_root=args.images_root,
            output_path=output_path,
            batch_size=args.batch_size,
        )
    )

    total = len(valid_records)
    print(f"\n{'='*60}")
    print(f"Output : {output_path}")
    print(f"Total  : {total}")
    print(f"✅ Success : {success_count} ({100*success_count/max(total,1):.1f}%)")
    print(f"❌ Failed  : {total - success_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
