"""
Data processing pipeline - Async version.

Pipeline per record:
  0. Validate chosen (length, no think tokens)
  1. Verify answer against GT
  2. Extract perception attributes
  3. Rewrite chosen with wrong perception → 2 rejected samples
"""

import asyncio
import itertools
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import infer_field_mapping, normalize_record
from client_async import AsyncClient, run_async
from utils import Logger, iter_jsonl, normalize_answer
from prompts.prompts import (
    PROMPT_VERIFY_ANSWER,
    PROMPT_EXTRACT_PERCEPTION,
    PROMPT_REWRITE_REJECTEDS,
)


# Maps annotation filename prefix → image subdirectory prefix
DATASET_PREFIX_MAP = {
    "ai2d_": "ai2d/",
    "chartqa_": "chartqa/",
    "CLEVR_math_": "CLEVR/",
    "docvqa_": "docvqa/",
    "dvqa_": "dvqa/",
    "figureqa_": "FigureQA/",
    "geo170k_": "Geo170K/",
    "geometry3k_": "Geometry3K/",
    "geomverse_": "GeomVerse/",
    "geoqa+_": "geoqa_plus/",
    "geos_": "GEOS/",
    "gqa_": "gqa/",
    "iconqa_": "iconqa/",
    "inat_": "inat2018/",
    "infographics_": "InfoVQA/",
    "koniq10k_": "koniq/",
    "llavar_": "LLaVAR/",
    "m3cot_": "M3CoT/",
    "mapqa_": "MapQA/",
    "MathV360K_": "MathV360K/",
    "mavis_function_": "MAVIS-Function/",
    "mavis_geo_": "MAVIS-Geometry/",
    "nlvr2_": "nlvr2/",
    "okvqa_": "coco/",
    "openbmb_": "RLAIF-V/",
    "RLAIF-V-Dataset": "RLAIF-V/",
    "sam_": "SA-1B/",
    "scienceqa_": "ScienceQA/",
    "spot_the_diff_": "spot-the-diff/",
    "SROIE_": "SROIE/",
    "super_clevr_": "Super-CLEVR/",
    "tabmwp_": "TabMWP/",
    "tallyqa_": "vg/",
    "textvqa_": "TextVQA/",
    "unigeo_": "UniGeo/",
    "vqav2_": "coco/",
    "vsr_": "coco/",
    "wildvision_": "wildvision/",
    "_data_gen": "vg/",
}

MAX_CHOSEN_LENGTH = 1600


# ============================================================================
# JSON parsing helpers
# ============================================================================

def parse_json_response(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace >= 0 and last_brace > first_brace:
        json_text = text[first_brace:last_brace+1]
        try:
            return json.loads(json_text)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            fixed = json_text.replace('\\[', '\\\\[').replace('\\]', '\\\\]')
            fixed = fixed.replace('\\(', '\\\\(').replace('\\)', '\\\\)')
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def parse_verify_result(text: str) -> bool:
    obj = parse_json_response(text)
    return obj.get("is_correct", False) is True if obj else False


def parse_perception_attrs(text: str) -> List[str]:
    obj = parse_json_response(text)
    if not obj:
        return []
    attrs = obj.get("perception_attrs", [])
    if not isinstance(attrs, list):
        return []
    cleaned, seen = [], set()
    for attr in attrs:
        if not isinstance(attr, (str, int, float)):
            continue
        attr_str = str(attr).strip()
        if not attr_str or len(attr_str) > 150:
            continue
        norm = ' '.join(attr_str.lower().split())
        if norm not in seen:
            seen.add(norm)
            cleaned.append(attr_str)
            if len(cleaned) >= 6:
                break
    return cleaned


def extract_answer_from_text(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r'\\boxed\s*\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r'[Ff]inal\s+[Aa]nswer\s*:\s*(.+?)(?:\n|$)', text)
    if matches:
        return re.sub(r'[.!?;,]+$', '', matches[-1].strip()).strip()
    return ""


def parse_rejecteds_response(text: str) -> Optional[Dict[str, str]]:
    obj = parse_json_response(text)
    if not obj:
        return None
    result = {}
    for i in [1, 2]:
        rp = obj.get(f"rejected_perception_{i}", "")
        if isinstance(rp, str) and rp:
            result[f"rejected_perception_{i}"] = rp
            result[f"rejected_perception_answer_{i}"] = extract_answer_from_text(rp)
    return result if len(result) >= 4 else None


# ============================================================================
# Async processing
# ============================================================================

async def process_record_pipeline(rec, idx, norm, client):
    """
    End-to-end pipeline for a single record.
    Returns (completed_record, reason) where reason is "success" or a failure code.
    """
    total_start = time.time()
    image_name = "unknown"
    try:
        image_path = norm.get("image_path")
        question   = norm.get("question")
        answer     = norm.get("answer")
        chosen     = norm.get("chosen")

        image_name = os.path.basename(image_path) if image_path else "unknown"
        q_preview  = (question or "")[:40].replace('\n', ' ')

        print(f"\n{'='*61}")
        print(f"[Sample {idx}]  image: {image_name[:40]}  q: {q_preview}...")
        print(f"{'='*61}")

        if not all([image_path, question, answer, chosen]):
            Logger.fail(f"[{idx}] Missing required fields", indent=1)
            return None, "missing_fields"

        if not os.path.exists(image_path):
            Logger.fail(f"[{idx}] Image not found: {image_path}", indent=1)
            return None, "image_load_error"

        # --- Step 0: validate chosen ---
        if any(p in chosen.lower() for p in ['<think>', '</think>', '<thinking>', '</thinking>']):
            Logger.fail(f"[{idx}] chosen contains think token — skip", indent=1)
            return None, "chosen_has_think_token"

        if len(chosen) > MAX_CHOSEN_LENGTH:
            Logger.fail(f"[{idx}] chosen too long ({len(chosen)} > {MAX_CHOSEN_LENGTH}) — skip", indent=1)
            return None, "chosen_too_long"

        # --- Step 1: verify answer ---
        t = time.time()
        Logger.info(f"[{idx}] Step 1/3: verify answer", indent=1)
        verify_raw = await client.call_api(
            image_path,
            PROMPT_VERIFY_ANSWER.format(question=question, answer_gt=answer, chosen=chosen),
        )
        Logger.info(f"[{idx}] Step 1/3 done ({time.time()-t:.2f}s)", indent=1)

        if not parse_verify_result(verify_raw):
            Logger.fail(f"[{idx}] Wrong answer — skip", indent=1)
            return None, "verify_failed"
        Logger.success(f"[{idx}] Answer verified", indent=1)

        # --- Step 2: extract perception ---
        t = time.time()
        Logger.info(f"[{idx}] Step 2/3: extract perception", indent=1)
        perception_raw = await client.call_api(
            image_path,
            PROMPT_EXTRACT_PERCEPTION.format(question=question, answer_gt=answer, chosen=chosen),
            max_tokens=512,
        )
        Logger.info(f"[{idx}] Step 2/3 done ({time.time()-t:.2f}s)", indent=1)

        perception_attrs = parse_perception_attrs(perception_raw)
        if len(perception_attrs) < 3:
            Logger.fail(f"[{idx}] Insufficient perception attrs ({len(perception_attrs)}) — skip", indent=1)
            return None, "perception_insufficient"
        Logger.success(f"[{idx}] Perception: {perception_attrs}", indent=1)

        # --- Step 3: rewrite rejected ---
        t = time.time()
        Logger.info(f"[{idx}] Step 3/3: rewrite rejected", indent=1)
        rejected_raw = await client.call_api(
            image_path,
            PROMPT_REWRITE_REJECTEDS.format(
                question=question,
                answer_gt=answer,
                chosen=chosen,
                perception_attrs_json=json.dumps({"perception_attrs": perception_attrs}, ensure_ascii=False),
            ),
        )
        Logger.info(f"[{idx}] Step 3/3 done ({time.time()-t:.2f}s)", indent=1)

        parsed = parse_rejecteds_response(rejected_raw)
        if not parsed:
            Logger.fail(f"[{idx}] Failed to parse 2 rejected samples", indent=1)
            return None, "rejected_parse_failed"

        # --- Step 4: validate rejected != GT ---
        answer_norm = normalize_answer(answer)
        valid_count = 0
        for i in [1, 2]:
            rej_ans = parsed.get(f"rejected_perception_answer_{i}", "")
            if normalize_answer(rej_ans) == answer_norm:
                Logger.fail(f"[{idx}] rejected_{i} same as GT — skip", indent=1)
            else:
                valid_count += 1
                Logger.success(f"[{idx}] rejected_{i}: GT={answer} → rej={rej_ans}", indent=1)

        if valid_count == 0:
            Logger.fail(f"[{idx}] All rejected samples equal GT", indent=1)
            return None, "same_as_gt"

        total_time = time.time() - total_start
        Logger.success(f"[{idx}] Done — {image_name} ({valid_count}/2 valid, {total_time:.2f}s)", indent=1)

        rec_out = dict(rec)
        rec_out["__normalized__"] = {
            "image_path": image_path,
            "question":   question,
            "answer":     answer,
            "chosen":     chosen,
            "rejected_perception_1":        parsed["rejected_perception_1"],
            "rejected_perception_answer_1": parsed["rejected_perception_answer_1"],
            "rejected_perception_2":        parsed["rejected_perception_2"],
            "rejected_perception_answer_2": parsed["rejected_perception_answer_2"],
        }
        return rec_out, "success"

    except Exception as e:
        total_time = time.time() - total_start
        Logger.fail(f"[{idx}] Exception ({total_time:.2f}s): {type(e).__name__}: {e}", indent=1)
        error_msg = str(e).lower()
        if any(k in error_msg for k in ['image', 'file not found', 'no such file', 'cannot identify image']):
            return None, "image_load_error"
        if type(e).__name__ in ['FileNotFoundError', 'IOError', 'OSError']:
            return None, "image_load_error"
        return None, "exception"


async def process_records_batch(records, client, batch_size=5):
    tasks = [process_record_pipeline(rec, idx, norm, client) for rec, idx, norm in records]
    results = []
    for i in range(0, len(tasks), batch_size):
        batch_results = await asyncio.gather(*tasks[i:i+batch_size], return_exceptions=True)
        for r in batch_results:
            results.append((None, "exception") if isinstance(r, Exception) else r)
    return results


async def process_file_streaming(
    jsonl_path, images_root, client, output_path,
    max_rows=None, batch_size=5, skip_rows=0,
):
    """
    Stream-process a JSONL file: read → pipeline → append to output.
    Returns number of successfully processed samples.
    """
    Logger.section(f"Processing: {os.path.basename(jsonl_path)}")

    mapping = infer_field_mapping(jsonl_path, sample_lines=2000)
    Logger.info(f"Field mapping: {mapping}", indent=0)

    file_basename = os.path.basename(jsonl_path)
    image_prefix = next(
        (prefix for pattern, prefix in DATASET_PREFIX_MAP.items() if file_basename.startswith(pattern)),
        None
    )
    Logger.info(f"Image prefix: {image_prefix}", indent=0)

    FAILURE_LABELS = {
        "missing_fields": "Missing fields",
        "image_load_error": "Image load error",
        "chosen_has_think_token": "Think token in chosen",
        "chosen_too_long": "Chosen too long",
        "verify_failed": "Wrong answer",
        "perception_insufficient": "Insufficient perception (<3)",
        "rejected_parse_failed": "Rejected parse failed",
        "same_as_gt": "Rejected == GT",
        "exception": "Exception",
    }
    failure_stats = {k: 0 for k in FAILURE_LABELS}

    success_count = 0
    total_processed = 0
    consecutive_image_error_batches = 0

    iterator = iter_jsonl(jsonl_path)
    iterator = itertools.islice(iterator, skip_rows, None)
    if max_rows:
        iterator = itertools.islice(iterator, max_rows)

    while True:
        batch = []
        for _ in range(batch_size):
            try:
                rec = next(iterator)
                total_processed += 1
                norm = normalize_record(rec, mapping)
                rel_path = norm.get("image_path")
                norm["image_path"] = os.path.join(images_root, f"{image_prefix}{rel_path}")
                batch.append((rec, total_processed, norm))
            except StopIteration:
                break

        if not batch:
            break

        print(f"\n{'#'*60}")
        print(f"📦 Batch {total_processed-len(batch)+1}-{total_processed} ({len(batch)} samples)")
        print(f"{'#'*60}")

        batch_start = time.time()
        results = await process_records_batch(batch, client, batch_size)
        batch_time = time.time() - batch_start
        batch_stats = {k: 0 for k in FAILURE_LABELS}

        with open(output_path, 'a', encoding='utf-8') as f:
            for rec, reason in results:
                if reason == "success":
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    success_count += 1
                else:
                    failure_stats[reason] += 1
                    batch_stats[reason] += 1

        success_in_batch = sum(1 for _, r in results if r == "success")
        image_errors = batch_stats["image_load_error"]
        if image_errors == len(results):
            consecutive_image_error_batches += 1
            if consecutive_image_error_batches >= 2:
                raise RuntimeError(
                    f"{consecutive_image_error_batches} consecutive batches all failed with image errors. "
                    f"Check images_root={images_root} and prefix={image_prefix}"
                )
        elif success_in_batch > 0:
            consecutive_image_error_batches = 0

        print(f"✅ Batch done: {success_in_batch}/{len(results)} success | {batch_time:.2f}s")
        if success_in_batch < len(results):
            for reason, count in batch_stats.items():
                if count > 0:
                    print(f"   - {FAILURE_LABELS[reason]}: {count}")
        if total_processed > 0:
            print(f"   Cumulative: {success_count}/{total_processed} ({100*success_count/total_processed:.1f}%)")

    # Final stats
    print(f"\n{'='*60}")
    print(f"📊 File done: {os.path.basename(jsonl_path)}")
    if total_processed > 0:
        print(f"✅ Success: {success_count}/{total_processed} ({100*success_count/total_processed:.1f}%)")
        failures = total_processed - success_count
        if failures:
            print(f"❌ Failures: {failures}")
            for reason, count in failure_stats.items():
                if count > 0:
                    print(f"   - {FAILURE_LABELS[reason]}: {count} ({100*count/total_processed:.1f}%)")
    print(f"{'='*60}\n")

    Logger.result(f"Done: {success_count}/{total_processed} success")
    return success_count


def process_file_streaming_sync(
    jsonl_path: str,
    images_root: str,
    client: AsyncClient,
    output_path: str,
    max_rows: Optional[int] = None,
    batch_size: int = 5,
    skip_rows: int = 0,
) -> int:
    return run_async(
        process_file_streaming(jsonl_path, images_root, client, output_path, max_rows, batch_size, skip_rows)
    )
