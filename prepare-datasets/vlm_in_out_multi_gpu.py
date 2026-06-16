"""
Offline VLM paraphrase using vLLM (Tensor Parallel).

Takes a directory of filtered synthetic JSONL files and paraphrases the
chosen / rejected_perception_1 / rejected_perception_2 fields using a
locally-loaded LLM via vLLM, producing stylistically diverse alternatives.

Usage:
    python vlm_in_out_multi_gpu.py \\
        --input_dir   /path/to/filtered-output \\
        --model       Qwen/Qwen2.5-VL-7B-Instruct \\
        --tensor-parallel-size 2

    CUDA_VISIBLE_DEVICES=0,1,2,3 python vlm_in_out_multi_gpu.py \\
        --input_dir   /path/to/filtered-output \\
        --model       Qwen/Qwen3-VL-30B-A3B-Instruct \\
        --tensor-parallel-size 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import Logger, iter_jsonl


ENFORCE_EAGER = False
GPU_MEMORY_UTILIZATION = 0.8
TEMPERATURE = 0.8
TOP_P = 0.6
MAX_TOKENS = 4096

PARAPHRASE_PROMPT = """Rewrite the following response in a completely different style and structure, as if written by another person.

**MUST PRESERVE:**
- All factual information, reasoning logic, and visual perception details
- Final answer format and structure (labels, sections, conclusion)
- Original response length (keep a similar length as the input text)

**YOU MAY CHANGE:**
- Sentence order and paragraph structure
- Transitions and pacing
- How ideas are grouped or expressed

**DO NOT:**
- Add new information or reasoning
- Fix errors (keep all existing errors)
- Change the final answer or its format

Text to rewrite:
{text}

Provide ONLY the rewritten text itself.
Do NOT include any preamble, explanations, meta-commentary, or descriptions of what you did.
Output the result directly without additional context.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Offline VLM paraphrase (multi-GPU)")
    parser.add_argument("--input_dir",           required=True,
                        help="Directory with filtered synthetic JSONL files")
    parser.add_argument("--model",               type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--tensor-parallel-size",type=int, default=1)
    parser.add_argument("--batch-size",          type=int, default=64)
    return parser.parse_args()


def prepare_batch_prompts(records, processor):
    prompts_list, metadata_list = [], []
    for idx, record in records:
        if "__normalized__" not in record:
            continue
        norm = record["__normalized__"]
        chosen_text      = norm.get("chosen")
        rejected_1_text  = norm.get("rejected_perception_1")
        rejected_2_text  = norm.get("rejected_perception_2")
        if not chosen_text:
            continue

        texts = [(chosen_text, "chosen")]
        if rejected_1_text:
            texts.append((rejected_1_text, "rejected_1"))
        if rejected_2_text:
            texts.append((rejected_2_text, "rejected_2"))

        for text, field_type in texts:
            prompt_text = processor.apply_chat_template(
                [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts_list.append({"prompt": prompt_text})
            metadata_list.append((idx, field_type))

    return prompts_list, metadata_list


def process_records(records_with_indices, llm, processor, batch_size, output_file_handle):
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
    processed_count = 0

    with tqdm(total=len(records_with_indices), desc="Processing") as pbar:
        for i in range(0, len(records_with_indices), batch_size):
            batch = records_with_indices[i:i+batch_size]
            prompts_list, metadata_list = prepare_batch_prompts(batch, processor)
            if not prompts_list:
                pbar.update(len(batch))
                continue

            try:
                outputs = llm.generate(prompts_list, sampling_params=sampling_params)
                output_idx = 0
                processed_indices = set()

                for orig_idx, record in batch:
                    if "__normalized__" not in record or orig_idx in processed_indices:
                        continue
                    norm = record["__normalized__"]
                    n_outputs = sum([
                        1,
                        bool(norm.get("rejected_perception_1")),
                        bool(norm.get("rejected_perception_2")),
                    ])
                    new_record = {**record, "row_index": orig_idx}
                    for j in range(n_outputs):
                        if output_idx >= len(outputs):
                            break
                        _, field_type = metadata_list[output_idx]
                        generated = outputs[output_idx].outputs[0].text.strip()
                        if field_type == "chosen":
                            new_record["__normalized__"]["chosen_1"] = generated
                        elif field_type == "rejected_1":
                            new_record["__normalized__"]["rejected_1"] = generated
                        elif field_type == "rejected_2":
                            new_record["__normalized__"]["rejected_2"] = generated
                        output_idx += 1

                    output_file_handle.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                    output_file_handle.flush()
                    processed_count += 1
                    processed_indices.add(orig_idx)

            except Exception as e:
                print(f"❌ Batch error: {e}")

            pbar.update(len(batch))

    return processed_count


def process_file(input_file, output_file, llm, processor, batch_size):
    Logger.section(f"Processing: {os.path.basename(input_file)}")
    all_records = list(enumerate(iter_jsonl(input_file)))
    Logger.info(f"Total records: {len(all_records)}")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        processed_count = process_records(all_records, llm, processor, batch_size, f)
    Logger.info(f"Saved {processed_count} records → {output_file}")


if __name__ == "__main__":
    args = parse_args()
    INPUT_DIR = args.input_dir
    MODEL = args.model
    TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
    BATCH_SIZE = args.batch_size

    model_tag = MODEL.split('/')[-1].lower().replace('-', '_').replace('.', '_')
    base_output_dir = f"{INPUT_DIR.rstrip('/')}_{model_tag}"
    idx = 1
    while True:
        output_dir = f"{base_output_dir}_{idx:02d}"
        if not os.path.exists(output_dir):
            break
        idx += 1

    Logger.section("Offline VLM Paraphrase")
    Logger.info(f"Model               : {MODEL}")
    Logger.info(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    Logger.info(f"Batch size          : {BATCH_SIZE}")
    Logger.info(f"Input dir           : {INPUT_DIR}")
    Logger.info(f"Output dir          : {output_dir}")

    input_files = sorted(Path(INPUT_DIR).glob("*.jsonl"))
    Logger.info(f"Files found: {len(input_files)}")

    Logger.section("Loading model")
    llm = LLM(
        MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        enforce_eager=ENFORCE_EAGER,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        max_model_len=65536,
    )
    processor = AutoProcessor.from_pretrained(MODEL, use_fast=True)
    Logger.success("Model loaded")

    start_time = time.time()
    for input_file in input_files:
        process_file(str(input_file), str(Path(output_dir) / input_file.name), llm, processor, BATCH_SIZE)

    Logger.result(f"All done in {time.time()-start_time:.2f}s")
