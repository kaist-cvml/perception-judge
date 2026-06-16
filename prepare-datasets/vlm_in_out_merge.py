"""
Merge paraphrase outputs from multiple models into a single dataset.

For each record in the base directory, randomly selects chosen / rejected_1 /
rejected_2 from among base + model paraphrase variants, and records provenance.

Usage:
    python vlm_in_out_merge.py \\
        --base_dir    /path/to/base-output \\
        --model_dirs  '{"model_a": "/path/to/base_model_a_01", "model_b": "/path/to/base_model_b_01"}' \\
        --output_dir  /path/to/merged   # default: <base_dir>_merge

The --model_dirs value is a JSON object mapping a short model name to its directory.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_jsonl(file_path, use_line_number=False):
    data = {}
    if not Path(file_path).exists():
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                row_idx = line_num if use_line_number else item.get('row_index')
                if row_idx is None:
                    continue
                if use_line_number:
                    item['row_index'] = row_idx
                data[row_idx] = item
    return data


def merge_data(base_data, model_datas, file_name=""):
    BASE_MODEL = "base"
    all_models = {BASE_MODEL: base_data, **model_datas}
    merged_results = []
    stats = {'total': 0, 'matched': defaultdict(int), 'missing': defaultdict(int)}

    for row_idx in sorted(base_data.keys()):
        stats['total'] += 1
        base_item = base_data[row_idx]
        candidates = {}

        if '__normalized__' in base_item:
            norm = base_item['__normalized__']
            candidates[BASE_MODEL] = {
                'chosen':     norm.get('chosen'),
                'rejected_1': norm.get('rejected_perception_1'),
                'rejected_2': norm.get('rejected_perception_2'),
            }
            stats['matched'][BASE_MODEL] += 1

        for model_name, model_data in model_datas.items():
            if row_idx not in model_data:
                stats['missing'][model_name] += 1
                continue
            model_item = model_data[row_idx]
            if model_item.get('row_index') != row_idx:
                continue
            if base_item.get('image') != model_item.get('image'):
                continue
            if '__normalized__' in model_item:
                norm = model_item['__normalized__']
                candidates[model_name] = {
                    'chosen':     norm.get('chosen'),
                    'rejected_1': norm.get('rejected_perception_1'),
                    'rejected_2': norm.get('rejected_perception_2'),
                }
                stats['matched'][model_name] += 1

        if not candidates:
            continue

        available = list(candidates.keys())
        selected = {
            'chosen':     random.choice(available),
            'rejected_1': random.choice(available),
            'rejected_2': random.choice(available),
        }

        merged_item = {**base_item}
        if '__normalized__' not in merged_item:
            merged_item['__normalized__'] = {}

        norm = merged_item['__normalized__']
        for field, model_name in selected.items():
            norm[field] = candidates[model_name][field]
        norm['chosen_from_model']     = selected['chosen']
        norm['rejected_1_from_model'] = selected['rejected_1']
        norm['rejected_2_from_model'] = selected['rejected_2']

        merged_results.append(merged_item)

    return merged_results, stats


def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Merge multi-model paraphrase outputs")
    parser.add_argument("--base_dir",    required=True,
                        help="Base output directory (the GPT/VLM generation output)")
    parser.add_argument("--model_dirs",  required=True,
                        help='JSON dict mapping model name → paraphrase output dir. '
                             'Example: \'{"qwen": "/path/to/dir1", "gemma": "/path/to/dir2"}\'')
    parser.add_argument("--output_dir",  default=None,
                        help="Output directory (default: <base_dir>_merge)")
    args = parser.parse_args()

    BASE_DIR   = Path(args.base_dir)
    MODEL_DIRS = {k: Path(v) for k, v in json.loads(args.model_dirs).items()}
    OUTPUT_DIR = Path(args.output_dir) if args.output_dir else Path(str(BASE_DIR) + "_merge")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Base dir   : {BASE_DIR}")
    print(f"Model dirs : {MODEL_DIRS}")
    print(f"Output dir : {OUTPUT_DIR}\n")

    base_files = sorted(BASE_DIR.glob("*.jsonl"))
    print(f"Processing {len(base_files)} file(s)...")

    for base_file in tqdm(base_files, desc="Merging"):
        name = base_file.name
        base_data   = load_jsonl(base_file, use_line_number=True)
        model_datas = {k: load_jsonl(d / name) for k, d in MODEL_DIRS.items()}

        merged, stats = merge_data(base_data, model_datas, file_name=name)
        save_jsonl(merged, OUTPUT_DIR / name)

        print(f"\n  ✓ {name}: {len(merged)}/{stats['total']} rows merged")
        for model_name in ["base"] + list(MODEL_DIRS.keys()):
            matched = stats['matched'].get(model_name, 0)
            missing = stats['missing'].get(model_name, 0)
            print(f"    {model_name}: matched={matched}, missing={missing}")

    print(f"\nDone → {OUTPUT_DIR}")


if __name__ == "__main__":
    random.seed(42)
    main()
