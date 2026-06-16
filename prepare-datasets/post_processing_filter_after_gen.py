"""
Post-processing: remove PERCEPTION_ATTRS artifact from generated outputs.

The generation pipeline occasionally leaves a trailing `\nPERCEPTION_ATTRS: {...}`
block inside chosen / rejected_perception_1 / rejected_perception_2 fields.
This script strips those blocks and writes clean copies of all JSONL files.

Usage:
    python post_processing_filter_after_gen.py \\
        --input_dir  /path/to/synthetic-output \\
        --output_dir /path/to/filtered-output
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from tqdm import tqdm

PERCEPTION_PATTERN = re.compile(r'\nPERCEPTION_ATTRS:\s*\{[^\}]*\}', re.DOTALL)

FIELDS_TO_CLEAN = ["chosen", "rejected_perception_1", "rejected_perception_2"]


def clean_text(text):
    if text is None:
        return text
    return PERCEPTION_PATTERN.sub('', text)


def process_row(row):
    removed = 0
    norm = row.get('__normalized__')
    if not norm:
        return row, removed
    for field in FIELDS_TO_CLEAN:
        original = norm.get(field)
        if original:
            cleaned = clean_text(original)
            if cleaned != original:
                removed += len(PERCEPTION_PATTERN.findall(original))
            norm[field] = cleaned
    return row, removed


def process_file(input_file, output_file):
    processed, removed = 0, 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row, n = process_row(json.loads(line))
                fout.write(json.dumps(row, ensure_ascii=False) + '\n')
                processed += 1
                removed += n
            except json.JSONDecodeError as e:
                print(f"JSON error in {input_file}: {e}")
    return processed, removed


def main():
    parser = argparse.ArgumentParser(description="Remove PERCEPTION_ATTRS from generated JSONL files")
    parser.add_argument("--input_dir",  required=True, help="Directory with generated JSONL files")
    parser.add_argument("--output_dir", required=True, help="Output directory for cleaned files")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL file(s) in {input_dir}")
    print(f"Output: {output_dir}\n")

    total_rows, total_removed = 0, 0
    for input_file in tqdm(jsonl_files, desc="Processing"):
        rows, removed = process_file(input_file, output_dir / input_file.name)
        total_rows += rows
        total_removed += removed
        print(f"  ✓ {input_file.name}: {rows} rows, {removed} patterns removed")

    print(f"\n{'='*60}")
    print(f"Done — {total_rows} rows processed, {total_removed} patterns removed")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
