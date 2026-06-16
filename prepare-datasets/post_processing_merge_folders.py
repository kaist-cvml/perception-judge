"""
Post-processing: merge two synthetic output folders into one.

Files that exist in both folders are row-merged (union).
Files that exist in only one folder are copied as-is.

Usage:
    python post_processing_merge_folders.py \\
        --folder1    /path/to/run_A \\
        --folder2    /path/to/run_B \\
        --output_dir /path/to/merged
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def read_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def write_jsonl(rows, path):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def merge_files(file1, file2):
    rows1 = read_jsonl(file1)
    rows2 = read_jsonl(file2)
    # Deduplicate by content (use JSON string as key)
    seen = set()
    merged = []
    for row in rows1 + rows2:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            merged.append(row)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge two synthetic output folders")
    parser.add_argument("--folder1",    required=True, help="First input folder")
    parser.add_argument("--folder2",    required=True, help="Second input folder")
    parser.add_argument("--output_dir", required=True, help="Output folder")
    args = parser.parse_args()

    folder1 = Path(args.folder1)
    folder2 = Path(args.folder2)
    output_folder = Path(args.output_dir)

    if output_folder.exists():
        shutil.rmtree(output_folder)
        print(f"Removed existing: {output_folder}")
    output_folder.mkdir(parents=True)
    print(f"Output: {output_folder}\n")

    files1 = {f.name for f in folder1.glob("*.jsonl")}
    files2 = {f.name for f in folder2.glob("*.jsonl")}
    common      = files1 & files2
    only_in_1   = files1 - files2
    only_in_2   = files2 - files1

    print(f"Folder1 only : {len(only_in_1)} file(s)")
    print(f"Folder2 only : {len(only_in_2)} file(s)")
    print(f"Common       : {len(common)} file(s)\n")

    stats = defaultdict(int)

    for name in tqdm(sorted(common), desc="Merging common files"):
        merged = merge_files(folder1 / name, folder2 / name)
        write_jsonl(merged, output_folder / name)
        stats['merged'] += 1
        print(f"  ✓ {name}: {merged.__len__()} rows")

    for name in sorted(only_in_1):
        shutil.copy(folder1 / name, output_folder / name)
        stats['copied'] += 1

    for name in sorted(only_in_2):
        shutil.copy(folder2 / name, output_folder / name)
        stats['copied'] += 1

    print(f"\n{'='*60}")
    print(f"Done — {stats['merged']} merged, {stats['copied']} copied")
    print(f"Output: {output_folder}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
