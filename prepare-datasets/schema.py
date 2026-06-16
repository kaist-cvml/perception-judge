import json
import os
from typing import Dict, Optional, Iterable


NORMALIZED_KEYS = {
    "image_path": ["image_path", "image", "image_url", "img_path", "img", "image_path_abs"],
    "question":   ["question", "instruction", "prompt", "query"],
    "answer":     ["answer", "answer_gt", "ground_truth", "gt", "label"],
    "chosen":     ["chosen", "chosen_response", "selected", "pos", "assistant_response", "response"],
    "rejected":   ["rejected", "rejected_response", "neg", "negative"],
}


def _find_key(record: Dict, candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in record and record[candidate] not in (None, ""):
            return candidate
    if "sample" in record and isinstance(record["sample"], dict):
        nested = record["sample"]
        for candidate in candidates:
            if candidate in nested and nested[candidate] not in (None, ""):
                return f"sample.{candidate}"
    return None


def _get_nested(record: Dict, dotted_key: str):
    target = record
    for part in dotted_key.split("."):
        if isinstance(target, dict) and part in target:
            target = target[part]
        else:
            return None
    return target


def infer_field_mapping(jsonl_path: str, sample_lines: int = 20) -> Dict[str, Optional[str]]:
    """Inspect first N rows to infer key mapping to normalized fields."""
    found_example: Optional[Dict] = None
    with open(jsonl_path, "r", encoding="utf-8") as fp:
        for _ in range(sample_lines):
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            found_example = obj
            if len(obj.keys()) >= 4:
                break

    if not found_example:
        return {k: None for k in NORMALIZED_KEYS.keys()}

    mapping: Dict[str, Optional[str]] = {}
    for normalized, candidates in NORMALIZED_KEYS.items():
        mapping[normalized] = _find_key(found_example, candidates)
    return mapping


def normalize_record(record: Dict, mapping: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    def get_value(key):
        return _get_nested(record, key) if key else None

    return {
        "image_path": get_value(mapping.get("image_path")),
        "question":   get_value(mapping.get("question")),
        "answer":     get_value(mapping.get("answer")),
        "chosen":     get_value(mapping.get("chosen")),
        "rejected":   get_value(mapping.get("rejected")),
    }
