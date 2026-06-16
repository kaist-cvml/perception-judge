import json
import os
import re
from typing import Dict, Iterator, List


class Logger:
    @staticmethod
    def section(title: str, total: int = None):
        suffix = f" (total {total})" if total is not None else ""
        print(f"\n{'-'*40}\n  {title}{suffix}\n{'-'*40}")

    @staticmethod
    def step(step_num: int, step_name: str, current: int = None, total: int = None):
        progress = f" [{current}/{total}]" if current and total else ""
        print(f"\n[Step {step_num}] {step_name}{progress}")

    @staticmethod
    def success(message: str, indent: int = 1):
        print(f"{'  ' * indent}✓ {message}")

    @staticmethod
    def fail(message: str, indent: int = 1):
        print(f"{'  ' * indent}✗ {message}")

    @staticmethod
    def info(message: str, indent: int = 1):
        print(f"{'  ' * indent}→ {message}")

    @staticmethod
    def result(message: str):
        print(f"\n{'-'*40}\n  {message}\n{'-'*40}\n")


def iter_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue


def write_jsonl(records: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_answer(answer: str) -> str:
    answer = str(answer).strip().lower()
    answer = re.sub(r'\\boxed\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\$+', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer.strip()
