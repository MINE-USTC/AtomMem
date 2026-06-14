"""One-command LoCoMo evaluation runner for AtomMem.

This benchmark script is intentionally separate from AtomMem's generic core
pipeline. It contains LoCoMo-specific file discovery, category filtering, gold
evidence matching, and answer metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(REPO_ROOT / ".env")

import config  # noqa: E402
from openai import OpenAI  # noqa: E402


DEFAULT_CATEGORIES = {1, 2, 3, 4}
RETRIEVAL_K = 10
DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"
DEFAULT_JUDGE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHECKPOINT_VERSION = 1
TOKEN_STAT_KEYS = ("calls", "total_tokens", "prompt_tokens", "completion_tokens")


def env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value.strip() else default


def load_pipeline_components() -> Dict[str, Any]:
    """Import the full pipeline lazily so help/config parsing stays lightweight."""
    from run_atommem_pipeline import (  # noqa: WPS433
        AtomMemPipeline,
        configure_runtime_paths,
        print_token_report,
    )

    return {
        "pipeline_cls": AtomMemPipeline,
        "configure_runtime_paths": configure_runtime_paths,
        "print_token_report": print_token_report,
    }


def first_existing_path(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_data_dir() -> Path:
    return first_existing_path([
        REPO_ROOT / "data" / "split_samples",
        WORKSPACE_ROOT / "split_samples",
    ])


def default_facts_dir() -> Path:
    return first_existing_path([
        REPO_ROOT / "data" / "locomo_preextracted_facts",
        WORKSPACE_ROOT / "data" / "locomo_preextracted_facts",
    ])


def conv_id_from_name(value: str) -> str:
    match = re.search(r"conv[-_]?(\d+)", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer LoCoMo conversation id from: {value}")
    return f"conv-{int(match.group(1))}"


def tokenize(text: Any) -> List[str]:
    return [
        token
        for token in re.sub(r"[^\w\s]", " ", str(text).lower()).split()
        if token
    ]


def calculate_f1(expected: Any, generated: Any) -> float:
    expected_tokens = set(tokenize(expected))
    generated_tokens = set(tokenize(generated))
    if not expected_tokens or not generated_tokens:
        return 0.0
    common = expected_tokens & generated_tokens
    if not common:
        return 0.0
    precision = len(common) / len(generated_tokens)
    recall = len(common) / len(expected_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def calculate_bleu1(expected: Any, generated: Any) -> float:
    expected_tokens = tokenize(expected)
    generated_tokens = tokenize(generated)
    if not expected_tokens or not generated_tokens:
        return 0.0
    expected_counter = Counter(expected_tokens)
    generated_counter = Counter(generated_tokens)
    clipped = sum((generated_counter & expected_counter).values())
    precision = clipped / len(generated_tokens)
    bp = 1.0
    if len(generated_tokens) < len(expected_tokens):
        bp = math.exp(1 - len(expected_tokens) / len(generated_tokens))
    return bp * precision


def recall_at_k(gold_evidence: Sequence[str], retrieved_dia_ids: Sequence[str], k: int = RETRIEVAL_K) -> float:
    gold = [item for item in gold_evidence if item]
    if not gold:
        return 1.0
    top = set(retrieved_dia_ids[:k])
    return sum(1 for item in gold if item in top) / len(gold)


def standard_answer_for(qa: Dict[str, Any]) -> str:
    return str(qa.get("answer", qa.get("standard_answer", qa.get("adversarial_answer", ""))))


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return to_jsonable(value.dict())
    return str(value)


def usage_to_dict(usage: Any) -> Optional[Dict[str, Any]]:
    if usage is None:
        return None
    data = to_jsonable(usage)
    return data if isinstance(data, dict) else {"raw": data}


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.IGNORECASE | re.DOTALL)
    if code_block:
        candidates.append(code_block.group(1).strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_judge_label(text: str) -> str:
    parsed = extract_json_object(text)
    if parsed is not None:
        label = str(parsed.get("label", "")).strip().upper()
        if label in {"CORRECT", "WRONG"}:
            return label
    labels = set(re.findall(r"\b(CORRECT|WRONG)\b", text.upper()))
    if len(labels) == 1:
        return labels.pop()
    raise ValueError(f"Could not parse judge label from response: {text!r}")


def render_judge_prompt(prompt_template: str, question: str, gold_answer: str, generated_answer: str) -> str:
    return prompt_template.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )


def add_usage_to_stats(stats: Dict[str, Any], usage: Optional[Dict[str, Any]], count_call: bool = True) -> None:
    if count_call:
        stats["calls"] = stats.get("calls", 0) + 1
    if isinstance(usage, dict):
        stats["prompt_tokens"] = stats.get("prompt_tokens", 0) + int(usage.get("prompt_tokens", 0) or 0)
        stats["completion_tokens"] = stats.get("completion_tokens", 0) + int(usage.get("completion_tokens", 0) or 0)
        total = usage.get("total_tokens")
        if total is None:
            total = int(usage.get("prompt_tokens", 0) or 0) + int(usage.get("completion_tokens", 0) or 0)
        stats["total_tokens"] = stats.get("total_tokens", 0) + int(total or 0)
    stats["total_tokens_k"] = round(stats.get("total_tokens", 0) / 1000, 2)


def empty_token_stats() -> Dict[str, Any]:
    return {
        "calls": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens_k": 0.0,
    }


def normalize_token_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = empty_token_stats()
    if isinstance(stats, dict):
        for key in TOKEN_STAT_KEYS:
            normalized[key] = int(stats.get(key, 0) or 0)
    normalized["total_tokens_k"] = round(normalized["total_tokens"] / 1000, 2)
    return normalized


def add_token_stats_in_place(stats: Dict[str, Any], increment: Dict[str, Any]) -> None:
    for key in TOKEN_STAT_KEYS:
        stats[key] = int(stats.get(key, 0) or 0) + int(increment.get(key, 0) or 0)
    stats["total_tokens_k"] = round(stats.get("total_tokens", 0) / 1000, 2)


def token_stats_delta(current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
    delta = empty_token_stats()
    for key in TOKEN_STAT_KEYS:
        delta[key] = max(0, int(current.get(key, 0) or 0) - int(previous.get(key, 0) or 0))
    delta["total_tokens_k"] = round(delta["total_tokens"] / 1000, 2)
    return delta


class LLMJudge:
    def __init__(
        self,
        prompt_file: Path,
        api_key: str,
        api_base: str,
        model: str,
        max_retries: int,
        retry_sleep: float,
    ) -> None:
        if not api_key:
            raise ValueError("Missing judge API key. Set ATOMMEM_JUDGE_API_KEY or ATOMMEM_API_KEY.")
        self.prompt_template = prompt_file.read_text(encoding="utf-8")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    def judge(self, question: str, gold_answer: str, generated_answer: str) -> Dict[str, Any]:
        prompt = render_judge_prompt(self.prompt_template, question, gold_answer, generated_answer)
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                label = parse_judge_label(content)
                return {
                    "label": label,
                    "usage": usage_to_dict(getattr(response, "usage", None)),
                    "error": None,
                }
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    wait_seconds = self.retry_sleep * attempt
                    print(f"  Judge attempt {attempt}/{self.max_retries} failed: {last_error}; retrying in {wait_seconds:.1f}s")
                    time.sleep(wait_seconds)
        return {
            "label": "ERROR",
            "usage": None,
            "error": last_error or "Unknown judge error",
        }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_qa_items(data_file: Path, categories: Set[int]) -> List[Dict[str, Any]]:
    data = load_json(data_file)
    raw_items = data.get("qa", []) if isinstance(data, dict) else []
    items: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_items, 1):
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        if category not in categories:
            continue
        qa = dict(item)
        qa.setdefault("question_id", index)
        items.append(qa)
    return items


def discover_cases(
    facts_dir: Path,
    data_dir: Path,
    conv_ids: Optional[Set[str]] = None,
    categories: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    categories = categories or DEFAULT_CATEGORIES
    fact_files = sorted(facts_dir.glob("conv-*.json"))
    cases: List[Dict[str, Any]] = []
    for facts_file in fact_files:
        conv_id = conv_id_from_name(facts_file.name)
        if conv_ids and conv_id not in conv_ids:
            continue
        data_file = data_dir / f"{conv_id}.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Missing LoCoMo QA file for {conv_id}: {data_file}")
        qa_items = load_qa_items(data_file, categories)
        cases.append({
            "conv_id": conv_id,
            "facts_file": facts_file,
            "data_file": data_file,
            "qa_items": qa_items,
        })
    if not cases:
        raise FileNotFoundError(f"No conv-*.json files found under {facts_dir}")
    return cases


def memory_files(output_dir: Path, eval_conv_id: str) -> List[Path]:
    return [
        output_dir / f"facts_{eval_conv_id}.json",
        output_dir / f"events_{eval_conv_id}.json",
        output_dir / f"profiles_{eval_conv_id}.json",
        output_dir / f"entity_graph_{eval_conv_id}.json",
    ]


def memory_complete_file(output_dir: Path, eval_conv_id: str) -> Path:
    return output_dir / f"memory_complete_{eval_conv_id}.json"


def memory_storage_ready(output_dir: Path, eval_conv_id: str) -> bool:
    facts_file = output_dir / f"facts_{eval_conv_id}.json"
    events_file = output_dir / f"events_{eval_conv_id}.json"
    profiles_file = output_dir / f"profiles_{eval_conv_id}.json"
    if not (facts_file.exists() and events_file.exists() and profiles_file.exists()):
        return False
    try:
        facts_data = load_json(facts_file)
    except json.JSONDecodeError:
        return False
    facts = facts_data.get("facts", []) if isinstance(facts_data, dict) else facts_data
    return isinstance(facts, list) and bool(facts)


def last_source_dia_id_with_facts(facts_file: Path) -> Optional[str]:
    data = load_json(facts_file)
    results = data.get("results", []) if isinstance(data, dict) else []
    for item in reversed(results):
        if isinstance(item, dict) and item.get("facts") and item.get("dia_id"):
            return str(item["dia_id"])
    return None


def legacy_memory_covers_source(output_dir: Path, eval_conv_id: str, facts_file: Path) -> bool:
    expected_last_dia_id = last_source_dia_id_with_facts(facts_file)
    if not expected_last_dia_id:
        return False
    facts_path = output_dir / f"facts_{eval_conv_id}.json"
    try:
        facts_data = load_json(facts_path)
    except json.JSONDecodeError:
        return False
    stored_facts = facts_data.get("facts", []) if isinstance(facts_data, dict) else facts_data
    if not isinstance(stored_facts, list):
        return False
    return any(isinstance(fact, dict) and fact.get("dia_id") == expected_last_dia_id for fact in stored_facts)


def mark_memory_complete(output_dir: Path, eval_conv_id: str, facts_file: Path, source: str) -> Dict[str, Any]:
    marker = {
        "eval_conv_id": eval_conv_id,
        "facts_file": str(facts_file),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source,
    }
    save_json(memory_complete_file(output_dir, eval_conv_id), marker)
    return marker


def memory_ready(output_dir: Path, eval_conv_id: str, facts_file: Path) -> bool:
    if not memory_storage_ready(output_dir, eval_conv_id):
        return False
    marker_file = memory_complete_file(output_dir, eval_conv_id)
    if marker_file.exists():
        return True
    if legacy_memory_covers_source(output_dir, eval_conv_id, facts_file):
        mark_memory_complete(output_dir, eval_conv_id, facts_file, source="legacy")
        return True
    return False


def clear_memory(output_dir: Path, eval_conv_id: str) -> None:
    output_root = output_dir.resolve()
    for path in [*memory_files(output_dir, eval_conv_id), memory_complete_file(output_dir, eval_conv_id)]:
        if not path.exists():
            continue
        resolved = path.resolve()
        if not str(resolved).lower().startswith(str(output_root).lower()):
            raise RuntimeError(f"Refusing to remove outside output directory: {resolved}")
        resolved.unlink()


def question_key(conv_id: str, question_id: Any) -> str:
    return f"{conv_id}::{question_id}"


def record_key(record: Dict[str, Any]) -> str:
    return question_key(str(record.get("conv_id", "")), record.get("question_id"))


def dedupe_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict):
            by_key[record_key(record)] = record
    return list(by_key.values())


def build_settings(
    args: argparse.Namespace,
    data_dir: Path,
    facts_dir: Path,
    categories: Set[int],
    cases: Sequence[Dict[str, Any]],
    judge_model: Optional[str],
) -> Dict[str, Any]:
    return {
        "data_dir": str(data_dir),
        "facts_dir": str(facts_dir),
        "output_dir": str(Path(args.output_dir).resolve()),
        "categories": sorted(categories),
        "case_ids": [case["conv_id"] for case in cases],
        "retrieval_k": RETRIEVAL_K,
        "graph_enabled": True,
        "judge_enabled": not args.skip_judge,
        "judge_model": None if args.skip_judge else judge_model,
    }


def checkpoint_file(output_dir: Path) -> Path:
    return output_dir / "locomo_eval_checkpoint.json"


def checkpoint_backup_file(output_dir: Path) -> Path:
    return output_dir / "locomo_eval_checkpoint.json.bak"


def new_checkpoint(settings: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "settings": settings,
        "records": [],
        "token_usage": empty_token_stats(),
        "judge_token_usage": empty_token_stats(),
        "token_usage_by_conversation": {},
        "completed_memory": {},
        "completed": False,
        "updated_at": None,
    }


def normalize_checkpoint(data: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint = new_checkpoint(settings)
    checkpoint["records"] = dedupe_records(data.get("records", []))
    checkpoint["token_usage"] = normalize_token_stats(data.get("token_usage"))
    checkpoint["judge_token_usage"] = normalize_token_stats(data.get("judge_token_usage"))
    raw_by_conv = data.get("token_usage_by_conversation", {})
    if isinstance(raw_by_conv, dict):
        checkpoint["token_usage_by_conversation"] = {
            str(conv_id): normalize_token_stats(stats)
            for conv_id, stats in raw_by_conv.items()
            if isinstance(stats, dict)
        }
    raw_memory = data.get("completed_memory", {})
    checkpoint["completed_memory"] = raw_memory if isinstance(raw_memory, dict) else {}
    checkpoint["completed"] = bool(data.get("completed", False))
    checkpoint["updated_at"] = data.get("updated_at")
    return checkpoint


def checkpoint_from_results(results_payload: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint = new_checkpoint(settings)
    checkpoint["records"] = dedupe_records(results_payload.get("results", []))
    checkpoint["token_usage"] = normalize_token_stats(results_payload.get("token_usage"))
    checkpoint["judge_token_usage"] = normalize_token_stats(results_payload.get("judge_token_usage"))
    raw_by_conv = results_payload.get("token_usage_by_conversation", {})
    if isinstance(raw_by_conv, dict):
        checkpoint["token_usage_by_conversation"] = {
            str(conv_id): normalize_token_stats(stats)
            for conv_id, stats in raw_by_conv.items()
            if isinstance(stats, dict)
        }
    checkpoint["completed"] = True
    return checkpoint


def load_checkpoint(output_dir: Path, settings: Dict[str, Any]) -> Dict[str, Any]:
    for path in (checkpoint_file(output_dir), checkpoint_backup_file(output_dir)):
        if not path.exists():
            continue
        try:
            data = load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = None
        if isinstance(data, dict) and data.get("version") == CHECKPOINT_VERSION and data.get("settings") == settings:
            if path == checkpoint_backup_file(output_dir):
                print(f"Recovered checkpoint from backup: {path}")
            return normalize_checkpoint(data, settings)
        print(f"Ignoring incompatible checkpoint: {path}")

    results_path = output_dir / "locomo_eval_results.json"
    if results_path.exists():
        try:
            results_payload = load_json(results_path)
        except json.JSONDecodeError:
            results_payload = None
        if isinstance(results_payload, dict) and results_payload.get("settings") == settings:
            return checkpoint_from_results(results_payload, settings)

    return new_checkpoint(settings)


def save_checkpoint(output_dir: Path, checkpoint: Dict[str, Any]) -> None:
    checkpoint["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path = checkpoint_file(output_dir)
    backup_path = checkpoint_backup_file(output_dir)
    if path.exists():
        try:
            current = load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            current = None
        if isinstance(current, dict) and current.get("version") == CHECKPOINT_VERSION:
            shutil.copy2(path, backup_path)
    save_json(path, checkpoint)


def completed_question_keys(checkpoint: Dict[str, Any]) -> Set[str]:
    return {record_key(record) for record in checkpoint.get("records", []) if isinstance(record, dict)}


def add_pipeline_delta(
    memory_pipeline: Any,
    previous_stats: Dict[str, Any],
    aggregate_token_stats: Dict[str, Any],
    case_token_stats: Dict[str, Any],
) -> Dict[str, Any]:
    current_stats = normalize_token_stats(memory_pipeline.get_aggregate_llm_statistics())
    delta = token_stats_delta(current_stats, previous_stats)
    add_token_stats_in_place(aggregate_token_stats, delta)
    add_token_stats_in_place(case_token_stats, delta)
    return current_stats


def retrieved_dia_ids(response: Dict[str, Any]) -> List[str]:
    facts = response.get("retrieved_facts", [])
    return [
        fact.get("dia_id")
        for fact in facts
        if isinstance(fact, dict) and fact.get("dia_id")
    ]


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def average(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def summarize_subset(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        judged = [row for row in subset if row.get("llm_judge_label") in {"CORRECT", "WRONG"}]
        correct_judgments = sum(1 for row in judged if row.get("llm_judge_label") == "CORRECT")
        return {
            "questions": len(subset),
            "avg_f1": average([row["f1"] for row in subset]),
            "avg_bleu1": average([row["bleu1"] for row in subset]),
            "avg_recall_at_10": average([row["recall_at_10"] for row in subset]),
            "judge_accuracy": correct_judgments / len(judged) if judged else None,
        }

    by_category: Dict[str, Any] = {}
    for category in sorted({row["category"] for row in records}):
        subset = [row for row in records if row["category"] == category]
        by_category[str(category)] = summarize_subset(subset)

    by_conv: Dict[str, Any] = {}
    for conv_id in sorted({row["conv_id"] for row in records}):
        subset = [row for row in records if row["conv_id"] == conv_id]
        by_conv[conv_id] = summarize_subset(subset)

    return {
        "overall": summarize_subset(records),
        "by_category": by_category,
        "by_conversation": by_conv,
    }


def write_report(path: Path, summary: Dict[str, Any], token_stats: Dict[str, Any]) -> None:
    def fmt_j(value: Any) -> str:
        return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"

    lines = [
        "# LoCoMo Evaluation Report",
        "",
        "## Overall",
        "",
        "| Questions | F1 | BLEU-1 | Recall@10 | J |",
        "|---:|---:|---:|---:|---:|",
    ]
    overall = summary["overall"]
    lines.append(
        f"| {overall['questions']} | {overall['avg_f1']:.4f} | "
        f"{overall['avg_bleu1']:.4f} | {overall['avg_recall_at_10']:.4f} | "
        f"{fmt_j(overall['judge_accuracy'])} |"
    )
    lines.extend([
        "",
        "## By Category",
        "",
        "| Category | Questions | F1 | BLEU-1 | Recall@10 | J |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for category, stats in summary["by_category"].items():
        lines.append(
            f"| {category} | {stats['questions']} | {stats['avg_f1']:.4f} | "
            f"{stats['avg_bleu1']:.4f} | {stats['avg_recall_at_10']:.4f} | "
            f"{fmt_j(stats['judge_accuracy'])} |"
        )
    lines.extend([
        "",
        "## Token Usage",
        "",
        f"- Total LLM calls: {token_stats.get('calls', 0):,}",
        f"- Total tokens: {token_stats.get('total_tokens', 0):,}",
        f"- Prompt tokens: {token_stats.get('prompt_tokens', 0):,}",
        f"- Completion tokens: {token_stats.get('completion_tokens', 0):,}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_case(
    case: Dict[str, Any],
    args: argparse.Namespace,
    aggregate_token_stats: Dict[str, Any],
    pipeline: Dict[str, Any],
    judge: Optional[LLMJudge],
    judge_token_stats: Dict[str, Any],
    per_case_token_stats: Dict[str, Any],
    checkpoint: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    conv_id = case["conv_id"]
    eval_conv_id = f"{conv_id}-locomo-atommem"
    output_dir = Path(args.output_dir).resolve()

    completed_keys = completed_question_keys(checkpoint)
    remaining_items = [
        qa for qa in case["qa_items"]
        if question_key(conv_id, qa.get("question_id")) not in completed_keys
    ]
    case_token_stats = normalize_token_stats(per_case_token_stats.get(conv_id))
    per_case_token_stats[conv_id] = case_token_stats

    if not remaining_items:
        print(f"[{conv_id}] All {len(case['qa_items'])} QA items already checkpointed; skipping")
        return [], case_token_stats

    is_memory_ready = memory_ready(output_dir, eval_conv_id, case["facts_file"])
    if not args.reuse_memory or (args.reuse_memory and not is_memory_ready):
        clear_memory(output_dir, eval_conv_id)
        checkpoint.setdefault("completed_memory", {}).pop(eval_conv_id, None)

    memory_pipeline = pipeline["pipeline_cls"](
        conversation_id=eval_conv_id,
        output_dir=str(output_dir),
        profile_top_k=config.PROFILE_TEMPORAL_TOP_K,
        min_profile_llm_score=config.PROFILE_TEMPORAL_MIN_LLM_SCORE,
        enable_graph=True,
        final_top_k=RETRIEVAL_K,
    )
    last_pipeline_stats = empty_token_stats()

    if args.reuse_memory and is_memory_ready:
        print(f"[{conv_id}] Reusing existing memory: {eval_conv_id}")
        marker_file = memory_complete_file(output_dir, eval_conv_id)
        if marker_file.exists():
            checkpoint.setdefault("completed_memory", {})[eval_conv_id] = load_json(marker_file)
            save_checkpoint(output_dir, checkpoint)
    else:
        print(f"[{conv_id}] Building memory from {case['facts_file']}")
        _metadata, dialogue_results = memory_pipeline.load_preextracted_facts(str(case["facts_file"]))
        memory_pipeline.process_all_facts(dialogue_results)
        last_pipeline_stats = add_pipeline_delta(
            memory_pipeline,
            last_pipeline_stats,
            aggregate_token_stats,
            case_token_stats,
        )
        checkpoint.setdefault("completed_memory", {})[eval_conv_id] = mark_memory_complete(
            output_dir,
            eval_conv_id,
            case["facts_file"],
            source="build",
        )
        checkpoint["token_usage"] = aggregate_token_stats
        checkpoint["judge_token_usage"] = judge_token_stats
        checkpoint["token_usage_by_conversation"] = per_case_token_stats
        save_checkpoint(output_dir, checkpoint)
        print(f"[{conv_id}] Memory build complete")

    records: List[Dict[str, Any]] = []
    for index, qa in enumerate(case["qa_items"], 1):
        q_key = question_key(conv_id, qa.get("question_id"))
        if q_key in completed_keys:
            continue
        question = str(qa.get("question", ""))
        print(f"[{conv_id} Q{index}/{len(case['qa_items'])}] {question}")
        start = time.time()
        response = memory_pipeline.query_responder.answer_query(question, category=qa.get("category"))
        answer = response.get("answer", "")
        retrieved = retrieved_dia_ids(response)
        gold = [item for item in qa.get("evidence", []) or [] if item]
        expected = standard_answer_for(qa)
        record = {
            "conv_id": conv_id,
            "question_id": qa.get("question_id", index),
            "category": qa.get("category"),
            "question": question,
            "standard_answer": expected,
            "generated_answer": answer,
            "gold_evidence": gold,
            "retrieved_dia_ids_at_10": retrieved[:RETRIEVAL_K],
            "recall_at_10": recall_at_k(gold, retrieved, RETRIEVAL_K),
            "f1": calculate_f1(expected, answer),
            "bleu1": calculate_bleu1(expected, answer),
            "latency_seconds": time.time() - start,
        }
        if judge is None:
            record.update({
                "llm_judge_label": "SKIPPED",
            })
        else:
            judge_fields = judge.judge(question, expected, answer)
            record["llm_judge_label"] = judge_fields.get("label", "ERROR")
            add_usage_to_stats(judge_token_stats, judge_fields.get("usage"), count_call=True)
            add_usage_to_stats(aggregate_token_stats, judge_fields.get("usage"), count_call=True)
        records.append(record)
        checkpoint["records"].append(record)
        completed_keys.add(q_key)
        last_pipeline_stats = add_pipeline_delta(
            memory_pipeline,
            last_pipeline_stats,
            aggregate_token_stats,
            case_token_stats,
        )
        checkpoint["token_usage"] = aggregate_token_stats
        checkpoint["judge_token_usage"] = judge_token_stats
        checkpoint["token_usage_by_conversation"] = per_case_token_stats
        save_checkpoint(output_dir, checkpoint)
        judge_label = record.get("llm_judge_label", "SKIPPED")
        print(
            f"  F1={record['f1']:.4f} BLEU1={record['bleu1']:.4f} "
            f"R@10={record['recall_at_10']:.4f} J={judge_label}"
        )

    return records, case_token_stats


def parse_conv_ids(values: Optional[List[str]]) -> Optional[Set[str]]:
    if not values:
        return None
    conv_ids: Set[str] = set()
    for value in values:
        for part in value.split(","):
            if part.strip():
                conv_ids.add(conv_id_from_name(part.strip()))
    return conv_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AtomMem and evaluate LoCoMo QA in one command.")
    parser.add_argument("--data-dir", default=str(default_data_dir()), help="Directory containing conv-*.json LoCoMo files")
    parser.add_argument("--facts-dir", default=str(default_facts_dir()), help="Directory containing pre-extracted conv-*.json fact files")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs" / "locomo_eval"), help="Evaluation output directory")
    parser.add_argument("--conv-ids", nargs="*", default=None, help="Optional subset, e.g. conv-26 conv-49 or conv-26,conv-49")
    parser.add_argument("--categories", nargs="*", type=int, default=sorted(DEFAULT_CATEGORIES), help="Question categories to evaluate")
    parser.add_argument("--reuse-memory", action="store_true", help="Reuse existing memory files in --output-dir instead of rebuilding")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-a-Judge scoring")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    facts_dir = Path(args.facts_dir).resolve()
    categories = set(args.categories)
    conv_ids = parse_conv_ids(args.conv_ids)

    cases = discover_cases(
        facts_dir=facts_dir,
        data_dir=data_dir,
        conv_ids=conv_ids,
        categories=categories,
    )
    judge_model = env_or_default("ATOMMEM_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    settings = build_settings(args, data_dir, facts_dir, categories, cases, judge_model)

    print("=" * 72)
    print("AtomMem LoCoMo Evaluation")
    print("=" * 72)
    print(f"Data directory:      {data_dir}")
    print(f"Facts directory:     {facts_dir}")
    print(f"Output directory:    {output_dir}")
    print(f"Conversations:       {', '.join(case['conv_id'] for case in cases)}")
    print(f"Categories:          {sorted(categories)}")
    print()

    pipeline = load_pipeline_components()
    pipeline["configure_runtime_paths"](str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(output_dir, settings)
    checkpoint["settings"] = settings
    checkpoint["records"] = dedupe_records(checkpoint.get("records", []))
    if checkpoint["records"]:
        print(
            f"Resuming checkpoint: {len(checkpoint['records'])} QA records, "
            f"{checkpoint['token_usage'].get('total_tokens', 0):,} accumulated tokens"
        )
        print()

    total_start = time.time()
    judge: Optional[LLMJudge] = None
    if not args.skip_judge:
        judge = LLMJudge(
            prompt_file=REPO_ROOT / "prompts" / "judge_prompt.txt",
            api_key=env_or_default("ATOMMEM_JUDGE_API_KEY", config.API_KEY),
            api_base=env_or_default("ATOMMEM_JUDGE_API_BASE", DEFAULT_JUDGE_API_BASE),
            model=judge_model,
            max_retries=3,
            retry_sleep=2.0,
        )
    all_records: List[Dict[str, Any]] = checkpoint["records"]
    aggregate_token_stats: Dict[str, Any] = normalize_token_stats(checkpoint.get("token_usage"))
    judge_token_stats: Dict[str, Any] = normalize_token_stats(checkpoint.get("judge_token_usage"))
    per_case_token_stats: Dict[str, Any] = {
        str(conv_id): normalize_token_stats(stats)
        for conv_id, stats in checkpoint.get("token_usage_by_conversation", {}).items()
        if isinstance(stats, dict)
    }
    checkpoint["token_usage"] = aggregate_token_stats
    checkpoint["judge_token_usage"] = judge_token_stats
    checkpoint["token_usage_by_conversation"] = per_case_token_stats

    for case in cases:
        run_case(
            case,
            args,
            aggregate_token_stats,
            pipeline,
            judge,
            judge_token_stats,
            per_case_token_stats,
            checkpoint,
        )

    all_records = dedupe_records(checkpoint["records"])
    checkpoint["records"] = all_records
    summary = summarize_records(all_records)
    payload = {
        "settings": settings,
        "runtime_seconds": time.time() - total_start,
        "summary": summary,
        "token_usage": aggregate_token_stats,
        "judge_token_usage": judge_token_stats,
        "token_usage_by_conversation": per_case_token_stats,
        "results": all_records,
    }
    save_json(output_dir / "locomo_eval_results.json", payload)
    write_report(output_dir / "locomo_eval_report.md", summary, aggregate_token_stats)
    checkpoint["completed"] = True
    checkpoint["token_usage"] = aggregate_token_stats
    checkpoint["judge_token_usage"] = judge_token_stats
    checkpoint["token_usage_by_conversation"] = per_case_token_stats
    save_checkpoint(output_dir, checkpoint)

    print("\nEvaluation complete.")
    print(f"Results: {output_dir / 'locomo_eval_results.json'}")
    print(f"Report:  {output_dir / 'locomo_eval_report.md'}")
    pipeline["print_token_report"](aggregate_token_stats)


if __name__ == "__main__":
    main()
