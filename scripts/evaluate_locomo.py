"""One-command LoCoMo evaluation runner for AtomMem.

This benchmark script is intentionally separate from AtomMem's generic core
pipeline. It contains LoCoMo-specific file discovery, category filtering, gold
evidence matching, and answer metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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


def env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value.strip() else default


def load_pipeline_components() -> Dict[str, Any]:
    """Import the full pipeline lazily so help/config parsing stays lightweight."""
    from run_atommem_pipeline import (  # noqa: WPS433
        AtomMemPipelineTester,
        configure_runtime_paths,
        merge_llm_stats,
        print_token_report,
    )

    return {
        "tester_cls": AtomMemPipelineTester,
        "configure_runtime_paths": configure_runtime_paths,
        "merge_llm_stats": merge_llm_stats,
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
        REPO_ROOT / "data" / "preextracted_facts",
        WORKSPACE_ROOT / "data" / "preextracted_facts",
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


def merge_stats_in_place(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key in ("calls", "total_tokens", "prompt_tokens", "completion_tokens"):
        target[key] = target.get(key, 0) + source.get(key, 0)
    target["total_tokens_k"] = round(target.get("total_tokens", 0) / 1000, 2)


@contextlib.contextmanager
def suppress_pipeline_stdout() -> Iterable[None]:
    """Hide verbose memory-build logs while keeping evaluation progress concise."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


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
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def load_qa_items(data_file: Path, categories: Set[int], limit: Optional[int] = None) -> List[Dict[str, Any]]:
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
        if limit is not None and len(items) >= limit:
            break
    return items


def discover_cases(
    facts_dir: Path,
    data_dir: Path,
    conv_ids: Optional[Set[str]] = None,
    categories: Optional[Set[int]] = None,
    limit_per_conv: Optional[int] = None,
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
        qa_items = load_qa_items(data_file, categories, limit=limit_per_conv)
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


def memory_ready(output_dir: Path, eval_conv_id: str) -> bool:
    facts_file, events_file, profiles_file, _graph_file = memory_files(output_dir, eval_conv_id)
    if not (facts_file.exists() and events_file.exists() and profiles_file.exists()):
        return False
    try:
        facts_data = load_json(facts_file)
    except json.JSONDecodeError:
        return False
    facts = facts_data.get("facts", []) if isinstance(facts_data, dict) else facts_data
    return isinstance(facts, list) and bool(facts)


def clear_memory(output_dir: Path, eval_conv_id: str) -> None:
    output_root = output_dir.resolve()
    for path in memory_files(output_dir, eval_conv_id):
        if not path.exists():
            continue
        resolved = path.resolve()
        if not str(resolved).lower().startswith(str(output_root).lower()):
            raise RuntimeError(f"Refusing to remove outside output directory: {resolved}")
        resolved.unlink()


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
        total_hits = 0
        total_gold = 0
        for row in subset:
            gold = [item for item in row.get("gold_evidence", []) or [] if item]
            retrieved = set(row.get("retrieved_dia_ids_at_10", []) or [])
            total_gold += len(gold)
            total_hits += sum(1 for item in gold if item in retrieved)
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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    conv_id = case["conv_id"]
    eval_conv_id = f"{conv_id}-locomo-atommem"
    output_dir = Path(args.output_dir).resolve()

    if not args.reuse_memory:
        clear_memory(output_dir, eval_conv_id)

    with suppress_pipeline_stdout():
        tester = pipeline["tester_cls"](
            conversation_id=eval_conv_id,
            output_dir=str(output_dir),
            profile_top_k=args.profile_top_k,
            min_profile_llm_score=args.min_profile_llm_score,
            enable_graph=not args.disable_graph,
            final_top_k=RETRIEVAL_K,
            disable_evidence_summary=args.disable_evidence_summary,
        )

    if args.reuse_memory and memory_ready(output_dir, eval_conv_id):
        print(f"[{conv_id}] Reusing existing memory: {eval_conv_id}")
    else:
        print(f"[{conv_id}] Building memory from {case['facts_file']}")
        with suppress_pipeline_stdout():
            _metadata, dialogue_results = tester.load_preextracted_facts(str(case["facts_file"]))
            tester.process_all_facts(dialogue_results)
        print(f"[{conv_id}] Memory build complete")

    records: List[Dict[str, Any]] = []
    jsonl_path = output_dir / "locomo_eval_predictions.jsonl"
    for index, qa in enumerate(case["qa_items"], 1):
        question = str(qa.get("question", ""))
        print(f"[{conv_id} Q{index}/{len(case['qa_items'])}] {question}")
        start = time.time()
        response = tester.query_responder.answer_query(question)
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
        append_jsonl(jsonl_path, [record])
        judge_label = record.get("llm_judge_label", "SKIPPED")
        print(
            f"  F1={record['f1']:.4f} BLEU1={record['bleu1']:.4f} "
            f"R@10={record['recall_at_10']:.4f} J={judge_label}"
        )

    case_stats = tester.get_aggregate_llm_statistics()
    aggregate_token_stats.update(pipeline["merge_llm_stats"](aggregate_token_stats, case_stats))
    return records, case_stats


def parse_conv_ids(values: Optional[List[str]]) -> Optional[Set[str]]:
    if not values:
        return None
    conv_ids: Set[str] = set()
    for value in values:
        for part in value.split(","):
            if part.strip():
                conv_ids.add(conv_id_from_name(part.strip()))
    return conv_ids


def parse_categories(args: argparse.Namespace) -> Set[int]:
    if args.include_category5:
        return {1, 2, 3, 4, 5}
    return set(args.categories)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AtomMem and evaluate LoCoMo QA in one command.")
    parser.add_argument("--data-dir", default=str(default_data_dir()), help="Directory containing conv-*.json LoCoMo files")
    parser.add_argument("--facts-dir", default=str(default_facts_dir()), help="Directory containing pre-extracted conv-*.json fact files")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs" / "locomo_eval"), help="Evaluation output directory")
    parser.add_argument("--conv-ids", nargs="*", default=None, help="Optional subset, e.g. conv-26 conv-49 or conv-26,conv-49")
    parser.add_argument("--categories", nargs="*", type=int, default=sorted(DEFAULT_CATEGORIES), help="Question categories to evaluate")
    parser.add_argument("--include-category5", action="store_true", help="Also evaluate category 5 questions")
    parser.add_argument("--limit-per-conv", type=int, default=None, help="Debug limit for QA items per conversation")
    parser.add_argument("--reuse-memory", action="store_true", help="Reuse existing memory files in --output-dir instead of rebuilding")
    parser.add_argument("--disable-graph", action="store_true", help="Disable entity/event/turn graph reranking")
    parser.add_argument("--disable-evidence-summary", action="store_true", help="Answer directly with retrieved facts/profiles")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-a-Judge scoring")
    parser.add_argument("--judge-prompt-file", default=str(REPO_ROOT / "prompts" / "judge_prompt.txt"))
    parser.add_argument("--judge-model", default=env_or_default("ATOMMEM_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--judge-api-base", default=env_or_default("ATOMMEM_JUDGE_API_BASE", DEFAULT_JUDGE_API_BASE))
    parser.add_argument("--judge-api-key", default=env_or_default("ATOMMEM_JUDGE_API_KEY", config.API_KEY))
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--judge-retry-sleep", type=float, default=2.0)
    parser.add_argument("--profile-top-k", type=int, default=config.PROFILE_TEMPORAL_TOP_K)
    parser.add_argument("--min-profile-llm-score", type=float, default=config.PROFILE_TEMPORAL_MIN_LLM_SCORE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    facts_dir = Path(args.facts_dir).resolve()
    categories = parse_categories(args)
    conv_ids = parse_conv_ids(args.conv_ids)

    cases = discover_cases(
        facts_dir=facts_dir,
        data_dir=data_dir,
        conv_ids=conv_ids,
        categories=categories,
        limit_per_conv=args.limit_per_conv,
    )

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
    predictions_path = output_dir / "locomo_eval_predictions.jsonl"
    if predictions_path.exists() and not args.reuse_memory:
        predictions_path.unlink()

    total_start = time.time()
    judge: Optional[LLMJudge] = None
    if not args.skip_judge:
        judge = LLMJudge(
            prompt_file=Path(args.judge_prompt_file).resolve(),
            api_key=args.judge_api_key,
            api_base=args.judge_api_base,
            model=args.judge_model,
            max_retries=args.judge_max_retries,
            retry_sleep=args.judge_retry_sleep,
        )
    all_records: List[Dict[str, Any]] = []
    aggregate_token_stats: Dict[str, Any] = {
        "calls": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens_k": 0.0,
    }
    judge_token_stats: Dict[str, Any] = {
        "calls": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens_k": 0.0,
    }
    per_case_token_stats: Dict[str, Any] = {}

    for case in cases:
        records, token_stats = run_case(
            case,
            args,
            aggregate_token_stats,
            pipeline,
            judge,
            judge_token_stats,
        )
        all_records.extend(records)
        per_case_token_stats[case["conv_id"]] = token_stats

    summary = summarize_records(all_records)
    payload = {
        "settings": {
            "data_dir": str(data_dir),
            "facts_dir": str(facts_dir),
            "output_dir": str(output_dir),
            "categories": sorted(categories),
            "retrieval_k": RETRIEVAL_K,
            "graph_enabled": not args.disable_graph,
            "evidence_summary_enabled": not args.disable_evidence_summary,
            "judge_enabled": not args.skip_judge,
            "judge_model": None if args.skip_judge else args.judge_model,
        },
        "runtime_seconds": time.time() - total_start,
        "summary": summary,
        "token_usage": aggregate_token_stats,
        "judge_token_usage": judge_token_stats,
        "token_usage_by_conversation": per_case_token_stats,
        "results": all_records,
    }
    save_json(output_dir / "locomo_eval_results.json", payload)
    save_json(output_dir / "locomo_eval_summary.json", {
        "settings": payload["settings"],
        "runtime_seconds": payload["runtime_seconds"],
        "summary": summary,
        "token_usage": aggregate_token_stats,
        "judge_token_usage": judge_token_stats,
    })
    write_report(output_dir / "locomo_eval_report.md", summary, aggregate_token_stats)

    print("\nEvaluation complete.")
    print(f"Results: {output_dir / 'locomo_eval_results.json'}")
    print(f"Summary: {output_dir / 'locomo_eval_summary.json'}")
    print(f"Report:  {output_dir / 'locomo_eval_report.md'}")
    pipeline["print_token_report"](aggregate_token_stats)


if __name__ == "__main__":
    main()
