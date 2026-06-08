"""Public AtomMem end-to-end runner.

This script can:
1. optionally extract atomic facts from a raw split-sample conversation;
2. build AtomMem memories with metadata, fact-seeded events, and temporal profiles;
3. answer QA items with tuned entity/event/turn graph reranking.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


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


def _early_dry_run_if_requested() -> None:
    """Validate CLI paths for --dry-run before loading model-heavy modules."""
    if "--dry-run" not in sys.argv or "--help" in sys.argv or "-h" in sys.argv:
        return

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conv-id", required=True)
    parser.add_argument("--eval-conv-id", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--questions-file", default=None)
    parser.add_argument("--facts-file", default=None)
    parser.add_argument("--extract-facts", action="store_true")
    parser.add_argument("--output-dir", default="runs/atommem")
    parser.add_argument("--disable-graph", action="store_true")
    parser.add_argument("--disable-evidence-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, _unknown = parser.parse_known_args()

    eval_conv_id = args.eval_conv_id or f"{args.conv_id.strip() or 'conversation'}-atommem"
    facts_file = args.facts_file

    print("=" * 72)
    print("AtomMem Public Pipeline")
    print("=" * 72)
    print(f"Conversation:        {args.conv_id}")
    print(f"Storage/output ID:   {eval_conv_id}")
    print(f"Data file:           {args.data_file or '(not provided)'}")
    print(f"Questions file:      {args.questions_file or '(data file questions if present)'}")
    print(f"Facts file:          {facts_file or '(will extract)' if args.extract_facts else facts_file}")
    print(f"Output directory:    {Path(args.output_dir).resolve()}")
    print(f"Graph retrieval:     {'disabled' if args.disable_graph else 'enabled'}")
    print(f"Evidence summary:    {'disabled' if args.disable_evidence_summary else 'enabled'}")
    print()

    if args.data_file and not Path(args.data_file).exists():
        raise FileNotFoundError(args.data_file)
    if args.questions_file and not Path(args.questions_file).exists():
        raise FileNotFoundError(args.questions_file)
    if facts_file and not Path(facts_file).exists():
        raise FileNotFoundError(facts_file)
    print("Dry run OK. No LLM or embedding calls were made.")
    raise SystemExit(0)


_early_dry_run_if_requested()

from atommem_core.graph_rerank import RerankGraphConfig, SeedOnlyGraphReranker  # noqa: E402
from atommem_core.incremental_temporal_profile import (  # noqa: E402
    IncrementalTemporalProfilePipelineTester,
    IncrementalTemporalProfileQueryResponder,
)
from atommem_core.multichannel_graph import MultiChannelFactGraphIndex  # noqa: E402
from atommem_core.temporal_profile_version_chain import normalize_date_value  # noqa: E402
from openai import OpenAI  # noqa: E402
from src.llm_interface import LLMInterface  # noqa: E402
from src.retrieval import LayeredRetriever  # noqa: E402


class FactExecutor:
    """OpenAI-compatible atomic fact extractor for raw conversation files."""

    def __init__(self) -> None:
        if not config.FACT_EXECUTOR_API_KEY:
            raise ValueError(
                "Missing ATOMMEM_FACT_EXECUTOR_API_KEY/ATOMMEM_API_KEY. "
                "Set it in .env or pass a pre-extracted --facts-file."
            )
        self.client = OpenAI(
            api_key=config.FACT_EXECUTOR_API_KEY,
            base_url=config.FACT_EXECUTOR_API_BASE,
        )
        self.model = config.FACT_EXECUTOR_MODEL
        self.system_prompt = (
            "You extract high-value factual information from conversation turns. "
            "Return only a JSON array of complete standalone factual statements."
        )
        self.stats = {
            "calls": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        self.stats["calls"] += 1
        self.stats["total_tokens"] += total_tokens
        self.stats["prompt_tokens"] += prompt_tokens
        self.stats["completion_tokens"] += completion_tokens

    def extract_turn(
        self,
        current_dialogue: Dict[str, Any],
        previous_dialogue: Optional[Dict[str, Any]],
        session_time: str,
    ) -> List[str]:
        context = ""
        if previous_dialogue:
            context = (
                "\nPrevious dialogue:\n"
                f"[{previous_dialogue.get('dia_id')}] "
                f"{previous_dialogue.get('speaker')}: {previous_dialogue.get('text')}"
            )

        image_info = ""
        if current_dialogue.get("img_url"):
            image_parts = []
            if current_dialogue.get("blip_caption"):
                image_parts.append(f"Image description: {current_dialogue['blip_caption']}")
            if current_dialogue.get("query"):
                image_parts.append(f"Image keywords: {current_dialogue['query']}")
            if image_parts:
                image_info = "\n" + " ".join(image_parts)

        user_prompt = f"""Extract atomic facts from the current dialogue.

Requirements:
- Ignore greetings, pleasantries, acknowledgements, and low-value filler.
- Resolve pronouns and implicit references using the previous dialogue when needed.
- Keep each fact objective, complete, and standalone.
- Preserve specific time information when it can be inferred.
- Output a JSON array of strings.
{context}

Current dialogue:
[{current_dialogue.get('dia_id')}] {current_dialogue.get('speaker')}
Time: {session_time}
Text: "{current_dialogue.get('text', '')}{image_info}"
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        self._record_usage(response)
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```json"):
            content = content[7:].strip()
        if content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        try:
            facts = json.loads(content)
        except json.JSONDecodeError:
            return []
        if not isinstance(facts, list):
            return []
        return [str(fact).strip() for fact in facts if str(fact).strip()]


def session_number(session_key: str) -> int:
    try:
        return int(session_key.split("_")[1])
    except (IndexError, ValueError):
        return 0


def extract_facts_from_split_sample(
    data_file: str,
    output_file: str,
    start_session: int = 1,
    end_session: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    extractor = FactExecutor()
    data_path = Path(data_file)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    conversation = data.get("conversation", data)

    session_keys = [
        key for key in conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    ]
    session_keys = sorted(session_keys, key=session_number)
    session_keys = [
        key for key in session_keys
        if session_number(key) >= start_session
        and (end_session is None or session_number(key) <= end_session)
    ]

    results: List[Dict[str, Any]] = []
    total_dialogues = 0
    total_facts = 0
    for session_key in session_keys:
        dialogues = conversation.get(session_key, [])
        session_time = conversation.get(f"{session_key}_date_time", "")
        print(f"[fact-executor] {session_key}: {len(dialogues)} turns")
        for index, dialogue in enumerate(dialogues):
            previous = dialogues[index - 1] if index > 0 else None
            facts = extractor.extract_turn(dialogue, previous, session_time)
            total_dialogues += 1
            total_facts += len(facts)
            if not facts:
                continue
            results.append({
                "session": session_key,
                "session_number": session_number(session_key),
                "session_time": session_time,
                "dia_id": dialogue.get("dia_id"),
                "speaker": dialogue.get("speaker"),
                "text": dialogue.get("text", ""),
                "facts": facts,
            })

    payload = {
        "metadata": {
            "source_file": str(data_path),
            "total_sessions": len(session_keys),
            "session_range": f"session_{start_session}" + (f" to session_{end_session}" if end_session else " to end"),
            "total_dialogues": total_dialogues,
            "total_facts": total_facts,
            "dialogues_with_facts": len(results),
        },
        "results": results,
    }
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fact-executor] saved {total_facts} facts to {out_path}")
    extractor.stats["total_tokens_k"] = round(extractor.stats["total_tokens"] / 1000, 2)
    return str(out_path), extractor.stats


class AtomMemGraphQueryResponder(IncrementalTemporalProfileQueryResponder):
    """QA responder with no-graph seed retrieval followed by graph reranking."""

    def __init__(
        self,
        conversation_id: str,
        llm: Optional[LLMInterface] = None,
        enable_graph: bool = True,
        final_top_k: int = config.GRAPH_FINAL_TOP_K,
        disable_evidence_summary: bool = False,
    ) -> None:
        super().__init__(conversation_id, llm=llm)
        self.enable_graph = enable_graph
        self.final_top_k = final_top_k
        self.disable_evidence_summary = disable_evidence_summary
        self.base_retriever = LayeredRetriever()

    def answer_query(self, query: str) -> Dict[str, Any]:
        query_info, latency = self.prepare_query_info(query)
        return self.answer_query_with_query_info(query, query_info, shared_latency=latency)

    def prepare_query_info(self, query: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
        latency: Dict[str, float] = {}
        start = time.time()
        query_info = self._extract_query_intent(query)
        query_info["time"] = self._normalize_query_time_field(query_info.get("time"))
        latency["intent_extraction_ms"] = (time.time() - start) * 1000

        start = time.time()
        query_info["embedding"] = self.embedding_model.encode(query)
        latency["embedding_ms"] = (time.time() - start) * 1000
        return query_info, latency

    def answer_query_with_query_info(
        self,
        query: str,
        query_info: Dict[str, Any],
        shared_latency: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        latency = dict(shared_latency or {})
        run_query_info = copy.deepcopy(query_info)
        run_query_info["time"] = self._normalize_query_time_field(run_query_info.get("time"))

        if not run_query_info.get("embedding"):
            start = time.time()
            run_query_info["embedding"] = self.embedding_model.encode(query)
            latency["embedding_ms"] = (time.time() - start) * 1000

        all_facts = self.storage.load_facts()
        all_events = self.storage.load_events()
        all_profiles = self.storage.load_profiles()

        start = time.time()
        base_result = self.base_retriever.retrieve_for_query(
            run_query_info,
            copy.deepcopy(all_facts),
            copy.deepcopy(all_events),
            copy.deepcopy(all_profiles),
        )
        seed_facts = base_result.get("facts", [])[: config.GRAPH_MAX_SEED_FACTS]
        graph_debug: Dict[str, Any] = {
            "enabled": False,
            "reason": "disabled",
            "seed_fact_ids": [fact.get("fact_id") for fact in seed_facts if fact.get("fact_id")],
        }

        if self.enable_graph and seed_facts:
            graph_index = MultiChannelFactGraphIndex(all_facts, all_events, conv_id=self.conversation_id)
            graph_config = RerankGraphConfig(final_top_k=self.final_top_k)
            graph_reranker = SeedOnlyGraphReranker(graph_index, graph_config)
            final_facts, graph_debug = graph_reranker.retrieve_ranked_topk(
                run_query_info,
                seed_facts,
                all_facts,
                top_k=self.final_top_k,
            )
            if not final_facts:
                final_facts = seed_facts[: self.final_top_k]
        else:
            final_facts = seed_facts[: self.final_top_k]

        final_profiles = self._apply_profile_time_filter(
            base_result.get("profiles", []),
            run_query_info.get("time"),
        )
        event_contexts = self._build_event_contexts(final_facts, all_events)
        latency["retrieval_ms"] = (time.time() - start) * 1000

        start = time.time()
        if self.disable_evidence_summary:
            evidence_summary = self._fallback_evidence_summary(final_facts, final_profiles)
        else:
            evidence_summary = self._summarize_retrieved_evidence(
                query,
            run_query_info,
            final_facts,
            final_profiles,
        )
        evidence_summary["graph_debug"] = graph_debug
        latency["evidence_summary_ms"] = (time.time() - start) * 1000

        start = time.time()
        answer = self._generate_answer(
            query,
            final_facts,
            final_profiles,
            event_contexts,
            query_info=run_query_info,
            evidence_summary=evidence_summary,
        )
        latency["answer_generation_ms"] = (time.time() - start) * 1000

        return {
            "query": query,
            "answer": answer,
            "retrieved_evidence": [fact.get("dia_id") for fact in final_facts if fact.get("dia_id")],
            "retrieved_facts": final_facts,
            "retrieved_profiles": final_profiles,
            "event_contexts": event_contexts,
            "evidence_summary": evidence_summary,
            "graph_debug": graph_debug,
            "retrieval_rounds": 1,
            "no_useful_retrieval_rounds": 0,
            "stopped_for_insufficient_evidence": False,
            "query_info": {
                "need_specific": run_query_info.get("need_specific", True),
                "need_attribute": run_query_info.get("need_attribute", False),
                "people": run_query_info.get("people", []),
                "keywords": run_query_info.get("keywords", []),
                "time": run_query_info.get("time"),
            },
            "latency_breakdown": latency,
        }

    @staticmethod
    def _build_event_contexts(facts: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, str]:
        event_lookup = {event.get("event_id"): event for event in events if event.get("event_id")}
        contexts: Dict[str, str] = {}
        for fact in facts:
            summaries = []
            seen = set()
            for event_id in fact.get("event_ids", []) or []:
                summary = event_lookup.get(event_id, {}).get("summary", "")
                if summary and summary not in seen:
                    seen.add(summary)
                    summaries.append(summary)
            if summaries and fact.get("fact_id"):
                contexts[fact["fact_id"]] = " | ".join(summaries)
        return contexts

    @staticmethod
    def _normalize_query_time_field(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        raw_items = value if isinstance(value, list) else [value]
        normalized = []
        seen = set()
        for item in raw_items:
            date_value = normalize_date_value(item)
            if date_value and date_value not in seen:
                seen.add(date_value)
                normalized.append(date_value)
        return normalized or None


class AtomMemPipelineTester(IncrementalTemporalProfilePipelineTester):
    """Memory builder plus tuned entity/event/turn graph QA."""

    def __init__(
        self,
        conversation_id: str,
        output_dir: str,
        profile_top_k: int,
        min_profile_llm_score: float,
        enable_graph: bool,
        final_top_k: int,
        disable_evidence_summary: bool,
    ) -> None:
        super().__init__(
            conversation_id=conversation_id,
            output_dir=output_dir,
            profile_top_k=profile_top_k,
            min_profile_llm_score=min_profile_llm_score,
        )
        self.query_responder = AtomMemGraphQueryResponder(
            conversation_id=conversation_id,
            llm=self.llm,
            enable_graph=enable_graph,
            final_top_k=final_top_k,
            disable_evidence_summary=disable_evidence_summary,
        )
        mode = "entity/event/turn graph rerank" if enable_graph else "no graph"
        print(f"-> QA retrieval mode: {mode}; final_top_k={final_top_k}")


def configure_runtime_paths(output_dir: str) -> None:
    config.FACTS_DIR = str(Path(output_dir).resolve())
    config.PROMPTS_DIR = str((REPO_ROOT / "prompts").resolve())
    config.CONVERSATION_DIR = str((REPO_ROOT / "data").resolve())
    config.QA_DIR = config.CONVERSATION_DIR
    config.REPORTS_DIR = str((Path(output_dir).resolve() / "reports"))


def load_question_items(data_file: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(data_file).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    for key in ("questions", "qa", "queries"):
        items = data.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def extract_selected_evidence(
    evidence_summary: Dict[str, Any],
    retrieved_facts: List[Dict[str, Any]],
    retrieved_profiles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fact_ids = evidence_summary.get("useful_fact_ids", evidence_summary.get("supporting_fact_ids", [])) or []
    profile_ids = evidence_summary.get("useful_profile_ids", evidence_summary.get("supporting_profile_ids", [])) or []
    fact_lookup = {fact.get("fact_id"): fact for fact in retrieved_facts if fact.get("fact_id")}
    profile_lookup = {profile.get("profile_id"): profile for profile in retrieved_profiles if profile.get("profile_id")}
    return {
        "facts": [
            {
                "fact_id": fact_id,
                "fact": fact_lookup.get(fact_id, {}).get("fact", ""),
                "recorded_date": (
                    fact_lookup.get(fact_id, {}).get("time", ["", ""])[1]
                    if isinstance(fact_lookup.get(fact_id, {}).get("time"), list)
                    and len(fact_lookup.get(fact_id, {}).get("time", [])) > 1
                    else ""
                ),
            }
            for fact_id in fact_ids
            if fact_id in fact_lookup
        ],
        "profiles": [
            {
                "profile_id": profile_id,
                "content": profile_lookup.get(profile_id, {}).get("content", ""),
            }
            for profile_id in profile_ids
            if profile_id in profile_lookup
        ],
    }


def answer_question_items(
    tester: AtomMemPipelineTester,
    question_items: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    start = time.time()
    results: List[Dict[str, Any]] = []
    for index, item in enumerate(question_items, 1):
        question = item.get("question") or item.get("query")
        if not question:
            continue
        print(f"[Q{index}/{len(question_items)}] {question}")
        response = tester.query_responder.answer_query(str(question))
        retrieved_facts = response.get("retrieved_facts", [])
        retrieved_profiles = response.get("retrieved_profiles", [])
        evidence_summary = response.get("evidence_summary", {})
        source_metadata = {
            key: value
            for key, value in item.items()
            if key not in {"question", "query"}
        }
        results.append({
            "question_id": item.get("question_id", index),
            "question": question,
            "answer": response.get("answer", ""),
            "source_metadata": source_metadata,
            "query_info": response.get("query_info", {}),
            "retrieved_fact_ids": [fact.get("fact_id") for fact in retrieved_facts if fact.get("fact_id")],
            "retrieved_source_ids": [fact.get("dia_id") for fact in retrieved_facts if fact.get("dia_id")],
            "retrieved_profile_ids": [profile.get("profile_id") for profile in retrieved_profiles if profile.get("profile_id")],
            "selected_evidence": extract_selected_evidence(evidence_summary, retrieved_facts, retrieved_profiles),
            "evidence_summary": evidence_summary,
            "graph_debug": response.get("graph_debug", {}),
            "latency_breakdown": response.get("latency_breakdown", {}),
        })

    payload = {
        "total_questions": len(results),
        "runtime": time.time() - start,
        "results": results,
    }
    out_path = Path(output_dir) / f"qa_predictions_{tester.conversation_id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    jsonl_path = Path(output_dir) / f"qa_predictions_{tester.conversation_id}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved QA predictions to {out_path}")
    return payload


def derive_public_eval_conversation_id(source_conv_id: str, eval_conv_id: Optional[str]) -> str:
    if eval_conv_id:
        return eval_conv_id
    cleaned = (source_conv_id or "conversation").strip()
    return f"{cleaned}-atommem"


def print_token_report(stats: Dict[str, Any]) -> None:
    print("=" * 60)
    print("LLM CALL COUNTS AND TOKEN USAGE")
    print("=" * 60)
    print(f"\n  Total LLM calls:      {stats.get('calls', 0):>10,}")
    print(f"  Total tokens:         {stats.get('total_tokens', 0):>10,}  ({stats.get('total_tokens_k', 0):.1f}k)")
    print(f"    Prompt tokens:      {stats.get('prompt_tokens', 0):>10,}")
    print(f"    Completion tokens:  {stats.get('completion_tokens', 0):>10,}\n")


def merge_llm_stats(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(left)
    for key in ["calls", "total_tokens", "prompt_tokens", "completion_tokens"]:
        merged[key] = merged.get(key, 0) + right.get(key, 0)
    merged["total_tokens_k"] = round(merged.get("total_tokens", 0) / 1000, 2)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the public AtomMem pipeline.")
    parser.add_argument("--conv-id", required=True, help="Source conversation ID")
    parser.add_argument("--eval-conv-id", default=None, help="Storage/output ID; default adds AtomMem suffix")
    parser.add_argument("--data-file", default=None, help="Raw conversation JSON; may also contain optional questions")
    parser.add_argument("--questions-file", default=None, help="Optional JSON/JSONL-like question file with question/query fields")
    parser.add_argument("--facts-file", default=None, help="Pre-extracted facts JSON. If omitted, use --extract-facts.")
    parser.add_argument("--extract-facts", action="store_true", help="Extract facts from --data-file before memory build")
    parser.add_argument("--output-dir", default="runs/atommem", help="Output directory for generated memory and results")
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing memory files in --output-dir")
    parser.add_argument("--skip-qa", action="store_true", help="Build memory only")
    parser.add_argument("--disable-graph", action="store_true", help="Use no-graph seed retrieval for QA")
    parser.add_argument("--disable-evidence-summary", action="store_true", help="Send retrieved facts/profiles directly to answer generation")
    parser.add_argument("--final-top-k", type=int, default=config.GRAPH_FINAL_TOP_K, help="Number of final facts sent to answer generation")
    parser.add_argument("--profile-top-k", type=int, default=config.PROFILE_TEMPORAL_TOP_K)
    parser.add_argument("--min-profile-llm-score", type=float, default=config.PROFILE_TEMPORAL_MIN_LLM_SCORE)
    parser.add_argument("--start-session", type=str, default=None, help="Optional session label for memory build, e.g. session_3")
    parser.add_argument("--fact-start-session", type=int, default=1)
    parser.add_argument("--fact-end-session", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print planned actions without model calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime_paths(args.output_dir)

    data_file = args.data_file
    eval_conv_id = derive_public_eval_conversation_id(args.conv_id, args.eval_conv_id)
    facts_file = args.facts_file

    print("=" * 72)
    print("AtomMem Public Pipeline")
    print("=" * 72)
    print(f"Conversation:        {args.conv_id}")
    print(f"Storage/output ID:   {eval_conv_id}")
    print(f"Data file:           {data_file or '(not provided)'}")
    print(f"Questions file:      {args.questions_file or '(data file questions if present)'}")
    print(f"Facts file:          {facts_file or '(will extract)' if args.extract_facts else facts_file}")
    print(f"Output directory:    {Path(args.output_dir).resolve()}")
    print(f"Graph retrieval:     {'disabled' if args.disable_graph else 'enabled'}")
    print(f"Evidence summary:    {'disabled' if args.disable_evidence_summary else 'enabled'}")
    print()

    if args.dry_run:
        if data_file and not Path(data_file).exists():
            raise FileNotFoundError(data_file)
        if args.questions_file and not Path(args.questions_file).exists():
            raise FileNotFoundError(args.questions_file)
        if facts_file and not Path(facts_file).exists():
            raise FileNotFoundError(facts_file)
        print("Dry run OK. No LLM or embedding calls were made.")
        return

    if args.skip_build and args.skip_qa:
        raise ValueError("Cannot use --skip-build and --skip-qa together.")

    total_start = time.time()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fact_extraction_stats: Dict[str, Any] = {}

    if not args.skip_build:
        if not facts_file:
            if not args.extract_facts:
                raise ValueError("Provide --facts-file or pass --extract-facts with --data-file.")
            if not data_file:
                raise ValueError("--data-file is required when --extract-facts is enabled.")
            extracted_path = Path(args.output_dir) / f"preextracted_facts_{args.conv_id}.json"
            facts_file, fact_extraction_stats = extract_facts_from_split_sample(
                data_file,
                str(extracted_path),
                start_session=args.fact_start_session,
                end_session=args.fact_end_session,
            )

    tester = AtomMemPipelineTester(
        conversation_id=eval_conv_id,
        output_dir=args.output_dir,
        profile_top_k=args.profile_top_k,
        min_profile_llm_score=args.min_profile_llm_score,
        enable_graph=not args.disable_graph,
        final_top_k=args.final_top_k,
        disable_evidence_summary=args.disable_evidence_summary,
    )

    if not args.skip_build:
        _metadata, dialogue_results = tester.load_preextracted_facts(facts_file)
        tester.process_all_facts(dialogue_results, start_session=args.start_session)
    else:
        print("PHASE 1: skipped; using existing memory files.")
        tester.save_memory_snapshot()

    if not args.skip_qa:
        question_file = args.questions_file or data_file
        if not question_file:
            print("PHASE 2: skipped; no question file was provided.")
        else:
            question_items = load_question_items(question_file)
            if question_items:
                answer_question_items(tester, question_items, args.output_dir)
            else:
                print("PHASE 2: skipped; no question/query items found.")
    else:
        print("PHASE 2: skipped.")

    stats = tester.get_aggregate_llm_statistics()
    if fact_extraction_stats:
        stats = merge_llm_stats(stats, fact_extraction_stats)
    total_runtime = time.time() - total_start
    print(f"Total runtime: {total_runtime:.2f}s")
    print_token_report(stats)


if __name__ == "__main__":
    main()
