"""Local AtomMem chat demo server.

The demo is intentionally isolated from evaluation scripts. It provides an
online turn-by-turn experience:

1. answer the user's current message using existing memory;
2. asynchronously extract facts only from that user message;
3. process each extracted fact through storage, event attribution, and profile
   batch-trigger checks.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from openai import OpenAI


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
from atommem_core.graph_rerank import RerankGraphConfig, SeedOnlyGraphReranker  # noqa: E402
from atommem_core.multichannel_graph import MultiChannelFactGraphIndex  # noqa: E402
from scripts.run_atommem_pipeline import (  # noqa: E402
    AtomMemPipeline,
    configure_runtime_paths,
)
from src.retrieval import LayeredRetriever  # noqa: E402

try:  # noqa: WPS229
    import uvicorn
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover - friendly runtime hint
    raise SystemExit(
        "The demo requires FastAPI and Uvicorn. Install dependencies with:\n"
        "  pip install -r requirements.txt"
    ) from exc


DEFAULT_MEMORY_DIR = REPO_ROOT / "runs" / "demo_memory"
DEFAULT_STATIC_DIR = REPO_ROOT / "demo" / "static"
DEFAULT_ANSWER_PROMPT = REPO_ROOT / "prompts" / "demo_answer_prompt.txt"
SESSION_NAME = "session_1"


class DemoSettings(BaseModel):
    general_api_key: str = ""
    general_api_base: str = ""
    general_model: str = ""
    fact_api_key: str = ""
    fact_api_base: str = ""
    fact_model: str = ""
    answer_prompt: str = ""


class SessionRequest(BaseModel):
    settings: DemoSettings


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1)


class FlushRequest(BaseModel):
    session_id: str


@dataclass
class DemoDefaults:
    general_api_key: str
    general_api_base: str
    general_model: str
    fact_api_key: str
    fact_api_base: str
    fact_model: str
    answer_prompt: str
    output_dir: Path


class DemoFactExtractor:
    """OpenAI-compatible user-turn fact extractor."""

    def __init__(self, api_key: str, api_base: str, model: str) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            http_client=httpx.Client(timeout=httpx.Timeout(config.LLM_REQUEST_TIMEOUT, connect=10.0)),
        )
        self.model = model.strip() or self._discover_model()
        self.system_prompt = (
            "You are an AI assistant specialized in extracting high-value factual information from conversation messages."
        )
        self.user_prompt_template = """Extract high-value factual information from the given dialogue and rewrite it as objective third-person facts.

Requirements:
- Ignore greetings, pleasantries, acknowledgements, and other low-value content.
- Use dialogue context to infer implicit references, causality, or speaker intent when necessary.
- Resolve all pronouns and vague references into explicit entities.
- Infer specific time information when possible.
- Each fact must be complete and standalone.
- Output the result as a JSON array of strings.{context_text}

INFORMATION TO EXTRACT FROM Current Dialogue:
[{turn_id}] Speaker: {speaker}
Time: {time}
Dialogue: "{dialogue}"

Please extract all factual information from the current dialogue and output as a JSON array."""

    def _discover_model(self) -> str:
        models = self.client.models.list()
        if not models.data:
            raise ValueError("Fact extractor endpoint returned no models from models.list().")
        return models.data[0].id

    def extract(self, user_message: str, previous_user_message: str, session_time: str, turn_id: int) -> List[str]:
        context_text = ""
        if previous_user_message.strip():
            context_text = f"""

Context (previous dialogue):
[D{max(1, turn_id - 1)}] User: "{previous_user_message}"
"""
        user_prompt = self.user_prompt_template.format(
            context_text=context_text,
            turn_id=f"D{turn_id}",
            speaker="User",
            time=session_time,
            dialogue=user_message,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            top_p=0.8,
            temperature=0.3,
            max_tokens=2048,
            stream=False,
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = self._parse_json(content)
        if isinstance(parsed, list):
            raw_facts = parsed
        else:
            raw_facts = []
        if not isinstance(raw_facts, list):
            return []
        return [str(fact).strip() for fact in raw_facts if str(fact).strip()]

    @staticmethod
    def _parse_json(content: str) -> Any:
        if content.startswith("```json"):
            content = content[7:].strip()
        if content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("[")
            end = content.rfind("]")
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None


class DemoAnswerGenerator:
    """OpenAI-compatible assistant response generator."""

    def __init__(self, api_key: str, api_base: str, model: str) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            http_client=httpx.Client(timeout=httpx.Timeout(config.LLM_REQUEST_TIMEOUT, connect=10.0)),
        )
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return (response.choices[0].message.content or "").strip()


class OnlineAtomMemSession:
    """Single local demo chat session."""

    def __init__(self, session_id: str, settings: DemoSettings, output_dir: Path) -> None:
        self.session_id = session_id
        self.conversation_id = f"demo-{session_id}"
        self.settings = settings
        self.output_dir = output_dir
        self.messages: List[Dict[str, str]] = []
        self.user_turn_count = 0
        self.status = "idle"
        self.status_detail = "Ready"
        self.last_error = ""
        self.lock = threading.RLock()

        self.fact_extractor = DemoFactExtractor(
            api_key=settings.fact_api_key,
            api_base=settings.fact_api_base,
            model=settings.fact_model,
        )
        self.answer_generator = DemoAnswerGenerator(
            api_key=settings.general_api_key,
            api_base=settings.general_api_base,
            model=settings.general_model,
        )

        self._configure_general_llm(settings)
        self.memory_pipeline = AtomMemPipeline(
            conversation_id=self.conversation_id,
            output_dir=str(output_dir),
            profile_top_k=config.PROFILE_TEMPORAL_TOP_K,
            min_profile_llm_score=config.PROFILE_TEMPORAL_MIN_LLM_SCORE,
            enable_graph=True,
            final_top_k=config.GRAPH_FINAL_TOP_K,
        )
        self.retriever = LayeredRetriever()

    @staticmethod
    def _configure_general_llm(settings: DemoSettings) -> None:
        config.API_KEY = settings.general_api_key
        config.API_BASE = settings.general_api_base
        config.LLM_MODEL = settings.general_model

    def answer(self, user_message: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        user_message = user_message.strip()
        with self.lock:
            if self.status in {"generating_answer", "updating_memory"}:
                raise HTTPException(status_code=409, detail="This session is still processing the previous turn.")
            self.status = "generating_answer"
            self.status_detail = "Generating answer from current memory"
            self.last_error = ""
            previous_user_message = self._last_user_message()
            history = copy.deepcopy(self.messages[-10:])

        try:
            relevant_facts, relevant_profiles = self._retrieve_memory(user_message)
            answer = self._generate_answer(user_message, history, relevant_facts, relevant_profiles)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.status = "error"
                self.status_detail = "Answer generation failed"
                self.last_error = str(exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        with self.lock:
            self.user_turn_count += 1
            turn_id = self.user_turn_count
            self.messages.append({"role": "user", "content": user_message})
            self.messages.append({"role": "assistant", "content": answer})
            self.status = "updating_memory"
            self.status_detail = "Updating memory from the latest user message"

        background_tasks.add_task(self.update_memory_from_user_turn, turn_id, user_message, previous_user_message)
        return {
            "session_id": self.session_id,
            "assistant_message": answer,
        }

    def update_memory_from_user_turn(
        self,
        turn_id: int,
        user_message: str,
        previous_user_message: str,
    ) -> None:
        session_time = self._session_time()
        try:
            with self.lock:
                self.status_detail = "Extracting facts from user message"

            extracted_facts = self.fact_extractor.extract(user_message, previous_user_message, session_time, turn_id)
            existing_facts = list(self.memory_pipeline.fact_storage.storage.load_facts())
            previous_context = f"User: {previous_user_message}" if previous_user_message else ""

            for fact_index, fact_text in enumerate(extracted_facts, 1):
                with self.lock:
                    self.status_detail = f"Processing fact {fact_index}/{len(extracted_facts)}"

                fact_tuple = self.memory_pipeline.build_fact_tuple(
                    fact_text=fact_text,
                    dia_id=f"D{turn_id}",
                    session_time=session_time,
                    speaker="User",
                    previous_context=previous_context,
                    existing_facts=existing_facts,
                )
                result = self.memory_pipeline.fact_storage.process_new_fact(
                    fact_tuple,
                    use_batch_event_filter=True,
                )
                if result.get("action") != "IGNORE":
                    existing_facts.append(fact_tuple)

            with self.lock:
                self.status = "idle"
                self.status_detail = "Ready"
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.status = "error"
                self.status_detail = "Memory update failed"
                self.last_error = str(exc)

    def flush_profiles(self) -> Dict[str, Any]:
        with self.lock:
            self.status = "updating_memory"
            self.status_detail = "Flushing pending profile facts"
        try:
            self.memory_pipeline.fact_storage.force_profile_extraction()
            with self.lock:
                self.status = "idle"
                self.status_detail = "Ready"
            return self.snapshot()
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.status = "error"
                self.status_detail = "Profile flush failed"
                self.last_error = str(exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def snapshot(self) -> Dict[str, Any]:
        facts = self.memory_pipeline.fact_storage.storage.load_facts()
        events = self.memory_pipeline.fact_storage.storage.load_events()
        profiles = self.memory_pipeline.fact_storage.storage.load_profiles()
        with self.lock:
            status = self.status
            status_detail = self.status_detail
            last_error = self.last_error
            messages = copy.deepcopy(self.messages)
        pending_profiles = sum(1 for fact in facts if fact.get("needs_profile_extraction", False))
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "status": status,
            "status_detail": status_detail,
            "last_error": last_error,
            "messages": messages,
            "facts": facts,
            "events": events,
            "profiles": profiles,
            "pending_profile_facts": pending_profiles,
            "profile_batch_trigger": config.PROFILE_BATCH_TRIGGER,
        }

    def _retrieve_memory(self, user_message: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        all_facts = self.memory_pipeline.fact_storage.storage.load_facts()
        all_events = self.memory_pipeline.fact_storage.storage.load_events()
        all_profiles = self.memory_pipeline.fact_storage.storage.load_profiles()
        if not all_facts and not all_profiles:
            return [], []

        query_info, _latency = self.memory_pipeline.query_responder.prepare_query_info(user_message)
        base_result = self.retriever.retrieve_for_query(
            query_info,
            copy.deepcopy(all_facts),
            copy.deepcopy(all_events),
            copy.deepcopy(all_profiles),
        )
        seed_facts = base_result.get("facts", [])[: config.GRAPH_MAX_SEED_FACTS]
        if seed_facts:
            graph_index = MultiChannelFactGraphIndex(all_facts, all_events, conv_id=self.conversation_id)
            graph_config = RerankGraphConfig(final_top_k=config.GRAPH_FINAL_TOP_K)
            graph_reranker = SeedOnlyGraphReranker(graph_index, graph_config)
            final_facts, _graph_debug = graph_reranker.retrieve_ranked_topk(
                query_info,
                seed_facts,
                all_facts,
                top_k=config.GRAPH_FINAL_TOP_K,
            )
            if not final_facts:
                final_facts = seed_facts[: config.GRAPH_FINAL_TOP_K]
        else:
            final_facts = []

        final_profiles = self.memory_pipeline.query_responder._apply_profile_time_filter(
            base_result.get("profiles", []),
            query_info.get("time"),
        )
        return final_facts, final_profiles

    def _generate_answer(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        facts: List[Dict[str, Any]],
        profiles: List[Dict[str, Any]],
    ) -> str:
        history_text = self._format_history(history)
        facts_text = self._format_facts(facts)
        profiles_text = self._format_profiles(profiles)
        user_prompt = f"""Recent conversation:
{history_text}

Relevant memory profiles:
{profiles_text}

Relevant memory facts:
{facts_text}

Current user message:
{user_message}

Write the assistant's next reply.
"""
        system_prompt = self.settings.answer_prompt.strip()
        return self.answer_generator.generate(system_prompt, user_prompt)

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        if not history:
            return "(No previous turns in this demo session.)"
        rows = []
        for message in history:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            rows.append(f"{role}: {content}")
        return "\n".join(rows)

    @staticmethod
    def _format_facts(facts: List[Dict[str, Any]]) -> str:
        if not facts:
            return "(No relevant facts yet.)"
        rows = []
        for fact in facts:
            time_field = fact.get("time", [])
            recorded_date = time_field[1] if isinstance(time_field, list) and len(time_field) > 1 else ""
            rows.append(
                f"- {fact.get('fact_id')}: {fact.get('fact', '')}"
                f" | recorded_date={recorded_date or 'unknown'}"
            )
        return "\n".join(rows)

    @staticmethod
    def _format_profiles(profiles: List[Dict[str, Any]]) -> str:
        if not profiles:
            return "(No relevant profiles yet.)"
        rows = []
        for profile in profiles:
            valid_from = profile.get("valid_from", "")
            rows.append(
                f"- {profile.get('profile_id')}: {profile.get('content', '')}"
                f" | valid_from={valid_from or 'unknown'}"
            )
        return "\n".join(rows)

    def _last_user_message(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    @staticmethod
    def _session_time() -> str:
        now = datetime.now()
        return now.strftime("%I:%M %p on %d %B, %Y").lstrip("0")


class DemoServerState:
    def __init__(self, defaults: DemoDefaults) -> None:
        self.defaults = defaults
        self.sessions: Dict[str, OnlineAtomMemSession] = {}
        self.lock = threading.RLock()

    def create_session(self, settings: DemoSettings) -> OnlineAtomMemSession:
        normalized = self.normalize_settings(settings)
        self.validate_settings(normalized)
        session_id = uuid.uuid4().hex[:10]
        with self.lock:
            configure_runtime_paths(str(self.defaults.output_dir))
            session = OnlineAtomMemSession(session_id, normalized, self.defaults.output_dir)
            self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> OnlineAtomMemSession:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown demo session.")
        return session

    def normalize_settings(self, settings: DemoSettings) -> DemoSettings:
        answer_prompt = settings.answer_prompt.strip() or self.defaults.answer_prompt
        return DemoSettings(
            general_api_key=settings.general_api_key.strip(),
            general_api_base=settings.general_api_base.strip(),
            general_model=settings.general_model.strip(),
            fact_api_key=settings.fact_api_key.strip(),
            fact_api_base=settings.fact_api_base.strip(),
            fact_model=settings.fact_model.strip(),
            answer_prompt=answer_prompt,
        )

    @staticmethod
    def validate_settings(settings: DemoSettings) -> None:
        missing = [
            name
            for name, value in [
                ("general_api_key", settings.general_api_key),
                ("general_api_base", settings.general_api_base),
                ("general_model", settings.general_model),
                ("fact_api_key", settings.fact_api_key),
                ("fact_api_base", settings.fact_api_base),
            ]
            if not value
        ]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing settings: {', '.join(missing)}")


def create_app(defaults: DemoDefaults) -> FastAPI:
    state = DemoServerState(defaults)
    app = FastAPI(title="AtomMem Demo")
    app.mount("/static", StaticFiles(directory=str(DEFAULT_STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(DEFAULT_STATIC_DIR / "index.html")

    @app.get("/api/defaults")
    def get_defaults() -> Dict[str, Any]:
        return {
            "settings": {
                "general_api_key": defaults.general_api_key,
                "general_api_base": defaults.general_api_base,
                "general_model": defaults.general_model,
                "fact_api_key": defaults.fact_api_key,
                "fact_api_base": defaults.fact_api_base,
                "fact_model": defaults.fact_model,
                "answer_prompt": defaults.answer_prompt,
            },
            "profile_batch_trigger": config.PROFILE_BATCH_TRIGGER,
        }

    @app.post("/api/session")
    def create_session(request: SessionRequest) -> Dict[str, Any]:
        session = state.create_session(request.settings)
        return session.snapshot()

    @app.get("/api/memory/{session_id}")
    def get_memory(session_id: str) -> Dict[str, Any]:
        return state.get_session(session_id).snapshot()

    @app.post("/api/chat")
    def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        return state.get_session(request.session_id).answer(request.message, background_tasks)

    @app.post("/api/flush-profiles")
    def flush_profiles(request: FlushRequest) -> Dict[str, Any]:
        return state.get_session(request.session_id).flush_profiles()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local AtomMem chat demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default=str(DEFAULT_MEMORY_DIR))
    parser.add_argument("--general-api-key", default=config.API_KEY)
    parser.add_argument("--general-api-base", default=config.API_BASE)
    parser.add_argument("--general-model", default=config.LLM_MODEL)
    parser.add_argument("--fact-api-key", default=config.FACT_EXTRACTOR_API_KEY)
    parser.add_argument("--fact-api-base", default=config.FACT_EXTRACTOR_API_BASE)
    parser.add_argument("--fact-model", default=config.FACT_EXTRACTOR_MODEL)
    parser.add_argument("--answer-prompt", default=str(DEFAULT_ANSWER_PROMPT))
    return parser.parse_args()


def load_answer_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.is_absolute():
        prompt_path = (REPO_ROOT / prompt_path).resolve()
    return prompt_path.read_text(encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime_paths(str(output_dir))

    defaults = DemoDefaults(
        general_api_key=args.general_api_key or "",
        general_api_base=args.general_api_base or "",
        general_model=args.general_model or "",
        fact_api_key=args.fact_api_key or "",
        fact_api_base=args.fact_api_base or "",
        fact_model=args.fact_model or "",
        answer_prompt=load_answer_prompt(args.answer_prompt),
        output_dir=output_dir,
    )
    app = create_app(defaults)
    print(f"AtomMem demo running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
