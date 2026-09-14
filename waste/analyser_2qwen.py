from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import local
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from ollama import Client


# ============================================================
# CONFIG
# ============================================================

TOWER_IP = "192.168.100.100"
OLLAMA_BASE = f"http://{TOWER_IP}:11434"

CHROMA_PATH = "./chroma_db"
KNOWLEDGE_COLLECTION = "shadow_ai_knowledge"

EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
ANALYSIS_MODEL = "qwen2.5:14b"

DEFAULT_OUTPUT_FILE = "findings.jsonl"
DEFAULT_REVIEW_FILE = "needs_review.jsonl"

MAX_EVENTS_PER_CHUNK = 40
OVERLAP_EVENTS = 4
SESSION_GAP_SECONDS = 300
DEFAULT_QWEN_WORKERS = 2

# High-recall generic indicators. This is deliberately NOT an
# application/domain allow-list.
AI_PATTERNS = (
    r"\bgenai\b",
    r"\bgenerative[\s_-]?ai\b",
    r"\bllm\b",
    r"\blarge[\s_-]?language[\s_-]?model\b",
    r"\bchatbot\b",
    r"\bai[\s_-]?assistant\b",
    r"\bvirtual[\s_-]?assistant\b",
    r"\bconversational[\s_-]?ai\b",
    r"\bai[\s_-]?chat\b",
    r"\bchat[\s_-]?completion\b",
    r"/chat/completions?\b",
    r"/v1/(chat|messages|responses|completions?)\b",
    r"/api/(chat|converse|completion|generate)\b",
    r"\bgenerativelanguage\b",
    r"\banthropic[\s._-]?claude\b",
    r"\bbedrock\b",
    r"\bsagemaker\b",
    r"\bopenai\b",
    r"\bchatgpt\b",
    r"\bclaude\b",
    r"\bgemini\b",
    r"\bperplexity\b",
    r"\bcopilot\b",
)

AWS_AI_SOURCES = (
    "bedrock",
    "sagemaker",
    "comprehend",
    "rekognition",
    "textract",
    "polly",
    "lex.amazonaws",
    "personalize",
    "kendra",
    "forecast",
    "transcribe",
)

OBVIOUS_SCORE = 6
UNCERTAIN_SCORE = 2

_thread_local = local()
_policy_cache: dict[str, str] = {}
_policy_lock = threading.Lock()


# ============================================================
# OLLAMA / CHROMA
# ============================================================

def get_ollama_client() -> Client:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = Client(host=OLLAMA_BASE)
        _thread_local.client = client
    return client


def get_knowledge_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_function = OllamaEmbeddingFunction(
        url=f"{OLLAMA_BASE}/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )
    return client.get_or_create_collection(
        name=KNOWLEDGE_COLLECTION,
        embedding_function=embedding_function,
    )


def retrieve_policy(applications: list[str], query_text: str) -> str:
    key = "|".join(sorted(set(applications))) or "__generic__"

    with _policy_lock:
        if key in _policy_cache:
            return _policy_cache[key]

    try:
        collection = get_knowledge_collection()
        if collection.count() == 0:
            return ""

        query = (
            "Shadow AI security policy. Determine whether observed activity is "
            "approved or prohibited, data classification restrictions, DLP rules, "
            "severity criteria, and required response. "
            + query_text[:2500]
        )

        result = collection.query(
            query_texts=[query],
            n_results=min(8, collection.count()),
        )

        docs = result.get("documents") or [[]]
        passages = docs[0] if docs else []

        text = "\n\n".join(
            str(x).strip() for x in passages if str(x).strip()
        )

        with _policy_lock:
            _policy_cache[key] = text

        return text
    except Exception as exc:
        print(f"[WARN] RAG lookup failed: {exc}")
        return ""


# ============================================================
# NORMALIZATION
# ============================================================

def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return clean(value)
    return ""


def parse_epoch(value: str) -> float:
    value = clean(value)
    if not value:
        return 0.0

    try:
        return float(value)
    except ValueError:
        pass

    try:
        text = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def extract_json(value: str) -> dict[str, Any] | None:
    value = value.strip()
    if not value:
        return None

    try:
        obj = json.loads(value)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(value[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def flatten_strings(obj: Any, prefix: str = "", limit: int = 80) -> list[str]:
    out: list[str] = []

    if len(out) >= limit:
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            if len(out) >= limit:
                break
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten_strings(v, p, limit - len(out)))

    elif isinstance(obj, list):
        for i, v in enumerate(obj[:30]):
            if len(out) >= limit:
                break
            out.extend(flatten_strings(v, f"{prefix}[{i}]", limit - len(out)))

    elif obj not in (None, ""):
        out.append(f"{prefix}={clean(obj)}")

    return out


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    timestamp = first(row, "timestamp", "@timestamp", "time", "date")
    sensor = first(row, "sensor", "device", "agent", "host")
    source = first(row, "source", "log_source", "type", "event_type")
    event = first(row, "event", "message", "raw", "data")

    parsed = extract_json(event)

    if parsed:
        flat = flatten_strings(parsed)
        searchable = " ".join(flat)

        event_source = clean(parsed.get("eventSource"))
        user = (
            clean(parsed.get("userName"))
            or clean(parsed.get("userIdentity", {}).get("userName"))
            if isinstance(parsed.get("userIdentity"), dict)
            else clean(parsed.get("userName"))
        )

        domain = (
            clean(parsed.get("destinationHostname"))
            or clean(parsed.get("destination", {}).get("hostname"))
            if isinstance(parsed.get("destination"), dict)
            else clean(parsed.get("destinationHostname"))
        )

        hostname = (
            clean(parsed.get("host", {}).get("hostname"))
            if isinstance(parsed.get("host"), dict)
            else ""
        )

        if event_source:
            searchable += " " + event_source

        source_lower = event_source.lower()
    else:
        searchable = event
        user = first(row, "user", "username", "user_name")
        domain = first(
            row,
            "domain",
            "hostname",
            "dstname",
            "destinationHostname",
            "destination.hostname",
        )
        hostname = first(row, "hostname", "host")
        source_lower = source.lower()

    sent = first(row, "sent", "bytes_sent", "bytes_out", "tx_bytes")
    received = first(row, "rcvd", "bytes_received", "bytes_in", "rx_bytes")

    try:
        total_bytes = int(float(sent or 0)) + int(float(received or 0))
    except Exception:
        total_bytes = 0

    combined = " ".join(
        [
            searchable,
            domain,
            source,
            sensor,
        ]
    ).lower()

    hints = []
    for pattern in AI_PATTERNS:
        try:
            if re.search(pattern, combined, flags=re.IGNORECASE):
                hints.append(pattern)
        except re.error:
            pass

    aws_ai = any(x in source_lower or x in combined for x in AWS_AI_SOURCES)

    # Generic score: high recall, no giant vendor allow-list.
    score = 0
    if domain:
        score += 1
    if hints:
        score += min(5, len(hints))
    if aws_ai:
        score += 5
    if re.search(r"/(api|v1)/", combined):
        score += 1
    if re.search(r"\b(prompt|completion|embedding|inference|model_id|modelid)\b", combined):
        score += 2
    if re.search(r"\b(chat|conversation|assistant|message)\b", combined):
        score += 1

    if aws_ai:
        tier = "obvious"
    elif score >= OBVIOUS_SCORE:
        tier = "obvious"
    elif score >= UNCERTAIN_SCORE:
        tier = "uncertain"
    else:
        tier = "irrelevant"

    return {
        "timestamp": timestamp,
        "ts_epoch": parse_epoch(timestamp),
        "sensor": sensor,
        "source": source,
        "user": user,
        "hostname": hostname,
        "domain": domain,
        "bytes": total_bytes,
        "event": event[:12000],
        "ai_score": score,
        "match_tier": tier,
        "ai_hints": hints,
        "aws_ai_event": aws_ai,
    }


# ============================================================
# CHUNKING
# ============================================================

@dataclass
class EntityState:
    key: str
    events: list[dict[str, Any]] = field(default_factory=list)
    last_ts: float = 0.0


def entity_key(row: dict[str, Any]) -> str:
    user = row.get("user") or ""
    host = row.get("hostname") or ""
    sensor = row.get("sensor") or ""
    return f"{sensor}|{user}|{host}"


def build_chunk(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {}

    applications = sorted(
        {
            str(e.get("domain"))
            for e in events
            if e.get("domain")
        }
    )

    hints = sorted(
        {
            str(h)
            for e in events
            for h in e.get("ai_hints", [])
        }
    )

    tiers = [e.get("match_tier") for e in events]
    tier = "obvious" if "obvious" in tiers else "uncertain"

    total_bytes = sum(int(e.get("bytes") or 0) for e in events)

    start_ts = min(float(e.get("ts_epoch") or 0) for e in events)
    end_ts = max(float(e.get("ts_epoch") or 0) for e in events)

    return {
        "sensor": events[0].get("sensor", ""),
        "source": events[0].get("source", ""),
        "start_ts": events[0].get("timestamp", ""),
        "end_ts": events[-1].get("timestamp", ""),
        "start_epoch": start_ts,
        "end_epoch": end_ts,
        "event_count": len(events),
        "match_tier": tier,
        "applications_seen": applications[:50],
        "text_pattern_hints": hints[:50],
        "total_bytes": total_bytes,
        "events": events,
    }


def build_chunks(rows):
    states: dict[str, EntityState] = {}

    for row in rows:
        key = entity_key(row)
        ts = float(row.get("ts_epoch") or 0)

        state = states.get(key)

        if state is None:
            state = EntityState(key=key)
            states[key] = state

        if (
            state.events
            and ts
            and state.last_ts
            and ts - state.last_ts > SESSION_GAP_SECONDS
        ):
            candidates = [
                e for e in state.events
                if e.get("match_tier") != "irrelevant"
            ]
            if candidates:
                yield build_chunk(candidates)

            state.events.clear()

        state.events.append(row)
        state.last_ts = ts

        if len(state.events) >= MAX_EVENTS_PER_CHUNK:
            candidates = [
                e for e in state.events
                if e.get("match_tier") != "irrelevant"
            ]

            if candidates:
                yield build_chunk(candidates)

            # Preserve a small overlap so activity split across chunks
            # does not lose context.
            state.events = state.events[-OVERLAP_EVENTS:]


    for state in states.values():
        candidates = [
            e for e in state.events
            if e.get("match_tier") != "irrelevant"
        ]
        if candidates:
            yield build_chunk(candidates)


# ============================================================
# LIGHTWEIGHT FILTER
# ============================================================

def lightweight_filter(row: dict[str, Any]) -> bool:
    """
    Conservative gate.

    Only chunks with ZERO AI-like evidence are discarded.
    Anything uncertain or obvious proceeds to Qwen.
    """
    return row.get("match_tier") != "irrelevant"


# ============================================================
# QWEN
# ============================================================

SYSTEM_PROMPT = r"""
You are the final Shadow AI security decision engine.

Your job is HIGH RECALL. The Python pre-filter is intentionally conservative,
so do not assume that an unfamiliar domain, hostname, SaaS service, API,
cloud service, browser activity, or application is non-AI merely because it
is not in a known vendor list.

Analyze ONLY the supplied events and policy.

Classify the chunk as:
- shadow_ai: evidence supports external conversational AI, generative AI,
  LLM usage, AI assistant usage, AI inference/model API usage, or an AI
  service being accessed.
- not_shadow_ai: evidence is sufficiently strong that the observed activity
  is unrelated to AI.
- needs_review: evidence is ambiguous and a human should inspect it.

Important:
1. Unknown AI services can be Shadow AI.
2. Do not require a known vendor name.
3. API paths, model/inference terminology, conversation/completion patterns,
   AWS AI services, AI-specific event fields, and surrounding context can
   establish evidence.
4. Ordinary customer-support chat, website live-chat widgets, crawlers,
   monitoring, and generic web browsing are NOT automatically Shadow AI.
5. Do not invent facts.
6. Use the policy only to determine approval, severity, and action.
7. Return ONLY valid JSON.

Required JSON:
{
  "classification": "shadow_ai|not_shadow_ai|needs_review",
  "risk": "low|medium|high|critical",
  "confidence": 0,
  "application": "",
  "user": "",
  "reason": "",
  "evidence": [],
  "policy": "",
  "severity": "low|medium|high|critical",
  "recommended_action": ""
}
"""


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": event.get("timestamp"),
        "user": event.get("user"),
        "hostname": event.get("hostname"),
        "domain": event.get("domain"),
        "source": event.get("source"),
        "sensor": event.get("sensor"),
        "bytes": event.get("bytes"),
        "ai_score": event.get("ai_score"),
        "ai_hints": event.get("ai_hints"),
        "aws_ai_event": event.get("aws_ai_event"),
        "event": event.get("event", "")[:5000],
    }


def analyze_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    started = time.time()

    applications = chunk.get("applications_seen", [])
    query_text = json.dumps(
        {
            "tier": chunk.get("match_tier"),
            "applications": applications,
            "hints": chunk.get("text_pattern_hints"),
            "events": [
                compact_event(e)
                for e in chunk.get("events", [])
            ],
        },
        ensure_ascii=False,
    )

    policy = retrieve_policy(applications, query_text)

    payload = {
        "chunk": {
            "sensor": chunk.get("sensor"),
            "source": chunk.get("source"),
            "start_ts": chunk.get("start_ts"),
            "end_ts": chunk.get("end_ts"),
            "event_count": chunk.get("event_count"),
            "match_tier": chunk.get("match_tier"),
            "applications_seen": applications,
            "text_pattern_hints": chunk.get("text_pattern_hints"),
            "total_bytes": chunk.get("total_bytes"),
            "events": [
                compact_event(e)
                for e in chunk.get("events", [])
            ],
        },
        "retrieved_policy": policy,
    }

    prompt = json.dumps(payload, ensure_ascii=False)

    client = get_ollama_client()

    try:
        response = client.chat(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={
                "temperature": 0,
            },
        )

        content = response["message"]["content"]

        # Qwen sometimes emits thinking tags despite JSON mode.
        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()

        result = json.loads(content)

        if not isinstance(result, dict):
            raise ValueError("Qwen returned non-object JSON")

    except Exception as exc:
        result = {
            "classification": "needs_review",
            "risk": "high",
            "confidence": 0,
            "application": "",
            "user": "",
            "reason": f"Qwen analysis failed: {exc}",
            "evidence": [],
            "policy": policy,
            "severity": "high",
            "recommended_action": "Manual review required.",
        }

    result["_analysis_seconds"] = round(time.time() - started, 2)
    result["_sensor"] = chunk.get("sensor")
    result["_source"] = chunk.get("source")
    result["_start_ts"] = chunk.get("start_ts")
    result["_end_ts"] = chunk.get("end_ts")
    result["_applications"] = applications
    result["_match_tier"] = chunk.get("match_tier")
    result["_bytes"] = chunk.get("total_bytes")
    result["_event_count"] = chunk.get("event_count")

    return result


# ============================================================
# OUTPUT
# ============================================================

def write_jsonl(handle, obj: dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    handle.flush()


# ============================================================
# STREAMING PROCESSOR
# ============================================================

def process_file(
    csv_path: Path,
    output_path: Path,
    review_path: Path,
    workers: int,
    progress_every: int,
) -> None:

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    pending = set()
    submitted = 0
    completed = 0
    rows_seen = 0

    def drain(done):
        nonlocal completed

        for future in done:
            try:
                result = future.result()

                classification = str(
                    result.get("classification", "needs_review")
                ).lower()

                if classification == "shadow_ai":
                    write_jsonl(out_handle, result)
                elif classification == "needs_review":
                    write_jsonl(review_handle, result)

                completed += 1

                if completed % 10 == 0:
                    print(
                        f"\rQwen completed: {completed} | "
                        f"submitted: {submitted} | rows: {rows_seen}",
                        end="",
                        flush=True,
                    )

            except Exception as exc:
                write_jsonl(
                    review_handle,
                    {
                        "classification": "needs_review",
                        "reason": f"worker failure: {exc}",
                    },
                )
                completed += 1

    with (
        csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file,
        output_path.open("w", encoding="utf-8") as out_handle,
        review_path.open("w", encoding="utf-8") as review_handle,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):

        reader = csv.DictReader(csv_file)

        def submit(chunk):
            nonlocal submitted

            if not chunk:
                return

            if not lightweight_filter(chunk):
                return

            future = executor.submit(analyze_chunk, chunk)
            pending.add(future)
            submitted += 1

        local_buffer: list[dict[str, Any]] = []

        for raw in reader:
            rows_seen += 1

            normalized = normalize_row(raw)

            # We do NOT call Qwen here. Chunk first so Qwen receives
            # surrounding context.
            local_buffer.append(normalized)

            if len(local_buffer) >= MAX_EVENTS_PER_CHUNK * 2:
                for chunk in build_chunks(local_buffer):
                    submit(chunk)

                local_buffer.clear()

            if len(pending) >= max(workers * 2, 2):
                done, not_done = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                pending = not_done
                drain(done)

            if progress_every and rows_seen % progress_every == 0:
                print(
                    f"\nRows read: {rows_seen} | "
                    f"Qwen submitted: {submitted} | "
                    f"completed: {completed}"
                )

        for chunk in build_chunks(local_buffer):
            submit(chunk)

        local_buffer.clear()

        while pending:
            done, not_done = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )
            pending = not_done
            drain(done)

    print()
    print(f"Rows read:       {rows_seen}")
    print(f"Qwen submitted:  {submitted}")
    print(f"Qwen completed:  {completed}")
    print(f"Findings:        {output_path}")
    print(f"Needs review:    {review_path}")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-recall streaming Shadow AI analyzer"
    )

    parser.add_argument(
        "csv",
        type=Path,
        help="CSV containing timestamp,sensor,source,event",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_QWEN_WORKERS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
    )

    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path(DEFAULT_REVIEW_FILE),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    print("=" * 70)
    print("SHADOW AI ANALYZER")
    print("=" * 70)
    print(f"Input:       {args.csv}")
    print(f"Qwen:        {ANALYSIS_MODEL}")
    print(f"Ollama:      {OLLAMA_BASE}")
    print(f"Workers:     {args.workers}")
    print(f"Output:      {args.output}")
    print(f"Review:      {args.review_output}")
    print()
    print("Architecture:")
    print("ALL EVENTS")
    print("    -> generic high-recall lightweight filter")
    print("    -> obvious + uncertain candidates")
    print("    -> Qwen final decision")
    print("    -> findings.jsonl / needs_review.jsonl")
    print("=" * 70)

    # Fail early with a useful message instead of a mysterious exception.
    try:
        client = get_ollama_client()
        client.list()
    except Exception as exc:
        raise SystemExit(
            f"\nCannot reach Ollama at {OLLAMA_BASE}\n"
            f"Check Ollama/network/model availability.\n"
            f"Error: {exc}\n"
        )

    try:
        collection = get_knowledge_collection()
        print(f"Knowledge base documents: {collection.count()}")
        if collection.count() == 0:
            print(
                "[WARN] Chroma knowledge base is empty. "
                "Analysis will continue without policy RAG."
            )
    except Exception as exc:
        print(f"[WARN] Chroma unavailable: {exc}")

    process_file(
        csv_path=args.csv,
        output_path=args.output,
        review_path=args.review_output,
        workers=args.workers,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
