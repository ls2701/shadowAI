from __future__ import annotations

import argparse
import csv
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
from threading import local

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
OVERLAP_RATIO = 0.10
SESSION_GAP_SECONDS = 300
DEFAULT_QWEN_WORKERS = 2

# One compact list is intentionally kept as the source of truth.
# Add new domains here and AI detection automatically picks them up.
AI_APPLICATIONS: dict[str, tuple[str, bool]] = {
    # Generative AI / assistants
    "chatgpt.com": ("ChatGPT", False),
    "chat.openai.com": ("ChatGPT", False),
    "openai.com": ("ChatGPT", False),
    "oaistatic.com": ("ChatGPT", False),
    "oaiusercontent.com": ("ChatGPT", False),
    "claude.ai": ("Claude", False),
    "anthropic.com": ("Claude", False),
    "gemini.google.com": ("Gemini", False),
    "generativelanguage.googleapis.com": ("Gemini", False),
    "bard.google.com": ("Gemini", False),
    "perplexity.ai": ("Perplexity", False),
    "cursor.com": ("Cursor", False),
    "huggingface.co": ("Hugging Face", False),
    "cohere.ai": ("Cohere", False),
    "cohere.com": ("Cohere", False),
    "mistral.ai": ("Mistral", False),
    "poe.com": ("Poe", False),
    "character.ai": ("Character.AI", False),
    "deepseek.com": ("DeepSeek", False),
    "x.ai": ("Grok/x.ai", False),
    "grok.com": ("Grok/x.ai", False),
    "replicate.com": ("Replicate", False),
    "together.ai": ("Together AI", False),
    "groq.com": ("Groq", False),
    "fireworks.ai": ("Fireworks AI", False),
    "copilot.microsoft.com": ("Microsoft Copilot", True),
    "githubcopilot.com": ("GitHub Copilot", True),
    "bedrock-runtime": ("AWS Bedrock", False),
    "sagemaker": ("AWS SageMaker", False),
    "openai.azure.com": ("Azure OpenAI", False),
    "jasper.ai": ("Jasper AI", False),
    "writesonic.com": ("Writesonic", False),
    "copy.ai": ("Copy.ai", False),
    "elevenlabs.io": ("ElevenLabs", False),
    "runwayml.com": ("Runway ML", False),
    "midjourney.com": ("Midjourney", False),
    "stability.ai": ("Stability AI", False),

    # Chat / conversational AI platforms
    "intercom.io": ("Intercom", False),
    "drift.com": ("Drift", False),
    "zdassets.com": ("Zendesk Chat", False),
    "zopim.com": ("Zendesk Chat", False),
    "tidio.co": ("Tidio", False),
    "tidiochat.com": ("Tidio", False),
    "crisp.chat": ("Crisp", False),
    "livechatinc.com": ("LiveChat", False),
    "freshchat.com": ("Freshchat", False),
    "manychat.com": ("ManyChat", False),
    "chatfuel.com": ("Chatfuel", False),
    "landbot.io": ("Landbot", False),
    "kore.ai": ("Kore.ai", False),
    "yellow.ai": ("Yellow.ai", False),
    "haptik.ai": ("Haptik", False),
    "ada.cx": ("Ada", False),
    "verloop.io": ("Verloop", False),
    "mobilemonkey.com": ("MobileMonkey", False),
    "botpress.cloud": ("Botpress", False),
    "voiceflow.com": ("Voiceflow", False),
    "dialogflow.cloud.google": ("Dialogflow", False),
    "watsonassistant.watson.cloud.ibm.com": ("Watson Assistant", False),
    "rasa.com": ("Rasa", False),
    "chatbase.co": ("Chatbase", False),
}

# AWS eventSource is an explicit signal in CloudTrail.
AWS_AI_EVENT_SOURCES = (
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

AI_TEXT_HINTS = (
    'category="artificial intelligence"',
    "category=artificial intelligence",
    "artificial intelligence",
    "/backend-api/conversation",
    "/v1/messages",
    "/chat/completions",
    "/v1/complete",
    "/v1/chat",
    "generativelanguage",
    "anthropic.claude",
    "chatbot",
    "ai-chatbot",
    "genai",
    "generative-ai",
    "gpt-",
    "-gpt",
    "llm-",
    "-llm",
    "virtual assistant",
    "ai assistant",
    "conversational ai",
    "live chat",
    "livechat",
    "/api/chat",
    "/api/converse",
    "conversation_id",
    "chat_completion",
    "/bot/",
)

# Fast event-level hint detection: one regex scan instead of one `in` scan per hint.
AI_HINT_REGEX = re.compile(
    "|".join(re.escape(h) for h in sorted(AI_TEXT_HINTS, key=len, reverse=True)),
    re.IGNORECASE,
)

# O(1)-ish domain lookup by trying only DNS suffixes, rather than looping over
# every known application for every event.
AI_DOMAIN_INDEX = {k.lower().rstrip("."): v for k, v in AI_APPLICATIONS.items()}


# ============================================================
# CHROMA / OLLAMA
# ============================================================
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = OllamaEmbeddingFunction(
    url=f"{OLLAMA_BASE}/api/embeddings",
    model_name=EMBEDDING_MODEL,
)
knowledge = chroma_client.get_or_create_collection(
    name=KNOWLEDGE_COLLECTION,
    embedding_function=embedding_function,
)

_thread_local = local()
_policy_cache: dict[str, str] = {}
_policy_lock = threading.Lock()


def get_ollama_client() -> Client:
    """One client per worker thread so Qwen calls can run concurrently."""
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = Client(host=OLLAMA_BASE)
        _thread_local.client = client
    return client


# ============================================================
# FAST AI DETECTION
# ============================================================
def scan_text_hints(raw_text: str) -> list[str]:
    if not raw_text:
        return []
    return sorted({m.group(0).lower() for m in AI_HINT_REGEX.finditer(raw_text)})


def detect_ai_domain(domain: str) -> tuple[str | None, bool]:
    """Return (app, approved) using DNS suffixes; no loop over all AI domains."""
    if not domain:
        return None, False

    d = domain.lower().strip().rstrip(".")
    labels = d.split(".")

    # Check exact domain and then valid parent-domain suffixes.
    for i in range(len(labels) - 1):
        candidate = ".".join(labels[i:])
        hit = AI_DOMAIN_INDEX.get(candidate)
        if hit:
            return hit

    # Keep support for non-DNS identifiers such as "bedrock-runtime".
    return AI_DOMAIN_INDEX.get(d, (None, False))


# ============================================================
# NORMALIZATION
# ============================================================
def dns_decode(text: str) -> str | None:
    # Same intent as the original decoder, but compiled once.
    pattern = re.compile(r"((?:\(\d+\)[A-Za-z0-9_-]+)+)\(0\)")
    label_pattern = re.compile(r"\(\d+\)([A-Za-z0-9_-]+)")
    for match in pattern.finditer(text):
        labels = label_pattern.findall(match.group(1))
        if labels:
            return ".".join(labels).lower()
    return None


DNS_PATTERN = re.compile(r"((?:\(\d+\)[A-Za-z0-9_-]+)+)\(0\)")
DNS_LABEL_PATTERN = re.compile(r"\(\d+\)([A-Za-z0-9_-]+)")


def fast_dns_decode(text: str) -> str | None:
    m = DNS_PATTERN.search(text)
    if not m:
        return None
    labels = DNS_LABEL_PATTERN.findall(m.group(1))
    return ".".join(labels).lower() if labels else None


KV_QUOTED = re.compile(r'([A-Za-z0-9_.-]+)="([^"]*)"')
KV_PLAIN = re.compile(r"([A-Za-z0-9_.-]+)=([^\s]+)")


def extract_kv(event: str, key: str) -> str | None:
    # Exact key search avoids running two regexes against the complete event
    # dictionary for every field.
    quoted = re.search(rf'{re.escape(key)}="([^"]*)"', event)
    if quoted:
        return quoted.group(1)
    plain = re.search(rf"{re.escape(key)}=([^\s]+)", event)
    return plain.group(1) if plain else None


def first_str(value) -> str | None:
    return value if isinstance(value, str) and value else None


def normalize_row(sensor: str, source: str, ts: str, event: str) -> dict:
    out = {
        "timestamp": ts,
        "ts_epoch": None,
        "sensor": sensor,
        "source": source,
        "user": None,
        "hostname": None,
        "domain": None,
        "bytes": 0,
        # Keep no raw event in the common path. It is not sent to Qwen.
        "text_hints": [],
        "ai_app": None,
        "ai_approved": False,
    }

    # Parse timestamp once. Session logic reuses this value.
    try:
        out["ts_epoch"] = datetime.fromisoformat(ts.strip().replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, OSError):
        pass

    e = event
    low = e.lower()

    try:
        if "device_name=" in low or "id=firewall" in low:
            out["domain"] = (extract_kv(e, "dstname") or "").lower() or None
            try:
                out["bytes"] = int(extract_kv(e, "sent") or 0) + int(extract_kv(e, "rcvd") or 0)
            except ValueError:
                out["bytes"] = 0

        elif "mswineventlog" in low and "packet" in low:
            out["domain"] = fast_dns_decode(e)

        elif e.lstrip().startswith("{"):
            # Parse JSON once. Prefer structured access; use a regex fallback only
            # if the source doesn't expose the common structured fields.
            j = json.loads(e)
            if not isinstance(j, dict):
                j = {}

            event_source = first_str(j.get("eventSource"))
            if event_source:
                event_source = event_source.lower()
                out["hostname"] = event_source

                if any(svc in event_source for svc in AWS_AI_EVENT_SOURCES):
                    out["text_hints"].append(f"aws:{event_source}")

                out["user"] = first_str(j.get("userName"))
            else:
                host = j.get("host") if isinstance(j.get("host"), dict) else {}
                user_obj = host.get("user") if isinstance(host.get("user"), dict) else {}
                out["hostname"] = first_str(host.get("hostname"))
                out["user"] = first_str(user_obj.get("name"))

                # Common structured fields first: no json.dumps on the hot path.
                out["domain"] = (
                    first_str(j.get("destinationHostname"))
                    or first_str((j.get("destination") or {}).get("hostname") if isinstance(j.get("destination"), dict) else None)
                )

                if not out["user"]:
                    out["user"] = first_str(j.get("userName"))

                # Rare fallback for unusual AWN/Sysmon shapes. This preserves the
                # old broad behavior without serializing every JSON object.
                if not out["domain"] or not out["user"]:
                    flat = json.dumps(j, ensure_ascii=False, separators=(",", ":"))
                    if not out["domain"]:
                        m = re.search(r'destinationHostname["\s:]+([a-z0-9._-]+)', flat, re.I)
                        if m:
                            out["domain"] = m.group(1).lower()
                    if not out["user"]:
                        m = re.search(r'\\([A-Za-z0-9_.-]+)"', flat)
                        if m:
                            out["user"] = m.group(1)

    except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
        pass

    # Scan hints exactly once, on the full event.
    out["text_hints"].extend(scan_text_hints(e))
    out["text_hints"] = sorted(set(out["text_hints"]))

    # AI decision is attached to the row once; later chunk processing only
    # aggregates these fields, so known-domain matching is never repeated.
    app, approved = detect_ai_domain(out["domain"] or "")
    out["ai_app"] = app
    out["ai_approved"] = approved

    return out


# ============================================================
# STREAMING CHUNKING
# ============================================================
class EntityState:
    __slots__ = ("events", "previous_ts", "candidate_apps", "candidate_hints", "unknown_domains", "total_bytes")

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.previous_ts: float | None = None
        self.candidate_apps: dict[str, bool] = {}
        self.candidate_hints: set[str] = set()
        self.unknown_domains: set[str] = set()
        self.total_bytes = 0

    def reset_after_chunk(self, overlap: int) -> None:
        self.events = self.events[-overlap:] if overlap else []
        self.candidate_apps = {e["ai_app"]: e["ai_approved"] for e in self.events if e.get("ai_app")}
        self.candidate_hints = {h for e in self.events for h in (e.get("text_hints") or [])}
        self.unknown_domains = {
            e["domain"] for e in self.events if e.get("domain") and not e.get("ai_app") and not e.get("text_hints")
        }
        self.total_bytes = sum(e.get("bytes", 0) for e in self.events)


def event_is_ai_candidate(e: dict) -> bool:
    return bool(e.get("ai_app") or e.get("text_hints"))


def build_chunk_from_state(state: EntityState, sensor: str, source: str) -> dict:
    events = state.events
    apps = dict(state.candidate_apps)
    hints = sorted(state.candidate_hints)
    unknown = sorted(state.unknown_domains)
    is_ai_related = bool(apps or hints)
    tier = "domain" if apps else ("text_hint" if hints else "none")

    return {
        "sensor": sensor,
        "source": source,
        "start_ts": events[0]["timestamp"],
        "end_ts": events[-1]["timestamp"],
        "event_count": len(events),
        "events": events,
        "applications": apps,
        "text_hints": hints,
        "unknown_domains": unknown,
        "total_bytes": state.total_bytes,
        "is_ai_related": is_ai_related,
        "needs_review": bool(unknown) and not is_ai_related,
        "match_tier": tier,
    }


def add_event_to_state(state: EntityState, row: dict, sensor: str, source: str):
    """Yield completed session/chunks and keep only a tiny rolling buffer."""
    ts = row.get("ts_epoch")

    # Session boundary.
    if state.previous_ts is not None and ts is not None:
        if ts - state.previous_ts > SESSION_GAP_SECONDS:
            if len(state.events) > OVERLAP_COUNT:
                yield build_chunk_from_state(state, sensor, source)
            state.__init__()

    state.events.append(row)
    state.total_bytes += row.get("bytes", 0)
    if row.get("ai_app"):
        state.candidate_apps[row["ai_app"]] = row["ai_approved"]
    for h in row.get("text_hints") or []:
        state.candidate_hints.add(h)
    if row.get("domain") and not row.get("ai_app") and not row.get("text_hints"):
        state.unknown_domains.add(row["domain"])

    if ts is not None:
        state.previous_ts = ts

    # Emit a 40-event chunk, retain only 10% overlap.
    if len(state.events) >= MAX_EVENTS_PER_CHUNK:
        yield build_chunk_from_state(state, sensor, source)
        state.reset_after_chunk(OVERLAP_COUNT)


OVERLAP_COUNT = max(1, math.floor(MAX_EVENTS_PER_CHUNK * OVERLAP_RATIO))


# ============================================================
# BACKWARD-COMPATIBILITY API
# ============================================================
# The FastAPI server used by the UI may still call the original
# load_and_normalize()/build_intelligent_chunks() functions.  Keep those
# function names, but make them streaming generators so the API does not need
# the old all-rows-in-memory implementation.
def load_and_normalize(csv_path: Path):
    """Backward-compatible streaming replacement for the old loader."""
    with Path(csv_path).open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=",")
        for raw_row in reader:
            normalized = {
                (k or "").strip().lower(): (v or "").strip()
                for k, v in raw_row.items()
            }
            yield normalize_row(
                normalized.get("sensor", ""),
                normalized.get("source", ""),
                normalized.get("timestamp", ""),
                normalized.get("event", ""),
            )


def build_intelligent_chunks(rows):
    """Backward-compatible streaming chunk generator.

    This accepts the iterator returned by load_and_normalize() and yields the
    same chunk dictionaries used by the optimized pipeline. It does not load
    the complete file into memory.
    """
    states: dict[tuple[str, str], EntityState] = {}

    for row in rows:
        sensor = row.get("sensor", "")
        source = row.get("source", "")
        key = (sensor, source)
        state = states.setdefault(key, EntityState())

        for chunk in add_event_to_state(state, row, sensor, source):
            yield chunk

    # Flush final partial sessions.
    for (sensor, source), state in states.items():
        if len(state.events) > OVERLAP_COUNT:
            yield build_chunk_from_state(state, sensor, source)


# ============================================================
# POLICY RAG — cached, one query per app
# ============================================================
def retrieve_policy_for_app(app: str) -> str:
    cached = _policy_cache.get(app)
    if cached is not None:
        return cached

    # A single query replaces the original four per-chunk retrieval queries.
    query = (
        f"Shadow AI policy for {app}: approved or prohibited status, "
        "data classification, DLP restrictions, severity criteria, "
        "and required response/action."
    )

    with _policy_lock:
        cached = _policy_cache.get(app)
        if cached is not None:
            return cached

        res = knowledge.query(query_texts=[query], n_results=8)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        passages = []
        seen = set()
        for d, m in zip(docs, metas):
            key = d[:120]
            if key in seen:
                continue
            seen.add(key)
            source = m.get("source", "policy") if isinstance(m, dict) else "policy"
            passages.append(f"[source: {source}] {d}")

        context = "\n\n".join(passages)
        _policy_cache[app] = context
        return context


def retrieve_policy(chunk: dict) -> str:
    apps = sorted(chunk["applications"])
    if apps:
        contexts = [retrieve_policy_for_app(app) for app in apps]
        return "\n\n".join(c for c in contexts if c)

    # Text-hint-only chunks use one generic cached policy context.
    return retrieve_policy_for_app("generic_shadow_ai")


SYSTEM_PROMPT = """You are a Shadow AI Security Analyst.

You are given a chunk of grouped network/endpoint events plus retrieved security policy.

match_tier rules:
- domain: a known AI/chatbot domain matched exactly or as a valid subdomain. Strong evidence.
- text_hint: no known AI domain was confirmed; only a generic AI/chatbot/API hint matched. Verify it from the event context and lower confidence when ambiguous.
- none: no AI evidence.

A finding qualifies as Shadow AI chatbot/assistant usage only when the evidence indicates
communication with an external conversational AI, AI assistant, or generative-AI-backed tool.
Do not treat ordinary web browsing, customer-support ticketing, or unrelated bot/crawler traffic as Shadow AI.

Use ONLY the chunk and retrieved policy. Do not invent facts.
Return ONLY valid JSON.
"""


def analyze_chunk(chunk: dict) -> dict:
    # Keep the Qwen prompt compact: only fields actually needed for the verdict.
    analysis_started = time.monotonic()
    events_summary = json.dumps(
        [
            {
                "timestamp": e["timestamp"],
                "domain": e.get("domain"),
                "user": e.get("user"),
                "bytes": e.get("bytes", 0),
                "text_hints": e.get("text_hints", []) if chunk["match_tier"] == "text_hint" else [],
            }
            for e in chunk["events"]
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    rag_context = retrieve_policy(chunk)
    apps = list(chunk["applications"].keys())

    prompt = f"""CHUNK:
sensor={chunk['sensor']}
source={chunk['source']}
start={chunk['start_ts']}
end={chunk['end_ts']}
event_count={chunk['event_count']}
match_tier={chunk['match_tier']}
applications_seen={apps}
text_pattern_hints={chunk['text_hints']}
total_bytes={chunk['total_bytes']}

events={events_summary}

RETRIEVED SECURITY POLICY:
{rag_context or '(no policy retrieved — flag low confidence)'}

Return JSON with exactly these keys:
classification, risk, confidence, application, user, reason, evidence, policy, severity, recommended_action
"""

    client = get_ollama_client()
    response = client.chat(
        model=ANALYSIS_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": 0},
    )

    raw = response["message"]["content"].strip()
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    analysis_elapsed = time.monotonic() - analysis_started

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "classification": None,
            "confidence": "low",
            "reason": "LLM returned unparseable output",
            "parse_error": raw[:300],
        }

    # Metadata used only for terminal reporting/debugging.
    result["_analysis_seconds"] = round(analysis_elapsed, 2)
    result["_sensor"] = chunk["sensor"]
    result["_source"] = chunk["source"]
    result["_start_ts"] = chunk["start_ts"]
    result["_end_ts"] = chunk["end_ts"]
    result["_applications"] = list(chunk["applications"].keys())
    result["_approved"] = chunk["applications"]
    result["_match_tier"] = chunk["match_tier"]
    result["_bytes"] = chunk["total_bytes"]
    result["_event_count"] = chunk["event_count"]
    return result


# ============================================================
# STREAMING CSV + BOUNDED CONCURRENT ANALYSIS
# ============================================================
def submit_completed_chunk(executor, pending, chunk, max_pending):
    if not chunk["is_ai_related"]:
        return
    future = executor.submit(analyze_chunk, chunk)
    pending.add(future)

    # Backpressure: never queue an unbounded number of Qwen jobs.
    if len(pending) >= max_pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        return done
    return set()


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def format_rate(value: float) -> str:
    return f"{value:,.0f}"


def consume_done(done, pending, findings_fp, runtime_started, findings_counter=None):
    completed = []
    for future in done:
        pending.discard(future)
        result = future.result()
        findings_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
        findings_fp.flush()
        completed.append(result)

        apps = result.get("_applications") or []
        app_text = ", ".join(apps) if apps else "[]"
        tier = result.get("_match_tier", "unknown")
        bytes_sent = result.get("_bytes", 0)
        severity = result.get("severity", "-")
        action = result.get("recommended_action", "-")
        analysis_time = result.get("_analysis_seconds", 0)
        elapsed = time.monotonic() - runtime_started

        print(
            f"  {app_text!s:30} tier={tier:10} "
            f"{bytes_sent:>9}B  sev={severity:<9} "
            f"analysis={analysis_time:>6.1f}s  elapsed={format_duration(elapsed):>9}\n"
            f"    action: {action}"
        )
    return completed


def process_file(
    csv_path: Path,
    output_path: Path,
    review_path: Path,
    workers: int,
    progress_every: int,
) -> None:
    states: dict[tuple[str, str], EntityState] = {}
    pending = set()
    max_pending = max(workers * 2, 2)

    rows_seen = 0
    chunks_seen = 0
    ai_chunks = 0
    review_chunks = 0
    findings = 0
    started = time.monotonic()

    with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as fh, \
         output_path.open("w", encoding="utf-8") as findings_fp, \
         review_path.open("w", encoding="utf-8") as review_fp, \
         ThreadPoolExecutor(max_workers=workers) as executor:

        reader = csv.DictReader(fh, delimiter=",")

        for raw_row in reader:
            normalized = {
                (k or "").strip().lower(): (v or "").strip()
                for k, v in raw_row.items()
            }

            sensor = normalized.get("sensor", "")
            source = normalized.get("source", "")
            timestamp = normalized.get("timestamp", "")
            event = normalized.get("event", "")

            row = normalize_row(sensor, source, timestamp, event)
            key = (sensor, source)
            state = states.setdefault(key, EntityState())

            for chunk in add_event_to_state(state, row, sensor, source):
                chunks_seen += 1
                if chunk["needs_review"]:
                    review_chunks += 1
                    review_fp.write(
                        json.dumps(
                            {
                                "sensor": chunk["sensor"],
                                "source": chunk["source"],
                                "start_ts": chunk["start_ts"],
                                "end_ts": chunk["end_ts"],
                                "unknown_domains": chunk["unknown_domains"],
                                "event_count": chunk["event_count"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if chunk["is_ai_related"]:
                    ai_chunks += 1
                    print(
                        f"\n[AI chunk {ai_chunks}] "
                        f"apps={list(chunk['applications'].keys()) or '[]'} "
                        f"tier={chunk['match_tier']} events={chunk['event_count']} "
                        f"bytes={chunk['total_bytes']:,}"
                    )
                    done = submit_completed_chunk(executor, pending, chunk, max_pending)
                    if done:
                        completed = consume_done(done, pending, findings_fp, started)
                        findings += len(completed)

            rows_seen += 1
            if progress_every and rows_seen % progress_every == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                rate = rows_seen / elapsed
                print(
                    f"Processed {rows_seen:,} rows | {rate:,.0f} rows/s | "
                    f"AI chunks {ai_chunks:,} | findings {findings:,}"
                )

        # Flush final partial sessions. Do this after input EOF because a session
        # may be shorter than 40 events.
        for (sensor, source), state in states.items():
            if len(state.events) <= OVERLAP_COUNT:
                continue

            chunk = build_chunk_from_state(state, sensor, source)
            chunks_seen += 1

            if chunk["needs_review"]:
                review_chunks += 1
                review_fp.write(
                    json.dumps(
                        {
                            "sensor": chunk["sensor"],
                            "source": chunk["source"],
                            "start_ts": chunk["start_ts"],
                            "end_ts": chunk["end_ts"],
                            "unknown_domains": chunk["unknown_domains"],
                            "event_count": chunk["event_count"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if chunk["is_ai_related"]:
                ai_chunks += 1
                print(
                    f"\n[AI chunk {ai_chunks}] "
                    f"apps={list(chunk['applications'].keys()) or '[]'} "
                    f"tier={chunk['match_tier']} events={chunk['event_count']} "
                    f"bytes={chunk['total_bytes']:,}"
                )
                done = submit_completed_chunk(executor, pending, chunk, max_pending)
                if done:
                    completed = consume_done(done, pending, findings_fp, started)
                    findings += len(completed)

        # Drain all remaining Qwen jobs.
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            completed = consume_done(done, pending, findings_fp, started)
            findings += len(completed)

    elapsed = max(time.monotonic() - started, 0.001)
    print("\nDone.")
    print(f"Rows processed      : {rows_seen:,}")
    print(f"Chunks built        : {chunks_seen:,}")
    print(f"AI chunks submitted : {ai_chunks:,}")
    print(f"Findings written    : {findings:,}")
    print(f"Review chunks       : {review_chunks:,}")
    print(f"Elapsed             : {format_duration(elapsed)}")
    print(f"Throughput          : {format_rate(rows_seen / elapsed)} rows/s")
    print(f"Findings file       : {output_path}")
    print(f"Review file         : {review_path}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Optimized streaming Shadow AI analyzer")
    ap.add_argument("csv", type=Path, help="raw log CSV: timestamp,sensor,source,event")
    ap.add_argument("--workers", type=int, default=DEFAULT_QWEN_WORKERS,
                    help="parallel Qwen workers; start with 2 and benchmark 1/2/3/4")
    ap.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT_FILE))
    ap.add_argument("--review-output", type=Path, default=Path(DEFAULT_REVIEW_FILE))
    ap.add_argument("--progress-every", type=int, default=10_000)
    args = ap.parse_args()

    if knowledge.count() == 0:
        raise SystemExit("Knowledge base is EMPTY. Run: python build_knowledge_base.py")

    workers = max(1, args.workers)

    print(f"Knowledge RAG     : {knowledge.count()} policy chunks")
    print(f"Qwen workers      : {workers}")
    print(f"Chunk size        : {MAX_EVENTS_PER_CHUNK}")
    print(f"Overlap           : {OVERLAP_COUNT} events")
    print(f"Session gap       : {SESSION_GAP_SECONDS}s")
    print(f"Streaming mode    : ON")
    print(f"AI domain lookup  : indexed suffix matching")
    print(f"RAG strategy      : cached, one query per application")
    print()

    process_file(
        args.csv,
        args.output,
        args.review_output,
        workers,
        args.progress_every,
    )


if __name__ == "__main__":
    main()
