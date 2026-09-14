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
from collections import defaultdict
import queue

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from ollama import Client, ResponseError

# ============================================================
# CONFIG (adjustable via CLI)
# ============================================================
DEFAULT_OLLAMA_HOST = "192.168.100.100"
OLLAMA_PORT = 11434
CHROMA_PATH = "./chroma_db"
KNOWLEDGE_COLLECTION = "shadow_ai_knowledge"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
ANALYSIS_MODEL = "qwen2.5:14b"
DOMAIN_CLASSIFY_MODEL = "qwen2.5:7b-instruct"   # can be overridden via CLI

DEFAULT_OUTPUT_FILE = "findings.jsonl"
DEFAULT_REVIEW_FILE = "needs_review.jsonl"

MAX_EVENTS_PER_CHUNK = 40
OVERLAP_RATIO = 0.10
SESSION_GAP_SECONDS = 300
DEFAULT_QWEN_WORKERS = 2

DYNAMIC_CACHE_FILE = Path("dynamic_domain_cache.json")
CONFIDENCE_THRESHOLD = 0.8
DOMAIN_BATCH_SIZE = 10          # number of domains to classify in one LLM call
CACHE_SAVE_INTERVAL = 100       # save cache after every N classifications

# ============================================================
# STATIC AI DOMAIN LIST (abbreviated – add your full list)
# ============================================================
AI_APPLICATIONS: dict[str, tuple[str, bool]] = {
    "chatgpt.com": ("ChatGPT", False),
    "openai.com": ("ChatGPT", False),
    "claude.ai": ("Claude", False),
    "anthropic.com": ("Claude", False),
    "gemini.google.com": ("Gemini", False),
    "perplexity.ai": ("Perplexity", False),
    "cursor.com": ("Cursor", False),
    "huggingface.co": ("Hugging Face", False),
    # ... add all your static entries ...
}

AWS_AI_EVENT_SOURCES = (
    "bedrock", "sagemaker", "comprehend", "rekognition", "textract",
    "polly", "lex.amazonaws", "personalize", "kendra", "forecast", "transcribe",
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

AI_HINT_REGEX = re.compile(
    "|".join(re.escape(h) for h in sorted(AI_TEXT_HINTS, key=len, reverse=True)),
    re.IGNORECASE,
)

AI_DOMAIN_INDEX = {k.lower().rstrip("."): v for k, v in AI_APPLICATIONS.items()}

# ============================================================
# CHROMA / OLLAMA
# ============================================================
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = OllamaEmbeddingFunction(
    url=f"http://{DEFAULT_OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
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
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = Client(host=f"http://{DEFAULT_OLLAMA_HOST}:{OLLAMA_PORT}")
        _thread_local.client = client
    return client

# ============================================================
# DYNAMIC CACHE
# ============================================================
_cache_lock = threading.Lock()
_dynamic_cache: dict = {}
_cache_save_counter = 0

def load_dynamic_cache() -> None:
    global _dynamic_cache
    if DYNAMIC_CACHE_FILE.exists():
        with DYNAMIC_CACHE_FILE.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        for domain, v in raw.items():
            _dynamic_cache[domain] = {
                "status": v.get("status", "unknown"),
                "app": v.get("app"),
                "approved": v.get("approved", False),
                "confidence": v.get("confidence", 0.0),
                "last_attempt": v.get("last_attempt", 0)
            }
    else:
        _dynamic_cache = {}

def save_dynamic_cache() -> None:
    with _cache_lock:
        with DYNAMIC_CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(_dynamic_cache, f, indent=2, ensure_ascii=False)

def cache_get(domain: str) -> dict | None:
    with _cache_lock:
        return _dynamic_cache.get(domain)

def cache_set(domain: str, status: str, app: str | None, approved: bool, confidence: float) -> None:
    global _cache_save_counter
    with _cache_lock:
        _dynamic_cache[domain] = {
            "status": status,
            "app": app,
            "approved": approved,
            "confidence": confidence,
            "last_attempt": time.time()
        }
        _cache_save_counter += 1
        if _cache_save_counter % CACHE_SAVE_INTERVAL == 0:
            save_dynamic_cache()

# ============================================================
# DOMAIN CLASSIFIER (background, batched)
# ============================================================
class DomainClassifier:
    def __init__(self, model: str, batch_size: int = DOMAIN_BATCH_SIZE, classifier_workers: int = 4):
        self.model = model
        self.batch_size = batch_size
        self.pending_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=classifier_workers)
        self.futures = set()
        self._queued = set()               # <-- NEW: tracks domains already queued
        self.worker_thread = threading.Thread(target=self._collector, daemon=True)
        self.worker_thread.start()

    def submit(self, domain: str) -> None:
        """Add domain to classification queue only if brand new."""
        # Skip if already queued in this run
        if domain in self._queued:
            return
        entry = cache_get(domain)
        if entry is not None:
            return   # already known (ai, non_ai, or unknown from previous runs)
        # Mark as unknown in cache and in the queued set
        cache_set(domain, "unknown", None, False, 0.0)
        self._queued.add(domain)
        self.pending_queue.put(domain)

    def _collector(self) -> None:
        while not self.stop_event.is_set():
            domains = []
            try:
                while len(domains) < self.batch_size:
                    try:
                        domain = self.pending_queue.get(timeout=0.5)
                        domains.append(domain)
                    except queue.Empty:
                        break
                if domains:
                    future = self.executor.submit(self._classify_batch, domains)
                    self.futures.add(future)
                    # Clean up finished futures to save memory
                    self.futures = {f for f in self.futures if not f.done()}
            except Exception as e:
                print(f"[domain classifier collector error] {e}")
                time.sleep(1)

        # Process leftover domains after stop signal
        while not self.pending_queue.empty():
            domains = []
            try:
                domains.append(self.pending_queue.get_nowait())
                while len(domains) < self.batch_size:
                    domains.append(self.pending_queue.get_nowait())
            except queue.Empty:
                pass
            if domains:
                future = self.executor.submit(self._classify_batch, domains)
                self.futures.add(future)

        # Wait for all classification tasks to finish
        for future in self.futures:
            future.result()

    def _classify_batch(self, domains: list[str]) -> None:
        try:
            client = get_ollama_client()

            prompt = BATCH_CLASSIFY_PROMPT.format(
                domains="\n".join(domains)
            )

            resp = client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                format="json",
                options={
                    "temperature": 0
                },
            )

            raw = resp["message"]["content"].strip()

            # Remove thinking/reasoning tags if the model returns them
            if "<think>" in raw:
                raw = raw.split("</think>")[-1].strip()

            # Remove markdown code fences if present
            raw = (
                raw.replace("```json", "")
                .replace("```", "")
                .strip()
            )

            result = json.loads(raw)

            # Process each domain returned by the classifier
            for d in domains:
                res = result.get(d, {})

                classification = str(
                    res.get("classification", "UNKNOWN")
                ).upper()

                confidence = float(
                    res.get("confidence", 0.0) or 0.0
                )

                app_name = res.get("app_name")

                # ------------------------------------------------
                # HIGH-CONFIDENCE AI
                # ------------------------------------------------
                if (
                    classification == "AI"
                    and confidence >= CONFIDENCE_THRESHOLD
                ):
                    cache_set(
                        d,
                        "ai",
                        app_name or d,
                        False,
                        confidence
                    )

                    print(
                        f"  [classify] AI: {d} -> "
                        f"{app_name or d} "
                        f"(confidence={confidence:.2f})"
                    )

                # ------------------------------------------------
                # HIGH-CONFIDENCE NON-AI
                # ------------------------------------------------
                elif (
                    classification == "NON_AI"
                    and confidence >= CONFIDENCE_THRESHOLD
                ):
                    cache_set(
                        d,
                        "non_ai",
                        None,
                        False,
                        confidence
                    )

                    print(
                        f"  [classify] Non-AI: {d} "
                        f"(confidence={confidence:.2f})"
                    )

                # ------------------------------------------------
                # UNCERTAIN → UNKNOWN
                # ------------------------------------------------
                else:
                    cache_set(
                        d,
                        "unknown",
                        None,
                        False,
                        confidence
                    )

                    print(
                        f"  [classify] UNKNOWN: {d} "
                        f"(model={classification}, "
                        f"confidence={confidence:.2f})"
                    )

        except (
            ResponseError,
            json.JSONDecodeError,
            KeyError,
            ConnectionError,
            ValueError,
            TypeError,
        ) as e:

            print(
                f"  [classify error] {e}. "
                f"Marking batch as UNKNOWN."
            )

            # Never mark domains as NON-AI when the LLM itself failed.
            # Keep them UNKNOWN so they can be reviewed/retried later.
            for d in domains:
                cache_set(
                    d,
                    "unknown",
                    None,
                    False,
                    0.0
                )


    def shutdown(self, timeout=30) -> None:
        self.stop_event.set()
        self.worker_thread.join(timeout=timeout)
        self.executor.shutdown(wait=True, cancel_futures=False)
        
BATCH_CLASSIFY_PROMPT = """You are a cybersecurity analyst specializing in AI and generative-AI services.

For each domain below, determine whether it is a REAL and KNOWN:

AI service
chatbot
generative-AI service
LLM service
AI assistant
AI-powered application

Think carefully before deciding. Use your existing knowledge to identify the organization and purpose of the domain.

IMPORTANT:

Do NOT guess.
Do NOT classify a domain as AI just because it contains words like "ai", "chat", "bot", "gpt", or "llm".
The domain must be associated with an actual AI product or service.
Consider both the domain and its subdomain.
If you are not confident, return UNKNOWN.
Accuracy is more important than classifying every domain.
Do not invent company names, products, or evidence.

Before making the decision, internally consider:

What service or company does this domain belong to?
Is it actually an AI/chatbot/generative-AI service?
Is there enough evidence to make a confident decision?

Return ONLY valid JSON.

For each domain return:

{
"classification": "AI" | "NON_AI" | "UNKNOWN",
"is_ai": true | false,
"app_name": "Product name" | null,
"confidence": 0.0,
"reason": "Short explanation"
}

Use confidence as follows:

0.90-1.00 = very confident
0.80-0.89 = confident
Below 0.80 = UNKNOWN

Do not turn uncertainty into NON_AI.

Domains:
{domains}
"""

# Instantiate later after args are parsed
domain_classifier = None

# ============================================================
# FAST AI DETECTION
# ============================================================
def scan_text_hints(raw_text: str) -> list[str]:
    if not raw_text:
        return []
    return sorted({m.group(0).lower() for m in AI_HINT_REGEX.finditer(raw_text)})

def detect_ai_domain(domain: str) -> tuple[str | None, bool]:
    if not domain:
        return None, False
    d = domain.lower().strip().rstrip(".")
    labels = d.split(".")
    for i in range(len(labels) - 1):
        candidate = ".".join(labels[i:])
        hit = AI_DOMAIN_INDEX.get(candidate)
        if hit:
            return hit
    return AI_DOMAIN_INDEX.get(d, (None, False))

# ============================================================
# NORMALIZATION
# ============================================================
DNS_PATTERN = re.compile(r"((?:\(\d+\)[A-Za-z0-9_-]+)+)\(0\)")
DNS_LABEL_PATTERN = re.compile(r"\(\d+\)([A-Za-z0-9_-]+)")

def fast_dns_decode(text: str) -> str | None:
    m = DNS_PATTERN.search(text)
    if not m:
        return None
    labels = DNS_LABEL_PATTERN.findall(m.group(1))
    return ".".join(labels).lower() if labels else None

_KV_QUOTED = re.compile(r'([A-Za-z0-9_]+)="([^"]*)"')
_KV_PLAIN = re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')

def extract_kv(event: str, key: str) -> str | None:
    # Try quoted value: key="value"
    m = re.search(rf'{re.escape(key)}="([^"]*)"', event)
    if m:
        return m.group(1)
    # Try plain value: key=value
    m = re.search(rf'{re.escape(key)}=([^\s]+)', event)
    if m:
        return m.group(1)
    return None

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
        "text_hints": [],
        "ai_app": None,
        "ai_approved": False,
    }

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

                out["domain"] = (
                    first_str(j.get("destinationHostname"))
                    or first_str((j.get("destination") or {}).get("hostname") if isinstance(j.get("destination"), dict) else None)
                )
                if not out["user"]:
                    out["user"] = first_str(j.get("userName"))

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

    out["text_hints"].extend(scan_text_hints(e))
    out["text_hints"] = sorted(set(out["text_hints"]))

    # Static domain detection
    app, approved = detect_ai_domain(out["domain"] or "")

    # Dynamic cache check / queue for classification
    if app is None and out["domain"]:
        entry = cache_get(out["domain"])
        if entry is not None:
            if entry["status"] == "ai":
                app, approved = entry["app"], entry["approved"]
            # if "unknown" or "non_ai", do nothing
        else:
            # Completely new domain – queue for classification
            domain_classifier.submit(out["domain"])
            app, approved = None, False

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

OVERLAP_COUNT = max(1, math.floor(MAX_EVENTS_PER_CHUNK * OVERLAP_RATIO))

def build_chunk_from_state(state: EntityState, sensor: str, source: str, user: str | None = None) -> dict:
    events = state.events
    apps = dict(state.candidate_apps)
    hints = sorted(state.candidate_hints)
    unknown = sorted(state.unknown_domains)
    is_ai_related = bool(apps or hints)
    tier = "domain" if apps else ("text_hint" if hints else "none")

    return {
        "sensor": sensor,
        "source": source,
        "user": user,
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

def add_event_to_state(state: EntityState, row: dict, sensor: str, source: str, user: str | None):
    ts = row.get("ts_epoch")
    if state.previous_ts is not None and ts is not None:
        if ts - state.previous_ts > SESSION_GAP_SECONDS:
            if state.events:
                yield build_chunk_from_state(state, sensor, source, user)
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

    if len(state.events) >= MAX_EVENTS_PER_CHUNK:
        yield build_chunk_from_state(state, sensor, source, user)
        state.reset_after_chunk(OVERLAP_COUNT)

# ============================================================
# POLICY RAG
# ============================================================
def retrieve_policy_for_app(app: str) -> str:
    cached = _policy_cache.get(app)
    if cached is not None:
        return cached

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

def summarize_events_for_llm(events: list[dict]) -> list[dict]:
    agg = defaultdict(lambda: {
        "count": 0,
        "total_bytes": 0,
        "first_seen": None,
        "last_seen": None
    })
    for e in events:
        domain = e.get("domain") or "unknown"
        bucket = agg[domain]
        bucket["count"] += 1
        bucket["total_bytes"] += e.get("bytes", 0)
        if bucket["first_seen"] is None or e["timestamp"] < bucket["first_seen"]:
            bucket["first_seen"] = e["timestamp"]
        if bucket["last_seen"] is None or e["timestamp"] > bucket["last_seen"]:
            bucket["last_seen"] = e["timestamp"]

    result = []
    for domain, data in agg.items():
        result.append({
            "domain": domain,
            "event_count": data["count"],
            "total_bytes": data["total_bytes"],
            "first_seen": data["first_seen"],
            "last_seen": data["last_seen"]
        })
    return result

def analyze_chunk(chunk: dict) -> dict:
    analysis_started = time.monotonic()
    events_summary = summarize_events_for_llm(chunk["events"])
    rag_context = retrieve_policy(chunk)
    apps = list(chunk["applications"].keys())

    prompt = f"""CHUNK:
sensor={chunk['sensor']}
source={chunk['source']}
user={chunk.get('user', 'unknown')}
start={chunk['start_ts']}
end={chunk['end_ts']}
event_count={chunk['event_count']}
match_tier={chunk['match_tier']}
applications_seen={apps}
text_pattern_hints={chunk['text_hints']}
total_bytes={chunk['total_bytes']}

events_summary={json.dumps(events_summary, ensure_ascii=False)}

RETRIEVED SECURITY POLICY:
{rag_context or '(no policy retrieved — flag low confidence)'}

Return JSON with exactly these keys:
classification, risk, confidence, application, user, reason, evidence, policy, severity, recommended_action
"""

    try:
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
        result = json.loads(raw)
    except (ResponseError, json.JSONDecodeError, KeyError, ConnectionError) as e:
        print(f"  [analysis error] {e}. Returning fallback.")
        result = {
            "classification": None,
            "confidence": "low",
            "reason": f"LLM error: {e}",
            "parse_error": str(e)[:300],
        }

    result["_analysis_seconds"] = round(time.monotonic() - analysis_started, 2)
    result["_sensor"] = chunk["sensor"]
    result["_source"] = chunk["source"]
    result["_user"] = chunk.get("user")
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
        return set()
    future = executor.submit(analyze_chunk, chunk)
    pending.add(future)
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

def consume_done(done, pending, findings_fp, runtime_started):
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
        user = result.get("_user") or "unknown"
        elapsed = time.monotonic() - runtime_started

        print(
            f"  user={user:20} apps={app_text:30} tier={tier:10} "
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
    states: dict[tuple[str, str, str | None], EntityState] = {}
    pending = set()
    max_pending = max(workers * 2, 2)

    rows_seen = 0
    chunks_seen = 0
    ai_chunks = 0
    review_chunks = 0
    findings = 0
    started = time.monotonic()

    # Store chunks that need review (unknown domains) to re-check later
    review_chunks_memory = []

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
            user = row.get("user")
            key = (sensor, source, user)
            state = states.setdefault(key, EntityState())

            for chunk in add_event_to_state(state, row, sensor, source, user):
                chunks_seen += 1
                if chunk["needs_review"]:
                    review_chunks += 1
                    review_chunks_memory.append(chunk)
                    review_fp.write(
                        json.dumps(
                            {
                                "sensor": chunk["sensor"],
                                "source": chunk["source"],
                                "user": chunk.get("user"),
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
                        f"user={chunk.get('user', 'unknown')} "
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

        # Flush final partial sessions
        for (sensor, source, user), state in states.items():
            if state.events:
                chunk = build_chunk_from_state(state, sensor, source, user)
                chunks_seen += 1

                if chunk["needs_review"]:
                    review_chunks += 1
                    review_chunks_memory.append(chunk)
                    review_fp.write(
                        json.dumps(
                            {
                                "sensor": chunk["sensor"],
                                "source": chunk["source"],
                                "user": chunk.get("user"),
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
                        f"user={chunk.get('user', 'unknown')} "
                        f"apps={list(chunk['applications'].keys()) or '[]'} "
                        f"tier={chunk['match_tier']} events={chunk['event_count']} "
                        f"bytes={chunk['total_bytes']:,}"
                    )
                    done = submit_completed_chunk(executor, pending, chunk, max_pending)
                    if done:
                        completed = consume_done(done, pending, findings_fp, started)
                        findings += len(completed)

        # Drain initial pending analysis jobs
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            completed = consume_done(done, pending, findings_fp, started)
            findings += len(completed)

        # Wait for domain classifier to finish processing all unknown domains
        print("\nWaiting for domain classifier to finish...")
        domain_classifier.shutdown(timeout=30)

        # Re-examine review chunks: if any unknown domain is now classified as AI,
        # update the chunk and submit for analysis
        reclassified_chunks = 0
        for chunk in review_chunks_memory:
            new_apps = {}
            for domain in chunk.get("unknown_domains", []):
                entry = cache_get(domain)
                if entry and entry["status"] == "ai":
                    new_apps[entry["app"]] = entry.get("approved", False)

            if new_apps:
                chunk["applications"].update(new_apps)
                chunk["is_ai_related"] = True
                chunk["match_tier"] = "domain"
                # Remove the domains from unknown list (they are now known)
                chunk["unknown_domains"] = [d for d in chunk["unknown_domains"] if d not in new_apps]
                reclassified_chunks += 1
                print(
                    f"\n[Reclassified chunk] user={chunk.get('user', 'unknown')} "
                    f"apps={list(chunk['applications'].keys())} "
                    f"previously unknown={chunk['unknown_domains']}"
                )
                future = executor.submit(analyze_chunk, chunk)
                pending.add(future)

        if reclassified_chunks:
            print(f"\nReclassified {reclassified_chunks} chunk(s) after domain classification.")

        # Drain newly submitted analysis jobs
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            completed = consume_done(done, pending, findings_fp, started)
            findings += len(completed)

    # Save dynamic cache at end (outside the with, after files closed)
    save_dynamic_cache()

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
# BACKEND INTEGRATION FUNCTION
# ============================================================
def analyze_csv_file(
    csv_path: Path,
    output_path: Path,
    review_path: Path,
    workers: int = DEFAULT_QWEN_WORKERS,
    progress_every: int = 1000,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    classify_model: str = DOMAIN_CLASSIFY_MODEL,
    callback=None,
) -> dict:
    """
    Process a CSV file, write findings, and return summary statistics.
    `callback(job_id, message, partial_findings)` may be called to report progress.
    """
    global DEFAULT_OLLAMA_HOST, DOMAIN_CLASSIFY_MODEL, domain_classifier

    # Update global settings
    DEFAULT_OLLAMA_HOST = ollama_host
    DOMAIN_CLASSIFY_MODEL = classify_model

    # Reinitialize embedding/Chroma with correct host
    global embedding_function, chroma_client, knowledge
    embedding_function = OllamaEmbeddingFunction(
        url=f"http://{DEFAULT_OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    knowledge = chroma_client.get_or_create_collection(
        name=KNOWLEDGE_COLLECTION,
        embedding_function=embedding_function,
    )

    load_dynamic_cache()

    # Start domain classifier
    domain_classifier = DomainClassifier(model=DOMAIN_CLASSIFY_MODEL, batch_size=DOMAIN_BATCH_SIZE)

    # Run processing
    process_file_with_callback(
        csv_path,
        output_path,
        review_path,
        workers,
        progress_every,
        callback,
    )

    # Shutdown classifier and save cache
    domain_classifier.shutdown(timeout=30)
    save_dynamic_cache()

    # Read findings and compute stats
    findings = []
    severity_counts = {
        "Critical": 0,
        "High": 0,
        "Medium": 0,
        "Low": 0,
    }
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            findings.append(result)
            sev = result.get("severity") or result.get("risk")
            if sev in severity_counts:
                severity_counts[sev] += 1

    return {
        "findings": findings,
        "findings_count": len(findings),
        "severity_counts": severity_counts,
    }


def process_file_with_callback(
    csv_path: Path,
    output_path: Path,
    review_path: Path,
    workers: int,
    progress_every: int,
    callback=None,
) -> None:
    """Same as process_file but calls callback after each finding and progress update."""
    states: dict[tuple[str, str, str | None], EntityState] = {}
    pending = set()
    max_pending = max(workers * 2, 2)

    rows_seen = 0
    chunks_seen = 0
    ai_chunks = 0
    review_chunks = 0
    findings = 0
    started = time.monotonic()

    review_chunks_memory = []

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
            user = row.get("user")
            key = (sensor, source, user)
            state = states.setdefault(key, EntityState())

            for chunk in add_event_to_state(state, row, sensor, source, user):
                chunks_seen += 1
                if chunk["needs_review"]:
                    review_chunks += 1
                    review_chunks_memory.append(chunk)
                    review_fp.write(
                        json.dumps(
                            {
                                "sensor": chunk["sensor"],
                                "source": chunk["source"],
                                "user": chunk.get("user"),
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
                    if callback:
                        callback(
                            f"AI chunk {ai_chunks}: "
                            f"apps={list(chunk['applications'].keys()) or '[]'} "
                            f"tier={chunk['match_tier']} events={chunk['event_count']} "
                            f"bytes={chunk['total_bytes']:,}"
                        )
                    done = submit_completed_chunk(executor, pending, chunk, max_pending)
                    if done:
                        completed = consume_done(done, pending, findings_fp, started)
                        findings += len(completed)
                        if callback:
                            # Send partial findings count
                            callback(f"Findings so far: {findings}", findings=completed)

            rows_seen += 1
            if progress_every and rows_seen % progress_every == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                if callback:
                    callback(
                        f"Processed {rows_seen:,} rows | "
                        f"AI chunks {ai_chunks:,} | findings {findings:,}"
                    )

        # Flush final partial sessions
        for (sensor, source, user), state in states.items():
            if state.events:
                chunk = build_chunk_from_state(state, sensor, source, user)
                chunks_seen += 1

                if chunk["needs_review"]:
                    review_chunks += 1
                    review_chunks_memory.append(chunk)
                    review_fp.write(
                        json.dumps(
                            {
                                "sensor": chunk["sensor"],
                                "source": chunk["source"],
                                "user": chunk.get("user"),
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
                    if callback:
                        callback(
                            f"AI chunk {ai_chunks}: "
                            f"apps={list(chunk['applications'].keys()) or '[]'} "
                            f"tier={chunk['match_tier']} events={chunk['event_count']} "
                            f"bytes={chunk['total_bytes']:,}"
                        )
                    done = submit_completed_chunk(executor, pending, chunk, max_pending)
                    if done:
                        completed = consume_done(done, pending, findings_fp, started)
                        findings += len(completed)
                        if callback:
                            callback(f"Findings so far: {findings}", findings=completed)

        # Drain initial pending analysis jobs
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            completed = consume_done(done, pending, findings_fp, started)
            findings += len(completed)
            if callback:
                callback(f"Findings so far: {findings}", findings=completed)

        # Wait for domain classifier to finish
        if callback:
            callback("Waiting for domain classifier to finish...")
        domain_classifier.shutdown(timeout=30)

        # Re-examine review chunks
        reclassified_chunks = 0
        for chunk in review_chunks_memory:
            new_apps = {}
            for domain in chunk.get("unknown_domains", []):
                entry = cache_get(domain)
                if entry and entry["status"] == "ai":
                    new_apps[entry["app"]] = entry.get("approved", False)

            if new_apps:
                chunk["applications"].update(new_apps)
                chunk["is_ai_related"] = True
                chunk["match_tier"] = "domain"
                chunk["unknown_domains"] = [d for d in chunk["unknown_domains"] if d not in new_apps]
                reclassified_chunks += 1
                if callback:
                    callback(
                        f"Reclassified chunk: apps={list(chunk['applications'].keys())}"
                    )
                future = executor.submit(analyze_chunk, chunk)
                pending.add(future)

        if reclassified_chunks and callback:
            callback(f"Reclassified {reclassified_chunks} chunk(s)")

        # Drain newly submitted jobs
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            completed = consume_done(done, pending, findings_fp, started)
            findings += len(completed)
            if callback:
                callback(f"Findings so far: {findings}", findings=completed)

    # Save dynamic cache at end
    save_dynamic_cache()

# ============================================================
# MAIN
# ============================================================
def main() -> None:
    global DEFAULT_OLLAMA_HOST, DOMAIN_CLASSIFY_MODEL, domain_classifier

    ap = argparse.ArgumentParser(description="Streaming Shadow AI analyzer with background domain learning")
    ap.add_argument("csv", type=Path, help="raw log CSV: timestamp,sensor,source,event")
    ap.add_argument("--workers", type=int, default=DEFAULT_QWEN_WORKERS,
                    help="parallel Qwen workers for final analysis")
    ap.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT_FILE))
    ap.add_argument("--review-output", type=Path, default=Path(DEFAULT_REVIEW_FILE))
    ap.add_argument("--progress-every", type=int, default=10_000)
    ap.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST,
                    help=f"Ollama server host (default: {DEFAULT_OLLAMA_HOST})")
    ap.add_argument("--classify-model", type=str, default=DOMAIN_CLASSIFY_MODEL,
                    help="Small model used for domain classification (default: qwen2.5:7b-instruct)")
    args = ap.parse_args()

    DEFAULT_OLLAMA_HOST = args.ollama_host
    DOMAIN_CLASSIFY_MODEL = args.classify_model

    # Re-initialize Chroma and embedding with the correct host
    global embedding_function, chroma_client, knowledge
    embedding_function = OllamaEmbeddingFunction(
        url=f"http://{DEFAULT_OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    knowledge = chroma_client.get_or_create_collection(
        name=KNOWLEDGE_COLLECTION,
        embedding_function=embedding_function,
    )

    # Pre-flight: verify Ollama models exist
    try:
        client = Client(host=f"http://{DEFAULT_OLLAMA_HOST}:{OLLAMA_PORT}")
        models = [m["name"] for m in client.list()["models"]]
        required = [EMBEDDING_MODEL, ANALYSIS_MODEL, DOMAIN_CLASSIFY_MODEL]
        missing = [m for m in required if m not in models]
        if missing:
            raise SystemExit(f"Missing Ollama models: {missing}. Pull them with `ollama pull <model>`.")
        print("Ollama models verified.")
    except Exception as e:
        print(f"Warning: Could not verify Ollama models: {e}")

    if knowledge.count() == 0:
        raise SystemExit("Knowledge base is EMPTY. Run: python build_knowledge_base.py")

    load_dynamic_cache()

    # Start domain classifier (background thread)
    domain_classifier = DomainClassifier(model=DOMAIN_CLASSIFY_MODEL, batch_size=DOMAIN_BATCH_SIZE)

    workers = max(1, args.workers)

    print(f"Knowledge RAG     : {knowledge.count()} policy chunks")
    print(f"Qwen workers      : {workers}")
    print(f"Chunk size        : {MAX_EVENTS_PER_CHUNK}")
    print(f"Overlap           : {OVERLAP_COUNT} events")
    print(f"Session gap       : {SESSION_GAP_SECONDS}s")
    print(f"Streaming mode    : ON (background domain classification)")
    print(f"AI domain lookup  : indexed suffix matching + dynamic cache")
    print(f"RAG strategy      : cached, one query per application")
    print(f"Domain classify   : model={DOMAIN_CLASSIFY_MODEL}, batch_size={DOMAIN_BATCH_SIZE}, threshold={CONFIDENCE_THRESHOLD}")
    print()

    try:
        process_file(
            args.csv,
            args.output,
            args.review_output,
            workers,
            args.progress_every,
        )
    finally:
        domain_classifier.shutdown(timeout=30)
        save_dynamic_cache()

if __name__ == "__main__":
    main()