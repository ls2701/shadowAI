from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
from threading import local
from collections import defaultdict
from urllib.parse import urlparse
import queue

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from ollama import Client, ResponseError

# ============================================================
# CONFIG
# ============================================================
DEFAULT_OLLAMA_HOST = "192.168.100.100"
OLLAMA_PORT = 11434
CHROMA_PATH = "./chroma_db"
KNOWLEDGE_COLLECTION = "shadow_ai_knowledge"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
ANALYSIS_MODEL = "qwen2.5:14b"
DOMAIN_CLASSIFY_MODEL = "qwen2.5:14b"   # NOTE: must exist on the Ollama server

DEFAULT_OUTPUT_FILE = "findings.jsonl"
DEFAULT_REVIEW_FILE = "needs_review.jsonl"
MIN_HINT_EVENTS_FOR_LLM = 5
MAX_EVENTS_PER_CHUNK = 40
OVERLAP_RATIO = 0.10
SESSION_GAP_SECONDS = 300
DEFAULT_QWEN_WORKERS = 2
DOMAIN_BATCH_SIZE = 8
DOMAIN_CLASSIFIER_WORKERS = 1
DYNAMIC_CACHE_FILE = Path("dynamic_domain_cache.json")
CONFIDENCE_THRESHOLD = 0.95
MIN_DYNAMIC_DOMAIN_OBSERVATIONS = 2
CACHE_SAVE_INTERVAL = 100
UNKNOWN_RECHECK_SECONDS = 7 * 24 * 3600

# ============================================================
# DEBUG HELPERS
# ============================================================
DEBUG = False


def dprint(*args, **kwargs) -> None:
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


def dsection(title: str) -> None:
    if DEBUG:
        print(f"\n{'=' * 60}\n[DEBUG] {title}\n{'=' * 60}")


# ============================================================
# EARLY-SUBMISSION POOL
# ============================================================
_pending_review_lock = threading.Lock()
_pending_review_chunks: dict[str, list[dict]] = defaultdict(list)
_ready_to_submit_queue: "queue.Queue[tuple[dict, str, str, bool]]" = queue.Queue()
_review_submission_callback = None
_review_callback_lock = threading.Lock()


def reset_early_submission_state() -> None:
    global _review_submission_callback
    with _pending_review_lock:
        _pending_review_chunks.clear()
    while True:
        try:
            _ready_to_submit_queue.get_nowait()
        except queue.Empty:
            break
    with _review_callback_lock:
        _review_submission_callback = None


def set_review_submission_callback(cb) -> None:
    global _review_submission_callback
    with _review_callback_lock:
        _review_submission_callback = cb


def get_review_submission_callback():
    with _review_callback_lock:
        return _review_submission_callback


def register_chunk_for_early_submit(chunk: dict, was_analyzed: bool) -> None:
    unknown = chunk.get("unknown_domains") or []
    if not unknown:
        return

    chunk["_was_analyzed_before"] = was_analyzed
    chunk_id = chunk.get("chunk_id")

    with _pending_review_lock:
        for domain in unknown:
            entry = cache_get(domain)
            if entry and entry["status"] == "ai":
                dprint(
                    f"register_chunk_for_early_submit: chunk #{chunk_id} domain={domain} "
                    f"already AI, queueing immediately"
                )
                _ready_to_submit_queue.put(
                    (chunk, domain, entry["app"], entry.get("approved", False))
                )
            else:
                dprint(
                    f"register_chunk_for_early_submit: chunk #{chunk_id} domain={domain} "
                    f"pending (status={entry['status'] if entry else 'MISSING'})"
                )
                _pending_review_chunks[domain].append(chunk)


def _fire_early_submission(domain: str) -> None:
    with _pending_review_lock:
        waiting = _pending_review_chunks.pop(domain, [])
    if not waiting:
        return
    entry = cache_get(domain)
    if not entry or entry["status"] != "ai":
        return
    app_name = entry["app"]
    approved = entry.get("approved", False)
    dprint(
        f"_fire_early_submission: {domain} -> {app_name} "
        f"({len(waiting)} chunk(s) waiting)"
    )
    for chunk in waiting:
        _ready_to_submit_queue.put((chunk, domain, app_name, approved))


# ============================================================
# STATIC AI DOMAIN LIST
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
    "monica.im": ("Monica", False),
    "sider.ai": ("Sider", False),
    "getmerlin.in": ("Merlin", False),
}

AWS_AI_EVENT_SOURCES = (
    "bedrock", "sagemaker", "comprehend", "rekognition", "textract",
    "polly", "lex.amazonaws", "personalize", "kendra", "forecast", "transcribe",
)

AI_TEXT_HINT_PATTERNS = (
    re.compile(r"/backend-api/conversation(?:[/?]|$)", re.I),
    re.compile(r"/v1/chat/completions(?:[/?]|$)", re.I),
    re.compile(r"/v1/messages(?:[/?]|$)", re.I),
    re.compile(r"/v1/responses(?:[/?]|$)", re.I),
    re.compile(r"/v1/generations?(?:[/?]|$)", re.I),
    re.compile(r"generativelanguage\.googleapis\.com", re.I),
    re.compile(r"amazonaws\.com/.*/bedrock(?:[/?]|$)", re.I),
    re.compile(r"anthropic\.claude", re.I),
)

AI_HINT_LABELS = {
    "/backend-api/conversation": "OpenAI/ChatGPT conversation endpoint",
    "/v1/chat/completions": "OpenAI-compatible chat completion endpoint",
    "/v1/messages": "LLM messages endpoint",
    "/v1/responses": "OpenAI responses endpoint",
    "/v1/generations": "Generative model endpoint",
    "generativelanguage.googleapis.com": "Google Generative Language API",
    "amazonaws.com/.../bedrock": "AWS Bedrock API",
    "anthropic.claude": "AWS/Anthropic Claude service identifier",
}

AI_DOMAIN_INDEX = {k.lower().rstrip("."): v for k, v in AI_APPLICATIONS.items()}

# ============================================================
# DOMAIN FILTERING HELPERS (shared by several modules)
# ============================================================
_IP_RE = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}$')
_NON_HOSTNAME_SUFFIXES = (".exe", ".dll", ".log", ".txt", ".zip", ".json", ".xml")

# Domains that cannot be meaningfully classified by a public-LLM classifier.
_NON_CLASSIFIABLE_SUFFIXES = (
    ".in-addr.arpa",
    ".ip6.arpa",
    ".local",
    ".localdomain",
    ".internal",
    ".lan",
    ".home",
    ".corp",
)
_SERVICE_RECORD_RE = re.compile(r"^_[a-z0-9-]+\._(tcp|udp)\.", re.I)


def is_classifiable_domain(domain: str) -> bool:
    """
    Return True only for names a public-LLM classifier can meaningfully judge.

    Rejects:
      - bare IPv4/IPv6 addresses
      - reverse-DNS (.in-addr.arpa, .ip6.arpa)
      - mDNS / private suffixes (.local, .internal, .lan, .home, .corp)
      - DNS SRV records (_sip._tcp.example.com)
      - single-label names
    """
    if not domain:
        return False
    d = domain.lower().strip().rstrip(".")
    if not d:
        return False
    if _IP_RE.match(d):
        return False
    if ":" in d:                        # crude IPv6 rejection
        return False
    if d.endswith(_NON_CLASSIFIABLE_SUFFIXES):
        return False
    if _SERVICE_RECORD_RE.match(d):
        return False
    if "." not in d:
        return False
    return True


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
_cache_lock = threading.RLock()
_dynamic_cache: dict = {}
_cache_save_counter = 0


def load_dynamic_cache() -> None:
    global _dynamic_cache
    _dynamic_cache = {}

    if not DYNAMIC_CACHE_FILE.exists():
        dprint(f"No dynamic cache file at {DYNAMIC_CACHE_FILE} (starting fresh)")
        return

    try:
        with DYNAMIC_CACHE_FILE.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not load dynamic cache: {exc}. Starting fresh.")
        return

    if not isinstance(raw, dict):
        print("Warning: dynamic cache has invalid structure. Starting fresh.")
        return

    invalidated_dynamic_ai = 0

    for domain, v in raw.items():
        if not isinstance(v, dict):
            continue

        status = v.get("status", "unknown")
        app = v.get("app")
        try:
            confidence = float(v.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        static_app, _ = detect_ai_domain(domain)

        if status == "ai" and static_app is None:
            status = "unknown"
            app = None
            confidence = 0.0
            invalidated_dynamic_ai += 1

        _dynamic_cache[domain] = {
            "status": status,
            "app": app,
            "approved": bool(v.get("approved", False)),
            "confidence": confidence,
            "last_attempt": v.get("last_attempt", 0),
        }

    dprint(f"Loaded dynamic cache: {len(_dynamic_cache)} entries from {DYNAMIC_CACHE_FILE}")
    if invalidated_dynamic_ai:
        print(
            f"  [cache] reset {invalidated_dynamic_ai} previously learned AI domain(s) "
            "because older classifier results are no longer trusted"
        )

    statuses = defaultdict(int)
    for v in _dynamic_cache.values():
        statuses[v["status"]] += 1
    dprint(f"  status breakdown: {dict(statuses)}")


def save_dynamic_cache() -> None:
    with _cache_lock:
        try:
            with DYNAMIC_CACHE_FILE.open("w", encoding="utf-8") as f:
                json.dump(_dynamic_cache, f, indent=2, ensure_ascii=False)
            dprint(f"Saved dynamic cache: {len(_dynamic_cache)} entries -> {DYNAMIC_CACHE_FILE}")
        except OSError as exc:
            print(f"Warning: could not save dynamic cache: {exc}")


def cache_get(domain: str) -> dict | None:
    with _cache_lock:
        return _dynamic_cache.get(domain)


def cache_set(domain: str, status: str, app: str | None, approved: bool, confidence: float) -> None:
    global _cache_save_counter
    with _cache_lock:
        prev = _dynamic_cache.get(domain)
        _dynamic_cache[domain] = {
            "status": status,
            "app": app,
            "approved": approved,
            "confidence": confidence,
            "last_attempt": time.time(),
        }
        _cache_save_counter += 1
        if _cache_save_counter % CACHE_SAVE_INTERVAL == 0:
            save_dynamic_cache()
        if prev is None or prev.get("status") != status:
            dprint(f"cache_set: {domain} -> {status} (app={app}, conf={confidence:.2f})")


# ============================================================
# TIMESTAMP PARSING
# ============================================================
_EPOCH_MS_THRESHOLD = 10 ** 12

_STRPTIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%d/%b/%Y:%H:%M:%S %z",
    "%b %d %H:%M:%S",
    "%b  %d %H:%M:%S",
    "%d %b %Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
)

TIMESTAMP_PARSE_FAILURES = 0
_ts_fail_lock = threading.Lock()


def _bump_ts_failure() -> None:
    global TIMESTAMP_PARSE_FAILURES
    with _ts_fail_lock:
        TIMESTAMP_PARSE_FAILURES += 1


def reset_timestamp_failure_counter() -> None:
    global TIMESTAMP_PARSE_FAILURES
    with _ts_fail_lock:
        TIMESTAMP_PARSE_FAILURES = 0


def parse_timestamp(ts: str, default_year: int | None = None) -> float | None:
    if not ts:
        return None
    raw = ts.strip()
    if not raw:
        return None

    if re.fullmatch(r"\d{9,13}(\.\d+)?", raw):
        try:
            value = float(raw)
            if value >= _EPOCH_MS_THRESHOLD:
                value /= 1000.0
            return value
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        pass

    year = default_year or datetime.now().year
    for fmt in _STRPTIME_FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if "%Y" not in fmt and "%y" not in fmt:
            parsed = parsed.replace(year=year)
            now = datetime.now()
            if parsed.month == 12 and now.month == 1:
                parsed = parsed.replace(year=year - 1)
        return parsed.timestamp()

    _bump_ts_failure()
    dprint(f"parse_timestamp: could not parse '{raw}' with any known format")
    return None


# ============================================================
# DOMAIN CLASSIFIER
# ------------------------------------------------------------
# NOTE: BATCH_CLASSIFY_PROMPT contains literal JSON braces, so we MUST NOT
# use str.format() on it. We use a unique sentinel <<DOMAINS>> and .replace().
# ============================================================
BATCH_CLASSIFY_PROMPT = """You are a strict cybersecurity domain verifier.

Your job is NOT to find domains that merely sound AI-related.
For each domain, decide whether it is a REAL, KNOWN public AI product/service.

A domain may be classified AI ONLY when you are highly confident that:
- the domain belongs to a known AI, LLM, generative-AI, AI-assistant, or AI-coding product;
- the AI product is the service actually represented by that domain;
- this is based on known information, not on the words inside the domain.

NON-AI examples:
- normal websites containing "ai", "chat", "bot", "gpt", "llm", "assistant", etc.
- customer-support/live-chat platforms
- generic SaaS/API infrastructure
- analytics, monitoring, CDN, hosting, SSO, login, telemetry, crawler or bot services
- company homepages when the domain is not itself an AI product endpoint

CRITICAL:
- Do NOT infer AI from a substring in the hostname.
- Do NOT assume a parent company is an AI service.
- Do NOT guess.
- If the exact service is uncertain, return UNKNOWN.
- Accuracy is more important than recall.
- Only AI at confidence >= 0.95 is eligible to become confirmed AI.
- app_name MUST be the actual product/service name, not the company name, when known.

Return ONLY valid JSON in this form:
{
  "domain": {
    "classification": "AI" | "NON_AI" | "UNKNOWN",
    "is_ai": true | false,
    "app_name": "Exact known product name" | null,
    "confidence": 0.0,
    "reason": "Short factual reason"
  }
}

Domains:
<<DOMAINS>>
"""


class DomainClassifier:
    def __init__(
        self,
        model: str,
        batch_size: int = DOMAIN_BATCH_SIZE,
        classifier_workers: int = DOMAIN_CLASSIFIER_WORKERS,
        recheck_seconds: float = UNKNOWN_RECHECK_SECONDS,
    ):
        self.model = model
        self.batch_size = batch_size
        self.recheck_seconds = recheck_seconds
        self.pending_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=classifier_workers)
        self.futures = set()
        self._queued = set()
        self._queued_lock = threading.Lock()
        self._shutdown_done = False
        self._disabled = threading.Event()   # set after a fatal ResponseError
        self.worker_thread = threading.Thread(target=self._collector, daemon=True)
        self.worker_thread.start()
        dprint(
            f"DomainClassifier started: model={model}, batch_size={batch_size}, "
            f"workers={classifier_workers}, recheck_seconds={recheck_seconds:.0f}"
        )

    def submit(self, domain: str) -> None:
        if self._disabled.is_set() or self.stop_event.is_set():
            return
        if not is_classifiable_domain(domain):
            dprint(f"submit({domain}) rejected: not classifiable (IP / .local / .arpa / service record)")
            return

        with self._queued_lock:
            if domain in self._queued:
                dprint(f"submit({domain}) skipped: already queued this run")
                return

        entry = cache_get(domain)
        if entry is not None:
            status = entry.get("status", "unknown")
            age = time.time() - entry.get("last_attempt", 0)

            if status == "ai" or status == "non_ai":
                dprint(f"submit({domain}) skipped: cached status={status}")
                return

            # status == "unknown": only retry if the entry is stale.
            if status == "unknown" and age < self.recheck_seconds:
                dprint(
                    f"submit({domain}) skipped: cached 'unknown' only "
                    f"{age / 3600:.1f}h old (TTL={self.recheck_seconds / 3600:.1f}h)"
                )
                return

            dprint(f"submit({domain}): stale 'unknown' entry (age={age / 3600:.1f}h) -> requeueing")

        dprint(f"submit({domain}) -> queued for classification")
        with self._queued_lock:
            self._queued.add(domain)
        self.pending_queue.put(domain)

    def _collector(self) -> None:
        while not self.stop_event.is_set():
            if self._disabled.is_set():
                # Drain anything pending silently and exit the loop.
                try:
                    while True:
                        self.pending_queue.get_nowait()
                except queue.Empty:
                    pass
                return

            domains = []
            try:
                while len(domains) < self.batch_size:
                    try:
                        domain = self.pending_queue.get(timeout=0.5)
                        domains.append(domain)
                    except queue.Empty:
                        break
                if domains:
                    dprint(f"collector: dispatching batch {domains}")
                    future = self.executor.submit(self._classify_batch, domains)
                    self.futures.add(future)
                    self.futures = {f for f in self.futures if not f.done()}
            except Exception as e:
                print(f"[domain classifier collector error] {e}")
                time.sleep(1)

        # Drain leftovers during shutdown
        while not self.pending_queue.empty():
            domains = []
            try:
                domains.append(self.pending_queue.get_nowait())
                while len(domains) < self.batch_size:
                    try:
                        domains.append(self.pending_queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                pass
            if domains:
                dprint(f"collector(shutdown): dispatching leftover batch {domains}")
                future = self.executor.submit(self._classify_batch, domains)
                self.futures.add(future)

        for future in list(self.futures):
            try:
                future.result()
            except Exception as exc:
                dprint(f"collector: future raised during shutdown: {exc}")

    def _classify_batch(self, domains: list[str]) -> None:
        # Skip work entirely if we've already given up on the classifier.
        if self._disabled.is_set():
            return

        dprint(f"_classify_batch: sending {len(domains)} domains to {self.model}")
        dprint(f"  domains: {domains}")
        try:
            client = get_ollama_client()
            prompt = BATCH_CLASSIFY_PROMPT.replace("<<DOMAINS>>", "\n".join(domains))
            resp = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0},
            )
            raw = resp["message"]["content"].strip()
            if "<think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            dprint(f"  raw LLM response: {raw[:500]}")
            result = json.loads(raw)
            if not isinstance(result, dict):
                result = {}
            dprint(f"  parsed result keys: {list(result.keys())}")

            for d in domains:
                res = result.get(d, {})
                if not isinstance(res, dict):
                    res = {}
                classification = str(res.get("classification", "UNKNOWN")).upper()
                try:
                    confidence = float(res.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                app_name = res.get("app_name")
                reason = res.get("reason", "")

                dprint(
                    f"  {d}: classification={classification} "
                    f"confidence={confidence:.2f} app={app_name} reason={reason}"
                )

                if (
                    classification == "AI"
                    and confidence >= CONFIDENCE_THRESHOLD
                    and isinstance(app_name, str)
                    and app_name.strip()
                ):
                    cache_set(d, "ai", app_name.strip(), False, confidence)
                    print(f"  [classify] AI-CONFIRMED: {d} -> {app_name.strip()} (conf={confidence:.2f})")
                elif classification == "NON_AI" and confidence >= CONFIDENCE_THRESHOLD:
                    cache_set(d, "non_ai", None, False, confidence)
                    print(f"  [classify] Non-AI: {d} (conf={confidence:.2f})")
                else:
                    cache_set(d, "unknown", None, False, confidence)
                    print(f"  [classify] UNKNOWN: {d} (model={classification}, conf={confidence:.2f})")

                # ── FIX 5: emit live "classify" event for the dashboard ──
                entry = cache_get(d) or {}
                _emit(
                    "classify",
                    domain=d,
                    status=entry.get("status", "unknown"),
                    app=entry.get("app"),
                    confidence=float(entry.get("confidence", 0.0) or 0.0),
                )

            cb = get_review_submission_callback()
            if cb is not None:
                for d in domains:
                    entry = cache_get(d)
                    if entry and entry["status"] == "ai":
                        try:
                            cb(d)
                        except Exception as cb_exc:
                            dprint(f"early-submission callback failed for {d}: {cb_exc}")

        except ResponseError as e:
            # Server-side error (model missing, OOM, etc). Do NOT poison the
            # cache — leave domains as-is so they can be retried on the next
            # run once the server is healthy. Disable the classifier entirely
            # to avoid hammering the server with the same broken request.
            print(f"  [classify error] {type(e).__name__}: {e}")
            print(
                f"  [classify] DISABLING classifier for this run "
                f"(model={self.model!r} may be missing or the server is unhealthy). "
                f"{len(domains)} domain(s) left unresolved."
            )
            self._disabled.set()
            return
        except (ConnectionError, TimeoutError, OSError) as e:
            # Transient network errors — do not poison the cache, do not
            # disable the classifier. Just skip and let the next batch try.
            print(f"  [classify error] transient: {type(e).__name__}: {e}")
            return
        except Exception as e:
            import traceback
            print(f"  [classify error] {type(e).__name__}: {e}")
            traceback.print_exc()
            for d in domains:
                cache_set(d, "unknown", None, False, 0.0)

    def shutdown(self, timeout: float = 30) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        dprint("DomainClassifier.shutdown() called")
        self.stop_event.set()
        self.worker_thread.join(timeout=timeout)
        self.executor.shutdown(wait=True, cancel_futures=False)
        dprint("DomainClassifier shutdown complete")


domain_classifier: DomainClassifier | None = None

_domain_observation_counts: dict[str, int] = defaultdict(int)


# ============================================================
# LIVE EVENT STREAM (consumed by backend_api.py / the dashboard)
# ============================================================
PIPELINE_EVENTS_FILE = Path("pipeline_events.jsonl")
_event_lock = threading.Lock()


def _emit(kind: str, **fields) -> None:
    """Append one JSONL event for the live dashboard to pick up."""
    with _event_lock:
        try:
            with PIPELINE_EVENTS_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"ts": time.time(), "kind": kind, **fields},
                    ensure_ascii=False,
                ) + "\n")
        except OSError:
            pass


def reset_pipeline_events() -> None:
    with _event_lock:
        try:
            PIPELINE_EVENTS_FILE.write_text("", encoding="utf-8")
        except OSError:
            pass


# ============================================================
# FAST AI DETECTION
# ============================================================
def scan_text_hints(raw_text: str) -> list[str]:
    if not raw_text:
        return []

    hits = []
    for pattern in AI_TEXT_HINT_PATTERNS:
        if pattern.search(raw_text):
            label = pattern.pattern
            if "/backend-api/conversation" in label:
                hits.append(AI_HINT_LABELS["/backend-api/conversation"])
            elif "/v1/chat/completions" in label:
                hits.append(AI_HINT_LABELS["/v1/chat/completions"])
            elif "/v1/messages" in label:
                hits.append(AI_HINT_LABELS["/v1/messages"])
            elif "/v1/responses" in label:
                hits.append(AI_HINT_LABELS["/v1/responses"])
            elif "/v1/generation" in label:
                hits.append(AI_HINT_LABELS["/v1/generations"])
            elif "generativelanguage" in label:
                hits.append(AI_HINT_LABELS["generativelanguage.googleapis.com"])
            elif "amazonaws" in label and "bedrock" in label:
                hits.append(AI_HINT_LABELS["amazonaws.com/.../bedrock"])
            elif "anthropic" in label:
                hits.append(AI_HINT_LABELS["anthropic.claude"])

    return sorted(set(hits))


def detect_ai_domain(domain: str) -> tuple[str | None, bool]:
    if not domain:
        return None, False
    d = domain.lower().strip().rstrip(".")
    if _IP_RE.match(d):
        return None, False
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
_KV_QUOTED = re.compile(r'([A-Za-z0-9_]+)="([^"]*)"')
_KV_PLAIN = re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')
CEF_KV_RE = re.compile(r'(\w+)=((?:(?!\s+\w+=).)*)')
FQDN_GUESS_RE = re.compile(r'\b(?:(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.)+(?:[A-Za-z]{2,24})\b')

# ── FIX 9: unused tuple constants removed (FIREWALL_SENSOR_HINTS / WINDOWS_SENSOR_HINTS)

# ============================================================
# AUTHORITATIVE SENSOR/SOURCE → LOG-FORMAT MAP
# ------------------------------------------------------------
# Dispatch on the CSV's own sensor/source columns first. Content-based
# sniffing is only used as a fallback so that a JSON event whose payload
# happens to contain "device_name=" (or similar) cannot be routed into the
# firewall branch by mistake.
# ============================================================
_SENSOR_FORMAT_MAP: dict[str, str] = {
    # Firewalls / NGFW
    "fortigate": "firewall_kv",
    "fortinet": "firewall_kv",
    "paloalto": "firewall_kv",
    "pan-os": "firewall_kv",
    "panos": "firewall_kv",
    "checkpoint": "firewall_kv",
    "sonicwall": "firewall_kv",
    "cisco-asa": "firewall_kv",
    "asa": "firewall_kv",

    # Windows / Sysmon / DNS debug
    "sysmon": "windows_dns_packet",
    "mswineventlog": "windows_dns_packet",
    "win-dns": "windows_dns_packet",
    "dns-debug": "windows_dns_packet",
    "dns-server": "windows_dns_packet",

    # Structured cloud
    "aws": "json_structured",
    "guardduty": "json_structured",
    "cloudtrail": "json_structured",
    "vpc-flow": "json_structured",
    "route53": "json_structured",

    # CEF emitters
    "cef": "cef",
}
DOMAIN_KV_KEYS = (
    "dstname", "domain", "dest_hostname", "dsthost", "host", "hostname",
    "sni", "server_name", "dst", "dest", "url", "uri", "request", "requesturl",
)
CEF_DOMAIN_KEYS = ("dhost", "dvchost", "shost", "request", "requesturl", "dst")
USER_KV_KEYS = ("user", "username", "account", "src_user", "srcuser", "duser", "suser")
CEF_USER_KEYS = ("duser", "suser", "src_user")


def fast_dns_decode(text: str) -> str | None:
    m = DNS_PATTERN.search(text)
    if not m:
        return None
    labels = DNS_LABEL_PATTERN.findall(m.group(1))
    return ".".join(labels).lower() if labels else None


def extract_kv(event: str, key: str) -> str | None:
    m = re.search(rf'{re.escape(key)}="([^"]*)"', event)
    if m:
        return m.group(1)
    m = re.search(rf"{re.escape(key)}=([^\s]+)", event)
    if m:
        return m.group(1)
    return None


def first_str(value) -> str | None:
    return value if isinstance(value, str) and value else None


# ── FIX 7: token-based matching for the sensor/source → format map
def _tokenise(s: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", s.lower()) if tok}


def classify_log_format(sensor: str, source: str, low_event: str) -> str:
    """
    Dispatch order:
      1. Unambiguous structural prefixes (JSON '{', CEF 'CEF:')
      2. Authoritative sensor/source hints from the CSV columns (token match)
      3. Content-based heuristics (firewall KV, Windows DNS, generic KV)
      4. Fallback: unstructured
    """
    stripped = low_event.lstrip()

    # 1) Unambiguous structural markers — always win
    if stripped.startswith("cef:"):
        return "cef"
    if stripped.startswith("{"):
        return "json_structured"

    # 2) Authoritative sensor/source hints — token-based to avoid
    #    "asa" matching "NASA", "aws" matching "DrawSomething", etc.
    tokens = _tokenise(f"{sensor} {source}")
    for hint, fmt in _SENSOR_FORMAT_MAP.items():
        hint_tokens = _tokenise(hint)
        if hint_tokens and hint_tokens.issubset(tokens):
            return fmt

    # 3) Content-based fallback (only when the sensor doesn't claim a format)
    if "device_name=" in low_event or "id=firewall" in low_event:
        return "firewall_kv"
    if "mswineventlog" in low_event and "packet" in low_event:
        return "windows_dns_packet"
    if "=" in low_event:
        return "generic_kv"

    # 4) Fallback
    return "unstructured"


def parse_cef(event: str) -> dict:
    body = event
    parts = event.split("|", 7)
    if len(parts) >= 8:
        body = parts[7]
    return {m.group(1): m.group(2).strip() for m in CEF_KV_RE.finditer(body)}


def parse_generic_kv(event: str) -> dict:
    out: dict[str, str] = {}
    for m in _KV_QUOTED.finditer(event):
        out.setdefault(m.group(1), m.group(2))
    for m in _KV_PLAIN.finditer(event):
        out.setdefault(m.group(1), m.group(2))
    return out


def _hostname_from_maybe_url(value: str) -> str | None:
    if not value:
        return None
    v = value.strip()
    if not v:
        return None

    if "://" in v:
        try:
            parsed = urlparse(v)
            host = parsed.hostname
            return host.lower() if host else None
        except ValueError:
            return None

    if v.startswith("["):
        end = v.find("]")
        if end > 0:
            v = v[1:end]
        else:
            return None

    if ":" in v:
        parts = v.split(":")
        if _IP_RE.match(parts[0]):
            return None
        if "." in parts[0] and all(c.isalnum() or c in ".-" for c in parts[0]):
            v = parts[0]
        else:
            return None

    if _IP_RE.match(v):
        return None

    if "." not in v:
        return None
    if v.endswith(_NON_HOSTNAME_SUFFIXES):
        return None
    if not all(c.isalnum() or c in ".-" for c in v):
        return None

    return v.lower()


def extract_domain_from_kv(kv: dict, keys: tuple[str, ...] = DOMAIN_KV_KEYS) -> str | None:
    for key in keys:
        raw = kv.get(key)
        if not raw:
            continue
        host = _hostname_from_maybe_url(raw)
        if host:
            return host
    return None


def extract_user_from_kv(kv: dict, keys: tuple[str, ...] = USER_KV_KEYS) -> str | None:
    for key in keys:
        if kv.get(key):
            return kv[key]
    return None


def extract_bytes_from_kv(kv: dict) -> int:
    total = 0
    found = False
    for key in ("bytes_sent", "sent", "out_bytes", "bytesout", "out"):
        if key in kv:
            try:
                total += int(kv[key])
                found = True
            except (ValueError, TypeError):
                pass
    for key in ("bytes_recv", "rcvd", "in_bytes", "bytesin", "in"):
        if key in kv:
            try:
                total += int(kv[key])
                found = True
            except (ValueError, TypeError):
                pass
    if not found and "bytes" in kv:
        try:
            total = int(kv["bytes"])
        except (ValueError, TypeError):
            total = 0
    return total


def guess_domain_from_text(text: str) -> str | None:
    for candidate in FQDN_GUESS_RE.findall(text):
        low = candidate.lower()
        if _IP_RE.match(low) or low.count(".") == 0:
            continue
        if low.endswith(_NON_HOSTNAME_SUFFIXES):
            continue
        return low
    return None


def normalize_row(sensor: str, source: str, ts: str, event: str, log_year: int | None = None) -> dict:
    out = {
        "timestamp": ts,
        "ts_epoch": None,
        "sensor": sensor,
        "source": source,
        "user": None,
        "hostname": None,
        "domain": None,
        "domain_confidence": "high",
        "bytes": 0,
        "text_hints": [],
        "ai_app": None,
        "ai_approved": False,
        "ai_domain_confidence": 0.0,
        "log_format": "unstructured",
    }

    out["ts_epoch"] = parse_timestamp(ts, default_year=log_year)

    e = event or ""
    low = e.lower()
    fmt = classify_log_format(sensor, source, low)
    out["log_format"] = fmt

    try:
        if fmt == "firewall_kv":
            out["domain"] = (extract_kv(e, "dstname") or "").lower() or None
            out["user"] = extract_kv(e, "user") or extract_kv(e, "srcuser") or extract_kv(e, "duser")
            try:
                out["bytes"] = int(extract_kv(e, "sent") or 0) + int(extract_kv(e, "rcvd") or 0)
            except (ValueError, TypeError):
                out["bytes"] = 0
            if not out["domain"] or not out["user"] or not out["bytes"]:
                kv = parse_generic_kv(e)
                if not out["domain"]:
                    out["domain"] = extract_domain_from_kv(kv)
                if not out["user"]:
                    out["user"] = extract_user_from_kv(kv)
                if not out["bytes"]:
                    out["bytes"] = extract_bytes_from_kv(kv)

        elif fmt == "windows_dns_packet":
            out["domain"] = fast_dns_decode(e)
            if not out["domain"]:
                out["domain"] = guess_domain_from_text(e)
                out["domain_confidence"] = "low"

        elif fmt == "json_structured":
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

                dest = j.get("destination") if isinstance(j.get("destination"), dict) else {}
                out["domain"] = (
                    first_str(j.get("destinationHostname"))
                    or first_str(dest.get("hostname"))
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

        elif fmt == "cef":
            kv = parse_cef(e)
            out["domain"] = extract_domain_from_kv(kv, keys=CEF_DOMAIN_KEYS)
            out["user"] = extract_user_from_kv(kv, keys=CEF_USER_KEYS)
            out["bytes"] = extract_bytes_from_kv(kv)

        elif fmt == "generic_kv":
            kv = parse_generic_kv(e)
            out["domain"] = extract_domain_from_kv(kv)
            out["user"] = extract_user_from_kv(kv)
            out["bytes"] = extract_bytes_from_kv(kv)

        else:  # unstructured
            out["domain"] = guess_domain_from_text(e)
            out["domain_confidence"] = "low" if out["domain"] else "none"

    except Exception as exc:
        dprint(f"normalize_row: error parsing fmt={fmt} sensor={sensor} source={source}: {exc}")

    out["text_hints"].extend(scan_text_hints(e))
    out["text_hints"] = sorted(set(out["text_hints"]))

    # Static / dynamic AI matching
    app, approved = (None, False)
    domain_conf = 0.0

    domain = out["domain"]
    if domain and out["domain_confidence"] != "low" and is_classifiable_domain(domain):
        app, approved = detect_ai_domain(domain)
        if app is not None:
            domain_conf = 1.0
        else:
            _domain_observation_counts[domain] += 1

            entry = cache_get(domain)
            if entry is not None and entry["status"] == "ai":
                cached_conf = float(entry.get("confidence", 0.0))
                if cached_conf >= CONFIDENCE_THRESHOLD and entry.get("app"):
                    app = entry["app"]
                    approved = bool(entry.get("approved", False))
                    domain_conf = cached_conf
            elif domain_classifier is not None:
                if _domain_observation_counts[domain] >= MIN_DYNAMIC_DOMAIN_OBSERVATIONS:
                    domain_classifier.submit(domain)

    out["ai_app"] = app
    out["ai_approved"] = approved
    out["ai_domain_confidence"] = domain_conf
    return out


# ============================================================
# STREAMING CHUNKING
# ============================================================
_chunk_id_lock = threading.Lock()
_chunk_id_counter = itertools.count(1)


def next_chunk_id() -> int:
    with _chunk_id_lock:
        return next(_chunk_id_counter)


class EntityState:
    __slots__ = (
        "events", "previous_ts", "saw_unparsable_ts",
        "candidate_apps", "candidate_app_conf",
        "candidate_hints", "unknown_domains", "total_bytes", "log_formats",
    )

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.previous_ts: float | None = None
        self.saw_unparsable_ts: bool = False
        self.candidate_apps: dict[str, bool] = {}
        self.candidate_app_conf: dict[str, float] = {}
        self.candidate_hints: set[str] = set()
        self.unknown_domains: set[str] = set()
        self.total_bytes = 0
        self.log_formats: dict[str, int] = defaultdict(int)

    def reset_after_chunk(self, overlap: int) -> None:
        self.events = self.events[-overlap:] if overlap else []
        self.candidate_apps = {e["ai_app"]: e["ai_approved"] for e in self.events if e.get("ai_app")}
        self.candidate_app_conf = {}
        for e in self.events:
            app = e.get("ai_app")
            if not app:
                continue
            conf = float(e.get("ai_domain_confidence", 0.0))
            prev = self.candidate_app_conf.get(app, 0.0)
            if conf > prev:
                self.candidate_app_conf[app] = conf
        self.candidate_hints = {h for e in self.events for h in (e.get("text_hints") or [])}
        self.unknown_domains = {
            e["domain"] for e in self.events
            if e.get("domain") and e.get("domain_confidence") != "low"
            and not e.get("ai_app") and not e.get("text_hints")
            and is_classifiable_domain(e["domain"])
        }
        self.total_bytes = sum(e.get("bytes", 0) for e in self.events)
        fmt_counts: dict[str, int] = defaultdict(int)
        for e in self.events:
            fmt_counts[e.get("log_format", "unstructured")] += 1
        self.log_formats = fmt_counts
        # After reset, previous_ts reflects the last event in the overlap window.
        last_ts = None
        for e in reversed(self.events):
            if e.get("ts_epoch") is not None:
                last_ts = e["ts_epoch"]
                break
        self.previous_ts = last_ts
        self.saw_unparsable_ts = False


def add_event_to_state(state: EntityState, row: dict, sensor: str, source: str, user: str | None):
    """
    Yields completed chunks. Session-break rules:
      * Break when the gap between successive parsable timestamps exceeds
        SESSION_GAP_SECONDS.
      * Break conservatively when an unparsable timestamp was seen since the
        last parsable event — we cannot prove continuity, so we do not merge.
      * Timestamps that go backwards within a session are logged but do not
        force a break (Phase 2 already sorts by (sensor, source, user, ts)).
    """
    ts = row.get("ts_epoch")
    should_break = False

    if state.events and state.previous_ts is not None:
        if state.saw_unparsable_ts:
            should_break = True
            dprint(
                f"add_event_to_state: forcing session break for key=({sensor},{source},{user}) "
                f"— unparsable timestamp(s) since last parsable event"
            )
        elif ts is not None and (ts - state.previous_ts) > SESSION_GAP_SECONDS:
            should_break = True
        elif ts is not None and ts < state.previous_ts:
            dprint(
                f"add_event_to_state: non-increasing ts within session "
                f"(prev={state.previous_ts}, cur={ts}) for key=({sensor},{source},{user})"
            )

    if should_break:
        yield build_chunk_from_state(state, sensor, source, user)
        state.__init__()

    state.events.append(row)
    state.total_bytes += row.get("bytes", 0)
    state.log_formats[row.get("log_format", "unstructured")] += 1

    if row.get("ai_app"):
        state.candidate_apps[row["ai_app"]] = row["ai_approved"]
        conf = float(row.get("ai_domain_confidence", 0.0))
        prev = state.candidate_app_conf.get(row["ai_app"], 0.0)
        if conf > prev:
            state.candidate_app_conf[row["ai_app"]] = conf

    for h in row.get("text_hints") or []:
        state.candidate_hints.add(h)

    if (row.get("domain") and row.get("domain_confidence") != "low"
            and not row.get("ai_app") and not row.get("text_hints")
            and is_classifiable_domain(row["domain"])):
        state.unknown_domains.add(row["domain"])

    if ts is not None:
        state.previous_ts = ts
        state.saw_unparsable_ts = False
    else:
        state.saw_unparsable_ts = True

    if len(state.events) >= MAX_EVENTS_PER_CHUNK:
        yield build_chunk_from_state(state, sensor, source, user)
        state.reset_after_chunk(OVERLAP_COUNT)


OVERLAP_COUNT = max(1, math.floor(MAX_EVENTS_PER_CHUNK * OVERLAP_RATIO))


def build_chunk_from_state(state: EntityState, sensor: str, source: str, user: str | None = None) -> dict:
    events = state.events
    apps = dict(state.candidate_apps)
    hints = sorted(state.candidate_hints)
    unknown = sorted(state.unknown_domains)
    is_ai_related = bool(apps or hints)
    tier = "domain" if apps else ("text_hint" if hints else "none")

    return {
        "chunk_id": next_chunk_id(),
        "sensor": sensor,
        "source": source,
        "user": user,
        "start_ts": events[0]["timestamp"],
        "end_ts": events[-1]["timestamp"],
        "start_ts_epoch": events[0].get("ts_epoch"),
        "end_ts_epoch": events[-1].get("ts_epoch"),
        "event_count": len(events),
        "events": events,
        "applications": apps,
        "application_confidences": dict(state.candidate_app_conf),
        "text_hints": hints,
        "unknown_domains": unknown,
        "log_formats": dict(state.log_formats),
        "total_bytes": state.total_bytes,
        "is_ai_related": is_ai_related,
        "needs_review": bool(unknown) and not is_ai_related,
        "match_tier": tier,
    }

# ── FIX 2: the duplicate (older, buggy) def add_event_to_state that used to
#          sit right here has been DELETED. The correct version — with
#          saw_unparsable_ts handling — is the one above.


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

    try:
        res = knowledge.query(query_texts=[query], n_results=8)
    except Exception as exc:
        dprint(f"retrieve_policy_for_app({app}): query failed: {exc}")
        return ""

    docs = res.get("documents", [[]])[0] if res.get("documents") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []

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
    with _policy_lock:
        _policy_cache[app] = context
    return context


def retrieve_policy_for_apps(apps: list[str]) -> str:
    if apps:
        contexts = [retrieve_policy_for_app(app) for app in apps]
        return "\n\n".join(c for c in contexts if c)
    return retrieve_policy_for_app("generic_shadow_ai")


# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """You are a Shadow AI Security Analyst.

You receive a chunk of grouped network/endpoint events plus retrieved policy.

STRICT RULES:
1. The ONLY AI applications you may reference are the ones listed in
   `applications_seen`. Never name, mention, or imply any application not in
   that list. If a different AI tool seems relevant to the traffic, say so
   generically as "an unlisted AI service" instead of naming it.
2. Do NOT invent file contents, uploads, prompts, source code, or user intent.
   The events summary contains only domain names, event counts, byte counts,
   first/last seen timestamps, and log formats. Anything else you claim must
   come from that summary or the retrieved policy.
3. The `evidence` field must cite only facts present in the events summary
   (domains, counts, bytes, time window, log formats). No narrative.
4. Do not treat ordinary web browsing, customer-support ticketing, or
   unrelated bot/crawler traffic as Shadow AI.

match_tier rules:
- domain: a known AI/chatbot domain matched exactly or as a valid subdomain.
- text_hint: only a generic hint matched; verify from context and lower
  confidence when ambiguous.
- none: no AI evidence.

A finding qualifies as Shadow AI only when the evidence indicates communication
with an external conversational AI, AI assistant, or generative-AI-backed tool.

Return ONLY this JSON, with all keys present and no extras:
{
  "classification": "Shadow AI Usage" | "Possible Shadow AI Usage" | "Not Shadow AI",
  "severity": "Critical" | "High" | "Medium" | "Low",
  "confidence": 0.85,
  "application": "exact name from applications_seen, or null",
  "reason": "one or two sentences grounded in the events summary",
  "evidence": "concrete facts from the events summary only",
  "policy": "policy name from retrieved policy",
  "recommended_action": "concrete next step"
}

`confidence` MUST be a JSON number between 0.0 and 1.0, not a string.
"""


def summarize_events_for_llm(events: list[dict]) -> list[dict]:
    agg = defaultdict(lambda: {
        "count": 0,
        "total_bytes": 0,
        "first_seen": None,
        "first_seen_epoch": float("inf"),
        "last_seen": None,
        "last_seen_epoch": float("-inf"),
    })
    for e in events:
        domain = e.get("domain") or "unknown"
        bucket = agg[domain]
        bucket["count"] += 1
        bucket["total_bytes"] += e.get("bytes", 0)

        ts_str = e.get("timestamp")
        ts_epoch = e.get("ts_epoch")
        if ts_epoch is None:
            if bucket["first_seen"] is None or (ts_str and ts_str < bucket["first_seen"]):
                bucket["first_seen"] = ts_str
            if bucket["last_seen"] is None or (ts_str and ts_str > bucket["last_seen"]):
                bucket["last_seen"] = ts_str
        else:
            if ts_epoch < bucket["first_seen_epoch"]:
                bucket["first_seen_epoch"] = ts_epoch
                bucket["first_seen"] = ts_str
            if ts_epoch > bucket["last_seen_epoch"]:
                bucket["last_seen_epoch"] = ts_epoch
                bucket["last_seen"] = ts_str

    result = []
    for domain, data in agg.items():
        result.append({
            "domain": domain,
            "event_count": data["count"],
            "total_bytes": data["total_bytes"],
            "first_seen": data["first_seen"],
            "last_seen": data["last_seen"],
        })
    return result


_CONF_STRING_MAP = {
    "very high": 0.95, "high": 0.9,
    "medium": 0.6, "moderate": 0.6,
    "low": 0.3, "very low": 0.15,
}


def _coerce_confidence(value) -> float:
    if value is None:
        return 0.5
    if isinstance(value, (int, float)):
        try:
            f = float(value)
            if f > 1.0:
                f = f / 100.0
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            return 0.5
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _CONF_STRING_MAP:
            return _CONF_STRING_MAP[s]
        try:
            f = float(s)
            if f > 1.0:
                f = f / 100.0
            return max(0.0, min(1.0, f))
        except ValueError:
            return 0.5
    return 0.5


def build_evidence(snapshot: dict) -> dict:
    """
    Build the structured evidence block that ships with every finding.
    Uses only a snapshot of the chunk (safe under concurrency).
    """
    events = snapshot["events"]
    apps = snapshot["applications"]
    app_confs = snapshot["application_confidences"]

    # Per-domain aggregate
    domains_agg: dict[str, dict] = {}
    for e in events:
        d = e.get("domain") or "(none)"
        b = domains_agg.setdefault(d, {
            "event_count": 0,
            "total_bytes": 0,
            "first_seen": None,
            "last_seen": None,
            "users": set(),
        })
        b["event_count"] += 1
        b["total_bytes"] += int(e.get("bytes") or 0)
        ts_str = e.get("timestamp")
        if ts_str:
            if b["first_seen"] is None or ts_str < b["first_seen"]:
                b["first_seen"] = ts_str
            if b["last_seen"] is None or ts_str > b["last_seen"]:
                b["last_seen"] = ts_str
        if e.get("user"):
            b["users"].add(e["user"])

    domains_list = []
    for d, b in domains_agg.items():
        domains_list.append({
            "domain": d,
            "event_count": b["event_count"],
            "total_bytes": b["total_bytes"],
            "first_seen": b["first_seen"],
            "last_seen": b["last_seen"],
            "users": sorted(b["users"]),
        })
    domains_list.sort(key=lambda x: (-x["total_bytes"], x["domain"]))

    # Matched applications with confidence + provenance
    matched_apps = {}
    for app, approved in apps.items():
        conf = float(app_confs.get(app, 0.0))
        matched_apps[app] = {
            "approved": bool(approved),
            "confidence": conf,
            "source": "static_allowlist" if conf >= 1.0 else "dynamic_classifier",
        }

    # Sample events — up to 5 highest-byte rows (representative of exfil risk)
    sample_events = []
    for e in sorted(events, key=lambda x: -(x.get("bytes") or 0))[:5]:
        sample_events.append({
            "timestamp": e.get("timestamp"),
            "domain": e.get("domain"),
            "user": e.get("user"),
            "bytes": e.get("bytes"),
            "log_format": e.get("log_format"),
            "text_hints": list(e.get("text_hints") or []),
        })

    return {
        "chunk_id": snapshot["chunk_id"],
        "sensor": snapshot["sensor"],
        "source": snapshot["source"],
        "user": snapshot.get("user"),
        "start_ts": snapshot["start_ts"],
        "end_ts": snapshot["end_ts"],
        "event_count": snapshot["event_count"],
        "total_bytes": snapshot["total_bytes"],
        "match_tier": snapshot["match_tier"],
        "log_formats": snapshot["log_formats"],
        "matched_applications": matched_apps,
        "text_hints": list(snapshot.get("text_hints", [])),
        "domains": domains_list,
        "sample_events": sample_events,
        "reanalysis": bool(snapshot.get("_reanalysis")),
        "reanalysis_reason": snapshot.get("_reanalysis_reason"),
        "early_submit": bool(snapshot.get("_early_submitted")),
    }


def analyze_chunk(chunk: dict) -> dict:
    """
    Runs the chunk through ANALYSIS_MODEL and sanitizes the output.
    Takes a snapshot of all chunk fields up front so concurrent mutation by
    the early-submission path cannot cause inconsistent reads.
    """
    analysis_started = time.monotonic()
    result: dict = {}

    # ---- SNAPSHOT ----
    chunk_id = chunk.get("chunk_id")
    sensor = chunk["sensor"]
    source = chunk["source"]
    user = chunk.get("user")
    start_ts = chunk["start_ts"]
    end_ts = chunk["end_ts"]
    event_count = chunk["event_count"]
    match_tier = chunk["match_tier"]
    log_formats = dict(chunk.get("log_formats", {}))
    applications = dict(chunk["applications"])
    application_confidences = dict(chunk.get("application_confidences", {}))
    text_hints = list(chunk.get("text_hints", []))
    total_bytes = chunk["total_bytes"]
    events = chunk["events"]
    is_reanalysis = bool(chunk.get("_reanalysis", False))
    reanalysis_reason = chunk.get("_reanalysis_reason", "")
    is_early_submit = bool(chunk.get("_early_submitted", False))
    # ------------------

    try:
        events_summary = summarize_events_for_llm(events)
        rag_context = retrieve_policy_for_apps(list(applications.keys()))
        apps = list(applications.keys())

        reanalysis_note = ""
        if is_reanalysis:
            reanalysis_note = (
                "\nNOTE: This chunk is being RE-analyzed with stronger evidence "
                f"than the first pass. {reanalysis_reason}\n"
            )

        prompt = f"""CHUNK:
chunk_id={chunk_id}
sensor={sensor}
source={source}
user={user or 'unknown'}
start={start_ts}
end={end_ts}
event_count={event_count}
match_tier={match_tier}
log_formats={log_formats}
applications_seen={apps}
application_confidences={application_confidences}
text_pattern_hints={text_hints}
total_bytes={total_bytes}
{reanalysis_note}
events_summary={json.dumps(events_summary, ensure_ascii=False)}

RETRIEVED SECURITY POLICY:
{rag_context or '(no policy retrieved — flag low confidence)'}

Return ONLY the JSON object described in the system prompt. `confidence` must
be a numeric value between 0.0 and 1.0.
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
        result = json.loads(raw)
        if not isinstance(result, dict):
            result = {"classification": None, "confidence": 0.0, "reason": "model returned non-object"}

    except (ResponseError, json.JSONDecodeError, KeyError, ConnectionError) as e:
        print(f"  [analysis error] {e}. Returning fallback.")
        result = {
            "classification": None,
            "confidence": 0.0,
            "reason": f"LLM error: {e}",
            "parse_error": str(e)[:300],
        }
    except Exception as e:
        print(f"  [analysis error - unexpected] {e}. Returning fallback.")
        result = {
            "classification": None,
            "confidence": 0.0,
            "reason": f"Unexpected error: {e}",
            "parse_error": str(e)[:300],
        }

    # ---------------- SANITIZE ----------------
    result["confidence"] = _coerce_confidence(result.get("confidence"))

    allowed_apps = set(applications.keys())
    chosen = result.get("application")
    if isinstance(chosen, str):
        chosen = chosen.strip()
        if chosen and chosen not in allowed_apps:
            result["_rejected_application"] = chosen
            result["application"] = None
            result["confidence"] = min(result["confidence"], 0.5)
    elif chosen is not None and chosen not in allowed_apps:
        result["_rejected_application"] = str(chosen)
        result["application"] = None
        result["confidence"] = min(result["confidence"], 0.5)

    # ── FIX 1: this block was accidentally nested inside the elif above.
    #           It must run for EVERY path so the UI gets numeric confidence
    #           metadata even when the LLM returned a valid application or null.
    result["_application_confidences"] = dict(application_confidences)
    if application_confidences:
        result["_min_domain_confidence"] = min(application_confidences.values())
        result["_all_static"] = all(c >= 1.0 for c in application_confidences.values())
    else:
        result["_min_domain_confidence"] = None
        result["_all_static"] = False

    # Structured evidence block — this is what the report / API surfaces.
    try:
        result["_evidence"] = build_evidence({
            "chunk_id": chunk_id,
            "sensor": sensor,
            "source": source,
            "user": user,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "event_count": event_count,
            "total_bytes": total_bytes,
            "match_tier": match_tier,
            "log_formats": log_formats,
            "applications": applications,
            "application_confidences": application_confidences,
            "text_hints": text_hints,
            "events": events,
            "_reanalysis": is_reanalysis,
            "_reanalysis_reason": reanalysis_reason,
            "_early_submitted": is_early_submit,
        })
    except Exception as exc:
        dprint(f"build_evidence failed: {exc}")
        result["_evidence"] = {"error": str(exc)[:200]}

    result["_analysis_seconds"] = round(time.monotonic() - analysis_started, 2)
    result["_chunk_id"] = chunk_id
    result["_sensor"] = sensor
    result["_source"] = source
    result["_user"] = user
    result["_start_ts"] = start_ts
    result["_end_ts"] = end_ts
    result["_applications"] = list(applications.keys())
    result["_approved"] = applications
    result["_match_tier"] = match_tier
    result["_log_formats"] = log_formats
    result["_bytes"] = total_bytes
    result["_event_count"] = event_count
    result["_reanalysis"] = is_reanalysis
    result["_early_submit"] = is_early_submit
    if reanalysis_reason:
        result["_reanalysis_reason"] = reanalysis_reason
    return result


# ============================================================
# STREAMING / BOUNDED CONCURRENT ANALYSIS
# ============================================================
class RunContext:
    __slots__ = ("executor", "pending", "max_pending", "findings_fp", "review_fp",
                 "started", "callback", "pending_domain_watch", "stats")

    def __init__(self, executor, findings_fp, review_fp, max_pending, started, callback):
        self.executor = executor
        self.pending = set()
        self.max_pending = max_pending
        self.findings_fp = findings_fp
        self.review_fp = review_fp
        self.started = started
        self.callback = callback
        self.pending_domain_watch: list[dict] = []
        self.stats: dict[str, int] = defaultdict(int)


def handle_chunk(ctx: RunContext, chunk: dict) -> None:
    ctx.stats["chunks_seen"] += 1
    _emit(
        "chunk",
        chunk_id=chunk["chunk_id"],
        tier=chunk["match_tier"],
        events=chunk["event_count"],
        bytes=chunk["total_bytes"],
        is_ai=bool(chunk["is_ai_related"]),
        needs_review=bool(chunk["needs_review"]),
        apps=list(chunk["applications"].keys()),
        unknown=chunk["unknown_domains"],
        user=chunk.get("user"),
    )
    for fmt, cnt in chunk.get("log_formats", {}).items():
        ctx.stats[f"fmt::{fmt}"] += cnt

    has_unknown = bool(chunk["unknown_domains"])

    if chunk["needs_review"]:
        ctx.stats["review_chunks"] += 1
        dprint(f"chunk #{chunk['chunk_id']} NEEDS REVIEW (unknown={chunk['unknown_domains']})")
        ctx.review_fp.write(json.dumps({
            "chunk_id": chunk["chunk_id"],
            "sensor": chunk["sensor"],
            "source": chunk["source"],
            "user": chunk.get("user"),
            "start_ts": chunk["start_ts"],
            "end_ts": chunk["end_ts"],
            "unknown_domains": chunk["unknown_domains"],
            "log_formats": chunk.get("log_formats", {}),
            "event_count": chunk["event_count"],
        }, ensure_ascii=False) + "\n")
        ctx.pending_domain_watch.append({"chunk": chunk, "already_analyzed": False})
        register_chunk_for_early_submit(chunk, was_analyzed=False)
        return

    if chunk["is_ai_related"]:
        # ── signal gate: don't waste 2 min of LLM on a chunk that has
        #    no real domain match, no bytes, no user, and only a handful
        #    of coincidental text-hint events.
        if not chunk["applications"]:
            hint_events = sum(1 for e in chunk["events"] if e.get("text_hints"))
            weak_signal = (
                chunk["total_bytes"] == 0
                and not chunk.get("user")
            )
            if weak_signal or hint_events < MIN_HINT_EVENTS_FOR_LLM:
                dprint(
                    f"chunk #{chunk['chunk_id']} skipped "
                    f"(hint_events={hint_events}, bytes={chunk['total_bytes']}, "
                    f"user={chunk.get('user')!r})"
                )
                ctx.stats["skipped_weak_hints"] = ctx.stats.get("skipped_weak_hints", 0) + 1
                return

        ctx.stats["ai_chunks"] += 1
        apps = list(chunk["applications"].keys()) or "[]"
        msg = (
            f"AI chunk #{chunk['chunk_id']}: user={chunk.get('user', 'unknown')} "
            f"apps={apps} tier={chunk['match_tier']} events={chunk['event_count']} "
            f"bytes={chunk['total_bytes']:,} formats={chunk.get('log_formats', {})}"
        )
        print(f"\n[{msg}]")
        if ctx.callback:
            ctx.callback(msg)

        if has_unknown:
            ctx.pending_domain_watch.append({"chunk": chunk, "already_analyzed": True})
            register_chunk_for_early_submit(chunk, was_analyzed=True)

        future = ctx.executor.submit(analyze_chunk, chunk)
        ctx.pending.add(future)
        if len(ctx.pending) >= ctx.max_pending:
            done, ctx.pending = wait(ctx.pending, return_when=FIRST_COMPLETED)
            completed = consume_done(done, ctx.pending, ctx.findings_fp, ctx.started)
            ctx.stats["findings"] += len(completed)
            if ctx.callback:
                ctx.callback(f"Findings so far: {ctx.stats['findings']}", findings=completed)
        return

    dprint(f"chunk #{chunk['chunk_id']} dropped: no AI evidence, no unresolved domains")


def _drain_ready_queue(ctx: RunContext) -> None:
    """
    Drains the ready-to-submit queue, grouping resolutions by chunk so a
    chunk with multiple unknown domains that resolve close together is
    submitted only once with the full set of new applications.
    """
    items: list[tuple[dict, str, str, bool]] = []
    while True:
        try:
            items.append(_ready_to_submit_queue.get_nowait())
        except queue.Empty:
            break
    if not items:
        return

    grouped: dict[int, list[tuple[str, str, bool]]] = {}
    chunk_refs: dict[int, dict] = {}
    for chunk, domain, app_name, approved in items:
        key = id(chunk)
        grouped.setdefault(key, []).append((domain, app_name, approved))
        chunk_refs[key] = chunk

    for key, resolutions in grouped.items():
        chunk = chunk_refs[key]
        chunk_id = chunk.get("chunk_id")
        resolved: set = chunk.setdefault("_resolved_domains", set())

        new_domains: list[str] = []
        for domain, app_name, approved in resolutions:
            if domain in resolved:
                dprint(
                    f"_drain_ready_queue: chunk #{chunk_id} domain={domain} "
                    f"already resolved, skipping"
                )
                continue
            resolved.add(domain)
            new_domains.append(domain)
            chunk["applications"][app_name] = approved
            chunk.setdefault("application_confidences", {})
            entry = cache_get(domain)
            chunk["application_confidences"][app_name] = (
                float(entry.get("confidence", 0.8)) if entry else 0.8
            )

        if not new_domains:
            continue

        chunk["is_ai_related"] = True
        chunk["match_tier"] = "domain"
        chunk["unknown_domains"] = [
            d for d in chunk.get("unknown_domains", []) if d not in resolved
        ]

        was_analyzed_before = bool(chunk.get("_was_analyzed_before", False))
        was_early_submitted = bool(chunk.get("_early_submitted", False))

        if was_analyzed_before or was_early_submitted:
            chunk["_reanalysis"] = True
            chunk["_reanalysis_reason"] = (
                f"Domain(s) {sorted(new_domains)} confirmed AI by background "
                f"classifier after initial analysis (early submission)."
            )
            kind = "EARLY-REANALYSIS"
        else:
            kind = "EARLY-SUBMIT"

        chunk["_early_submitted"] = True

        ctx.stats["ai_chunks"] += 1
        msg = (
            f"{kind} chunk #{chunk_id}: user={chunk.get('user', 'unknown')} "
            f"resolved_domains={sorted(new_domains)} "
            f"apps={list(chunk['applications'].keys())}"
        )
        print(f"\n[{msg}]")
        if ctx.callback:
            ctx.callback(msg)

        future = ctx.executor.submit(analyze_chunk, chunk)
        ctx.pending.add(future)
        if len(ctx.pending) >= ctx.max_pending:
            done, ctx.pending = wait(ctx.pending, return_when=FIRST_COMPLETED)
            completed = consume_done(done, ctx.pending, ctx.findings_fp, ctx.started)
            ctx.stats["findings"] += len(completed)
            if ctx.callback:
                ctx.callback(f"Findings so far: {ctx.stats['findings']}", findings=completed)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def format_rate(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def consume_done(done, pending, findings_fp, runtime_started):
    completed = []
    for future in done:
        pending.discard(future)
        try:
            result = future.result()
        except Exception as exc:
            dprint(f"consume_done: unexpected future failure: {exc}")
            result = {
                "classification": None,
                "confidence": 0.0,
                "reason": f"Unexpected pipeline error: {exc}",
            }

        findings_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
        findings_fp.flush()
        completed.append(result)

        # ── FIX 6: emit live "finding" event for the dashboard ──
        _emit(
            "finding",
            chunk_id=result.get("_chunk_id"),
            user=result.get("_user"),
            application=result.get("application"),
            severity=result.get("severity"),
            classification=result.get("classification"),
            confidence=result.get("confidence"),
            match_tier=result.get("_match_tier"),
            bytes=result.get("_bytes"),
            reason=(result.get("reason") or "")[:240],
        )

        apps = result.get("_applications") or []
        app_text = ", ".join(apps) if apps else "[]"
        tier = result.get("_match_tier", "unknown")
        bytes_sent = result.get("_bytes", 0)
        severity = result.get("severity", "-")
        action = result.get("recommended_action", "-")
        analysis_time = result.get("_analysis_seconds", 0)
        user = result.get("_user") or "unknown"
        chunk_id = result.get("_chunk_id", "-")
        formats = result.get("_log_formats", {})
        conf = result.get("confidence")
        min_dconf = result.get("_min_domain_confidence")
        tags = []
        if result.get("_reanalysis"):
            tags.append("[REANALYSIS]")
        if result.get("_early_submit"):
            tags.append("[EARLY]")
        tag = (" " + " ".join(tags)) if tags else ""
        elapsed = time.monotonic() - runtime_started

        conf_txt = f"conf={conf:.2f}" if isinstance(conf, (int, float)) else f"conf={conf}"
        dconf_txt = f"dconf={min_dconf:.2f}" if isinstance(min_dconf, (int, float)) else "dconf=?"

        print(
            f"  {tag}chunk_id={chunk_id} user={user:20} apps={app_text:30} tier={tier:10} "
            f"formats={formats} {bytes_sent:>9}B  sev={severity:<9} "
            f"{conf_txt} {dconf_txt} analysis={analysis_time:>6.1f}s  "
            f"elapsed={format_duration(elapsed):>9}\n"
            f"    action: {action}"
        )
    return completed


# ============================================================
# PIPELINE SUMMARY
# ============================================================
def print_pipeline_summary(*, rows_seen, rows_failed, chunks_seen, ai_chunks, review_chunks,
                            findings, reclassified_new, reanalyzed_upgraded, cache_stats,
                            format_counts, ooo_count, ts_parse_failures):
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Rows read from CSV             : {rows_seen:,}")
    print(f"  ├─ Failed to normalize       : {rows_failed:,}")
    print(f"  ├─ Out-of-order in source    : {ooo_count:,} (corrected by chronological sort)")
    print(f"  └─ Timestamp parse failures  : {ts_parse_failures:,}")
    print("Log formats seen:")
    for fmt, cnt in sorted(format_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fmt:20s}: {cnt:,}")
    print(f"Chunks built                   : {chunks_seen:,}")
    print(f"  ├─ AI-related (submitted)    : {ai_chunks:,}")
    print(f"  └─ Needs review (unknown)    : {review_chunks:,}")
    print(f"Chunks newly classified AI     : {reclassified_new:,}")
    print(f"Chunks reanalyzed (upgraded)   : {reanalyzed_upgraded:,}")
    print(f"Total findings written         : {findings:,}")
    print()
    print("Domain cache status counts:")
    for status, count in sorted(cache_stats.items()):
        print(f"  {status:10s}: {count}")
    print("=" * 60)
    if findings == 0:
        print("NOTE: No findings were produced.")
        if ai_chunks == 0 and review_chunks > 0:
            print("  -> All chunks were 'needs_review'. The classifier either")
            print("     found them non-AI, returned UNKNOWN, or the model failed.")
            print("     Check the [classify] lines above.")
        elif ai_chunks > 0 and findings == 0:
            print("  -> Chunks were submitted but the main LLM returned no findings.")
    print()


# ============================================================
# CORE PIPELINE
# ============================================================
def process_file(csv_path: Path, output_path: Path, review_path: Path,
                 workers: int, progress_every: int,
                 callback=None, log_year=None):
    global _domain_observation_counts

    reset_timestamp_failure_counter()
    reset_pipeline_events()
    _emit("run_started", csv=str(csv_path), workers=workers)
    reset_early_submission_state()
    _domain_observation_counts = defaultdict(int)

    started = time.monotonic()
    rows_seen = 0

    dsection("PHASE 1: READING + NORMALIZING ALL ROWS")
    normalized_rows: list[tuple[int, str, str, dict]] = []
    format_counts: dict[str, int] = defaultdict(int)
    last_ts_by_key: dict[tuple, float] = {}
    ooo_count = 0
    rows_failed = 0

    with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=",")

        for i, raw_row in enumerate(reader):
            try:
                normalized = {
                    (k or "").strip().lower(): (v or "").strip()
                    for k, v in raw_row.items()
                }
                sensor = normalized.get("sensor", "")
                source = normalized.get("source", "")
                timestamp = normalized.get("timestamp", "")
                event = normalized.get("event", "")

                row = normalize_row(sensor, source, timestamp, event, log_year=log_year)
                format_counts[row.get("log_format", "unstructured")] += 1

                user = row.get("user")
                key = (sensor, source, user)
                prev_ts = last_ts_by_key.get(key)
                if prev_ts is not None and row.get("ts_epoch") is not None and row["ts_epoch"] < prev_ts:
                    ooo_count += 1
                    dprint(
                        f"Phase 1: out-of-order ts for key={key}: "
                        f"{row['ts_epoch']} < {prev_ts} (row {i}) -> will be corrected by sort"
                    )
                if row.get("ts_epoch") is not None:
                    last_ts_by_key[key] = row["ts_epoch"]

                normalized_rows.append((i, sensor, source, row))
            except Exception as exc:
                rows_failed += 1
                dprint(f"Phase 1: failed to process row {i}: {exc}")
                continue

            rows_seen += 1
            if progress_every and rows_seen % progress_every == 0:
                # ── FIX 3: only emit "phase1_progress" here; the "phase1_done"
                #           emit that used to live in this block has been moved
                #           out (see below).
                _emit("phase1_progress", rows=rows_seen, failed=rows_failed)
                elapsed = max(time.monotonic() - started, 0.001)
                rate = rows_seen / elapsed
                msg = f"Phase 1: read {rows_seen:,} rows | {rate:,.0f} rows/s"
                print(msg)
                if callback:
                    callback(msg)

    dprint(
        f"Phase 1 complete: {len(normalized_rows):,} rows normalized, {rows_failed:,} failed, "
        f"{ooo_count:,} out-of-order relative to file order"
    )
    print(
        f"Phase 1 complete: {len(normalized_rows):,} rows normalized "
        f"({rows_failed:,} failed, {ooo_count:,} out-of-order in source order)"
    )
    if format_counts:
        print("Log formats detected:")
        for fmt, cnt in sorted(format_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {fmt:20s}: {cnt:,}")

    # ── FIX 3 (continued): emit phase1_done exactly once, here.
    _emit("phase1_done", rows=rows_seen, failed=rows_failed,
          formats=dict(format_counts), ooo=ooo_count)

    dsection("PHASE 2: SORTING ROWS PER ENTITY FOR CHRONOLOGICAL CHUNKING")

    def _sort_key(item):
        seq, sensor, source, row = item
        user = row.get("user") or ""
        ts = row.get("ts_epoch")
        return (sensor, source, user, ts if ts is not None else float("inf"), seq)

    normalized_rows.sort(key=_sort_key)
    print(
        f"Phase 2 complete: sorted {len(normalized_rows):,} rows per (sensor, source, user). "
        f"Rows with unparsable timestamps sort last within their session."
    )

    # ── FIX 4: emit phase2_done so the dashboard advances past phase 1.
    _emit("phase2_done", rows=len(normalized_rows))

    dsection("PHASE 3: CHUNKING + STREAMING ANALYSIS")
    states: dict[tuple, EntityState] = {}
    max_pending = max(workers * 2, 2)

    reclassified_new = 0
    reanalyzed_upgraded = 0

    with output_path.open("w", encoding="utf-8") as findings_fp, \
         review_path.open("w", encoding="utf-8") as review_fp, \
         ThreadPoolExecutor(max_workers=workers) as executor:

        ctx = RunContext(executor, findings_fp, review_fp, max_pending, started, callback)

        def _early_submit_callback(domain: str) -> None:
            _fire_early_submission(domain)

        set_review_submission_callback(_early_submit_callback)

        try:
            for seq, sensor, source, row in normalized_rows:
                _drain_ready_queue(ctx)

                user = row.get("user")
                key = (sensor, source, user)
                state = states.setdefault(key, EntityState())
                for chunk in add_event_to_state(state, row, sensor, source, user):
                    handle_chunk(ctx, chunk)

            dsection("FLUSHING FINAL PARTIAL SESSIONS")
            for (sensor, source, user), state in states.items():
                if state.events:
                    try:
                        chunk = build_chunk_from_state(state, sensor, source, user)
                        handle_chunk(ctx, chunk)
                    except Exception as exc:
                        dprint(f"Final flush: error handling state for key=({sensor},{source},{user}): {exc}")

            _drain_ready_queue(ctx)

            dsection("DRAINING INITIAL QWEN JOBS")
            while ctx.pending:
                done, ctx.pending = wait(ctx.pending, return_when=FIRST_COMPLETED)
                completed = consume_done(done, ctx.pending, findings_fp, started)
                ctx.stats["findings"] += len(completed)
                if callback:
                    callback(f"Findings so far: {ctx.stats['findings']}", findings=completed)

            dsection("WAITING FOR DOMAIN CLASSIFIER TO FINISH")
            print("\nWaiting for domain classifier to finish...")
            if callback:
                callback("Waiting for domain classifier to finish...")
            if domain_classifier is not None:
                domain_classifier.shutdown(timeout=30)

            _drain_ready_queue(ctx)

            while ctx.pending:
                done, ctx.pending = wait(ctx.pending, return_when=FIRST_COMPLETED)
                completed = consume_done(done, ctx.pending, findings_fp, started)
                ctx.stats["findings"] += len(completed)
                if callback:
                    callback(f"Findings so far: {ctx.stats['findings']}", findings=completed)

            save_dynamic_cache()

            dsection(f"RE-EXAMINING {len(ctx.pending_domain_watch)} CHUNK(S) WITH UNRESOLVED DOMAINS")
            for idx, entry in enumerate(ctx.pending_domain_watch, 1):
                chunk = entry["chunk"]
                resolved: set = chunk.setdefault("_resolved_domains", set())

                new_apps: dict[str, bool] = {}
                new_confs: dict[str, float] = {}
                ai_domains: set[str] = set()

                for domain in list(chunk.get("unknown_domains", [])):
                    if domain in resolved:
                        continue
                    cached = cache_get(domain)
                    dprint(
                        f"  watch #{idx} chunk_id={chunk.get('chunk_id')} domain={domain} "
                        f"cache_status={cached['status'] if cached else 'MISSING'}"
                    )
                    if cached and cached["status"] == "ai":
                        new_apps[cached["app"]] = cached.get("approved", False)
                        new_confs[cached["app"]] = float(cached.get("confidence", 0.0))
                        ai_domains.add(domain)

                if not new_apps:
                    continue

                try:
                    resolved.update(ai_domains)
                    chunk["applications"].update(new_apps)
                    chunk.setdefault("application_confidences", {})
                    chunk["application_confidences"].update(new_confs)
                    chunk["is_ai_related"] = True
                    chunk["match_tier"] = "domain"
                    chunk["unknown_domains"] = [
                        d for d in chunk["unknown_domains"] if d not in resolved
                    ]

                    already_submitted = (
                        bool(chunk.get("_early_submitted"))
                        or bool(entry["already_analyzed"])
                    )

                    if already_submitted:
                        chunk["_reanalysis"] = True
                        chunk["_reanalysis_reason"] = (
                            f"Additional domain(s) confirmed AI by the background classifier "
                            f"after the first analysis pass: {sorted(ai_domains)}"
                        )
                        reanalyzed_upgraded += 1
                        msg = (
                            f"Reanalyzing chunk_id={chunk['chunk_id']} (upgraded evidence): "
                            f"user={chunk.get('user')} apps={list(chunk['applications'].keys())} "
                            f"newly_confirmed={sorted(ai_domains)}"
                        )
                    else:
                        reclassified_new += 1
                        msg = (
                            f"Reclassified chunk_id={chunk['chunk_id']} (was needs_review): "
                            f"user={chunk.get('user')} apps={list(chunk['applications'].keys())} "
                            f"removed_unknowns={sorted(ai_domains)}"
                        )

                    print(f"\n[{msg}]")
                    if callback:
                        callback(msg)

                    future = executor.submit(analyze_chunk, chunk)
                    ctx.pending.add(future)
                    if len(ctx.pending) >= ctx.max_pending:
                        done, ctx.pending = wait(ctx.pending, return_when=FIRST_COMPLETED)
                        completed = consume_done(done, ctx.pending, findings_fp, started)
                        ctx.stats["findings"] += len(completed)
                        if callback:
                            callback(f"Findings so far: {ctx.stats['findings']}", findings=completed)
                except Exception as exc:
                    dprint(f"  watch #{idx}: error during re-examination: {exc}")

            if reclassified_new or reanalyzed_upgraded:
                print(
                    f"\n{reclassified_new} chunk(s) newly classified as AI; "
                    f"{reanalyzed_upgraded} previously-analyzed chunk(s) upgraded with stronger evidence."
                )
                if callback:
                    callback(f"Reclassified {reclassified_new}, reanalyzed {reanalyzed_upgraded}")

            dsection("DRAINING NEWLY SUBMITTED QWEN JOBS")
            while ctx.pending:
                done, ctx.pending = wait(ctx.pending, return_when=FIRST_COMPLETED)
                completed = consume_done(done, ctx.pending, findings_fp, started)
                ctx.stats["findings"] += len(completed)
                if callback:
                    callback(f"Findings so far: {ctx.stats['findings']}", findings=completed)
        finally:
            set_review_submission_callback(None)

        save_dynamic_cache()

    with _cache_lock:
        cache_stats = defaultdict(int)
        for v in _dynamic_cache.values():
            cache_stats[v["status"]] += 1

    print_pipeline_summary(
        rows_seen=rows_seen,
        rows_failed=rows_failed,
        chunks_seen=ctx.stats["chunks_seen"],
        ai_chunks=ctx.stats["ai_chunks"],
        review_chunks=ctx.stats["review_chunks"],
        findings=ctx.stats["findings"],
        reclassified_new=reclassified_new,
        reanalyzed_upgraded=reanalyzed_upgraded,
        cache_stats=dict(cache_stats),
        format_counts=format_counts,
        ooo_count=ooo_count,
        ts_parse_failures=TIMESTAMP_PARSE_FAILURES,
    )

    skipped = ctx.stats.get("skipped_weak_hints", 0)
    if skipped:
        print(f"Weak-hint chunks skipped       : {skipped:,} "
              f"(no domains, no bytes, no user)")

    elapsed = max(time.monotonic() - started, 0.001)
    print("Done.")
    print(f"Rows processed      : {rows_seen:,}")
    print(f"Chunks built        : {ctx.stats['chunks_seen']:,}")
    print(f"AI chunks submitted : {ctx.stats['ai_chunks']:,}")
    print(f"Findings written    : {ctx.stats['findings']:,}")
    print(f"Review chunks       : {ctx.stats['review_chunks']:,}")
    print(f"Elapsed             : {format_duration(elapsed)}")
    print(f"Throughput          : {format_rate(rows_seen / elapsed)} rows/s")
    print(f"Findings file       : {output_path}")
    print(f"Review file         : {review_path}")

    # Final event so the dashboard knows the run is over.
    _emit(
        "run_finished",
        rows=rows_seen,
        chunks=ctx.stats["chunks_seen"],
        ai=ctx.stats["ai_chunks"],
        findings=ctx.stats["findings"],
        review=ctx.stats["review_chunks"],
        elapsed=time.monotonic() - started,
    )

# ============================================================
# BACKEND INTEGRATION FUNCTION
# ============================================================
def _shutdown_existing_classifier() -> None:
    global domain_classifier
    if domain_classifier is not None:
        try:
            domain_classifier.shutdown(timeout=30)
        except Exception as exc:
            dprint(f"Error shutting down existing DomainClassifier: {exc}")
        domain_classifier = None


def _extract_ollama_model_names(models_raw) -> list[str]:
    """Robust extraction of model names from an Ollama list() response."""
    raw_models = None
    if isinstance(models_raw, dict):
        raw_models = models_raw.get("models")
    if raw_models is None:
        raw_models = getattr(models_raw, "models", None)
    if raw_models is None:
        return []

    names: list[str] = []
    for m in raw_models:
        name = None
        if isinstance(m, dict):
            name = m.get("name") or m.get("model")
        else:
            name = getattr(m, "model", None) or getattr(m, "name", None)
        if name:
            names.append(name)
    return names


def _verify_required_models() -> None:
    """Fail fast if any required Ollama model is missing on the server."""
    required = [EMBEDDING_MODEL, ANALYSIS_MODEL, DOMAIN_CLASSIFY_MODEL]
    try:
        client = Client(host=f"http://{DEFAULT_OLLAMA_HOST}:{OLLAMA_PORT}")
        models_raw = client.list()
        models = _extract_ollama_model_names(models_raw)
    except Exception as e:
        print(f"WARNING: Could not verify Ollama models at "
              f"http://{DEFAULT_OLLAMA_HOST}:{OLLAMA_PORT}: {e}")
        print("  Continuing anyway — classification may fail if models are absent.")
        return

    missing = [m for m in required if m not in models]
    if missing:
        print(f"ERROR: Missing Ollama models on {DEFAULT_OLLAMA_HOST}: {missing}")
        print(f"  Available: {models}")
        print(f"  Pull the missing one(s) with:  ollama pull <model>")
        raise SystemExit(1)
    print("Ollama models verified.")


def analyze_csv_file(
    csv_path: Path,
    output_path: Path,
    review_path: Path,
    workers: int = DEFAULT_QWEN_WORKERS,
    progress_every: int = 1000,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    classify_model: str = DOMAIN_CLASSIFY_MODEL,
    log_year: int | None = None,
    recheck_unknown_days: float = 7.0,
    callback=None,
) -> dict:
    global DEFAULT_OLLAMA_HOST, DOMAIN_CLASSIFY_MODEL, domain_classifier
    global embedding_function, chroma_client, knowledge

    DEFAULT_OLLAMA_HOST = ollama_host
    DOMAIN_CLASSIFY_MODEL = classify_model

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    embedding_function = OllamaEmbeddingFunction(
        url=f"http://{DEFAULT_OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    knowledge = chroma_client.get_or_create_collection(
        name=KNOWLEDGE_COLLECTION,
        embedding_function=embedding_function,
    )

    _verify_required_models()
    load_dynamic_cache()

    _shutdown_existing_classifier()
    domain_classifier = DomainClassifier(
        model=DOMAIN_CLASSIFY_MODEL,
        batch_size=DOMAIN_BATCH_SIZE,
        recheck_seconds=recheck_unknown_days * 86400,
    )

    process_file(csv_path, output_path, review_path, workers, progress_every,
                 callback, log_year=log_year)

    save_dynamic_cache()

    findings = []
    severity_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                findings.append(result)
                sev = result.get("severity") or result.get("risk")
                if sev in severity_counts:
                    severity_counts[sev] += 1

    return {
        "findings": findings,
        "findings_count": len(findings),
        "severity_counts": severity_counts,
    }


# ============================================================
# MAIN
# ============================================================
def _print_file_hint(csv_path: Path) -> None:
    """Print helpful info when the input CSV is not found."""
    cwd = Path.cwd()
    print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
    print(f"  Current working directory: {cwd}", file=sys.stderr)
    try:
        files = sorted(p.name for p in cwd.iterdir() if p.is_file())
    except OSError:
        files = []
    csv_like = [f for f in files if f.lower().endswith(".csv")]
    if csv_like:
        print(f"  CSV files in this directory: {csv_like}", file=sys.stderr)
        print(f"  Did you mean one of these?", file=sys.stderr)
    else:
        print(f"  No .csv files found in {cwd}", file=sys.stderr)


def main() -> None:
    global DEFAULT_OLLAMA_HOST, DOMAIN_CLASSIFY_MODEL, domain_classifier, DEBUG
    global embedding_function, chroma_client, knowledge

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
                    help=f"Small model used for domain classification (default: {DOMAIN_CLASSIFY_MODEL})")
    ap.add_argument("--log-year", type=int, default=None,
                    help="Year to assume for timestamps that omit one (e.g. syslog).")
    ap.add_argument("--recheck-unknown-days", type=float, default=7.0,
                    help="Re-attempt classification for domains cached as UNKNOWN after this many days.")
    ap.add_argument("--debug", action="store_true",
                    help="Enable verbose debug output showing every internal step.")
    ap.add_argument("--reset-cache", action="store_true",
                    help="Delete the dynamic domain cache before running.")
    args = ap.parse_args()

    DEBUG = args.debug
    DEFAULT_OLLAMA_HOST = args.ollama_host
    DOMAIN_CLASSIFY_MODEL = args.classify_model

    # --- Input file check (fail early with a friendly message) ---
    if not args.csv.exists():
        _print_file_hint(args.csv)
        raise SystemExit(2)

    if args.reset_cache and DYNAMIC_CACHE_FILE.exists():
        DYNAMIC_CACHE_FILE.unlink()
        print(f"Deleted {DYNAMIC_CACHE_FILE}")

    embedding_function = OllamaEmbeddingFunction(
        url=f"http://{DEFAULT_OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    knowledge = chroma_client.get_or_create_collection(
        name=KNOWLEDGE_COLLECTION,
        embedding_function=embedding_function,
    )

    _verify_required_models()

    if knowledge.count() == 0:
        raise SystemExit("Knowledge base is EMPTY. Run: python build_knowledge_base.py")

    load_dynamic_cache()

    _shutdown_existing_classifier()
    domain_classifier = DomainClassifier(
        model=DOMAIN_CLASSIFY_MODEL,
        batch_size=DOMAIN_BATCH_SIZE,
        recheck_seconds=args.recheck_unknown_days * 86400,
    )

    workers = max(1, args.workers)

    print(f"Knowledge RAG        : {knowledge.count()} policy chunks")
    print(f"Qwen workers         : {workers}")
    print(f"Chunk size           : {MAX_EVENTS_PER_CHUNK}")
    print(f"Overlap              : {OVERLAP_COUNT} events")
    print(f"Session gap          : {SESSION_GAP_SECONDS}s")
    print(f"Ollama host          : {DEFAULT_OLLAMA_HOST}")
    print(f"Classifier model     : {DOMAIN_CLASSIFY_MODEL}")
    print(f"Analyzer model       : {ANALYSIS_MODEL}")
    print(f"Recheck unknown after: {args.recheck_unknown_days:g} day(s)")
    print(f"Dynamic AI threshold : {CONFIDENCE_THRESHOLD:.2f}")
    print(f"Min domain sightings : {MIN_DYNAMIC_DOMAIN_OBSERVATIONS}")
    print(f"Early submission     : ON")
    print(f"Debug mode           : {'ON' if DEBUG else 'OFF'}")
    print()

    try:
        process_file(
            args.csv,
            args.output,
            args.review_output,
            workers,
            args.progress_every,
            log_year=args.log_year,
        )
    finally:
        _shutdown_existing_classifier()
        save_dynamic_cache()


if __name__ == "__main__":
    main()


# ---------
# drawbacks
# ---------

# it is getting stuck after few logs and not showing user names, confidece score is not good
# need to change llm