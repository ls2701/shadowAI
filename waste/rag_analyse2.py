"""

OPTIMIZED for 150MB-1GB files. Same detection logic and output shape as
before; six changes below cut CPU and memory cost in steps 1-4 (the parts
that run entirely on your laptop, with no network calls, so they're the
whole bottleneck for a big file). Steps 5-6 (RAG + qwen) are unchanged
except for caching (#6).

WHAT CHANGED AND WHY (biggest cost first):
  1. Dropped the `"raw": event[:2000]` field entirely. It was stored on
     EVERY row but never read again after normalize_row() ran - text hints
     are scanned on the original `event` string, not on this copy. For a
     1GB file with a few million rows this was gigabytes of dead memory.
  2. Sysmon/AWN JSON rows no longer do json.loads() + json.dumps() just to
     regex the result. Regex runs directly on the original string once.
  3. The 30 text-hint keyword checks are now one compiled regex alternation
     instead of 30 separate Python-level `in` checks per row.
  4. CSV header is parsed once; rows are read by column index instead of
     rebuilding + relowercasing a dict on every single row (DictReader
     overhead).
  5. Row normalization is parallelized across CPU cores with
     multiprocessing.Pool, since it's pure CPU work with zero shared state
     between rows - the textbook case for it. Falls back to serial
     automatically on small files or single-core machines.
  6. RAG policy retrieval is cached by which app(s)/hints a chunk matched,
     so 50 chunks that all say "ChatGPT" hit Chroma/Ollama once instead of
     50 times.

Prereqs (unchanged):
  - knowledge RAG already built:  python build_knowledge_base.py
  - Ollama on Tower reachable at TOWER_IP:11434 with nomic + qwen2.5:14b pulled

    pip install chromadb ollama
    python rag_analyse_optimized.py testlogs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from ollama import Client


# ---------------- CONFIG ----------------
TOWER_IP = "192.168.100.100"
OLLAMA_BASE = f"http://{TOWER_IP}:11434"

CHROMA_PATH = "./chroma_db"
KNOWLEDGE_COLLECTION = "shadow_ai_knowledge"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
ANALYSIS_MODEL = "qwen2.5:14b"

OUTPUT_FILE = "findings.jsonl"
RESULTS_PER_QUERY = 3
CONTEXT_QUERIES_PER_EVENT = 4

MAX_EVENTS_PER_CHUNK = 40
SESSION_GAP_SECONDS = 300
OVERLAP_RATIO = 0.10

# Set to 1 to force single-core (useful for debugging); None = auto-detect.
WORKERS = None
# Below this many rows, multiprocessing overhead isn't worth paying.
PARALLEL_THRESHOLD_ROWS = 20_000


# ---------------- CHROMA + OLLAMA (policy RAG only) ----------------
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = OllamaEmbeddingFunction(url=f"{OLLAMA_BASE}/api/embeddings", model_name=EMBEDDING_MODEL)
knowledge = client.get_or_create_collection(name=KNOWLEDGE_COLLECTION, embedding_function=embedding_function)
ollama_client = Client(host=OLLAMA_BASE)


# ---------------- TEXT-HINT LIST -> ONE COMPILED PATTERN (FIX #3) ----------------
AI_TEXT_HINTS = [
    'category="artificial intelligence"', 'category=artificial intelligence',
    "artificial intelligence",
    "/backend-api/conversation", "/v1/messages", "/chat/completions", "/v1/complete", "/v1/chat",
    "generativelanguage", "anthropic.claude",
    "chatbot", "ai-chatbot", "genai", "generative-ai",
    "gpt-", "-gpt", "llm-", "-llm",
    "virtual assistant", "ai assistant", "conversational ai",
    "live chat", "livechat", "/api/chat", "/api/converse",
    "conversation_id", "chat_completion", "/bot/",
]
# One alternation instead of 30 separate `in` checks per row. re.escape each
# literal since some contain regex-special chars like "." and "/".
_HINT_PATTERN = re.compile("|".join(re.escape(h) for h in AI_TEXT_HINTS), re.IGNORECASE)


def scan_text_hints(raw_text: str) -> list[str]:
    """Single regex pass over the FULL untruncated event text."""
    found = {m.group(0).lower() for m in _HINT_PATTERN.finditer(raw_text)}
    return sorted(found)


AWS_AI_EVENT_SOURCES = [
    "bedrock", "sagemaker", "comprehend", "rekognition", "textract",
    "polly", "lex.amazonaws", "personalize", "kendra", "forecast", "transcribe",
]

_DNS_LABEL_RE = re.compile(r'((?:\(\d+\)[A-Za-z0-9_-]+)+)\(0\)')
_DNS_PART_RE = re.compile(r'\(\d+\)([A-Za-z0-9_-]+)')
_CT_EVENTSOURCE_RE = re.compile(r'"eventSource"\s*:\s*"([^"]+)"')
_CT_USERNAME_RE = re.compile(r'"userName"\s*:\s*"([^"]+)"')
# Regex-only extraction for AWN/Sysmon JSON — see FIX #2. Non-greedy [^}]*?
# stops before crossing into a nested object, which matches this schema
# ("hostname" always appears before nested "user"/"interfaces" objects).
_AWN_HOSTNAME_RE = re.compile(r'"host"\s*:\s*\{[^}]*?"hostname"\s*:\s*"([^"]*)"')
_AWN_USERNAME_RE = re.compile(r'"user"\s*:\s*\{\s*"name"\s*:\s*"([^"]*)"')
_AWN_DEST_HOST_RE = re.compile(r'destinationHostname["\\:\s]+([a-z0-9.\-]+)', re.IGNORECASE)
_AWN_BACKSLASH_USER_RE = re.compile(r'\\\\([A-Za-z0-9_.\-]+)"')


def _dns_decode(text: str) -> str | None:
    m = _DNS_LABEL_RE.search(text)
    if not m:
        return None
    labels = _DNS_PART_RE.findall(m.group(1))
    return ".".join(labels).lower() if labels else None


def normalize_row(sensor: str, source: str, ts: str, event: str) -> dict:
    e = event
    out = {"timestamp": ts, "sensor": sensor, "source": source,
           "user": None, "hostname": None, "domain": None, "bytes": 0,
           "text_hints": scan_text_hints(event)}
    # NOTE: no "raw" field stored — see FIX #1 in the module docstring.

    def kv(key):
        m = re.search(rf'{key}="([^"]*)"', e) or re.search(rf'{key}=([^\s]+)', e)
        return m.group(1) if m else None

    try:
        if "device_name=" in e or "id=firewall" in e:           # Sophos / SonicWall
            out["domain"] = (kv("dstname") or "").lower() or None
            try:
                out["bytes"] = int(kv("sent") or 0) + int(kv("rcvd") or 0)
            except ValueError:
                out["bytes"] = 0
        elif "MSWinEventLog" in e and "PACKET" in e:             # Windows DNS
            out["domain"] = _dns_decode(e)
        elif e.strip().startswith("{") and '"eventSource"' in e and '"eventName"' in e:
            # AWS CloudTrail
            es_match = _CT_EVENTSOURCE_RE.search(e)
            event_source = (es_match.group(1) if es_match else "").lower()
            out["hostname"] = event_source
            if any(svc in event_source for svc in AWS_AI_EVENT_SOURCES):
                out["text_hints"] = sorted(set(out["text_hints"]) | {f"aws:{event_source}"})
            un_match = _CT_USERNAME_RE.search(e)
            if un_match:
                out["user"] = un_match.group(1)
        elif e.strip().startswith("{"):                          # Arctic Wolf / Sysmon JSON
            # FIX #2: regex directly on `e`, no json.loads()/json.dumps() round trip.
            hn = _AWN_HOSTNAME_RE.search(e)
            if hn:
                out["hostname"] = hn.group(1)
            un = _AWN_USERNAME_RE.search(e)
            if un:
                out["user"] = un.group(1)
            dh = _AWN_DEST_HOST_RE.search(e)
            if dh:
                out["domain"] = dh.group(1).lower()
            if not out["user"]:
                u = _AWN_BACKSLASH_USER_RE.search(e)
                if u:
                    out["user"] = u.group(1)
    except (AttributeError, ValueError):
        pass
    return out


def _normalize_tuple(args: tuple[str, str, str, str]) -> dict:
    """Top-level (picklable) wrapper so multiprocessing.Pool can call normalize_row."""
    return normalize_row(*args)


def _read_raw_rows(csv_path: Path):
    """FIX #4: header parsed once; rows read by column index, not DictReader."""
    with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=",")
        header = next(reader, None)
        if not header:
            return
        idx = {h.strip().lower(): i for i, h in enumerate(header)}
        i_sensor, i_source = idx.get("sensor"), idx.get("source")
        i_ts, i_event = idx.get("timestamp"), idx.get("event")

        def col(row, i):
            return row[i].strip() if i is not None and i < len(row) else ""

        for row in reader:
            yield (col(row, i_sensor), col(row, i_source), col(row, i_ts), col(row, i_event))


def load_and_normalize(csv_path: Path, workers: int | None = WORKERS) -> list[dict]:
    """FIX #5: parallel normalization across CPU cores for large files."""
    raw_rows = list(_read_raw_rows(csv_path))
    n = len(raw_rows)
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)

    if workers <= 1 or n < PARALLEL_THRESHOLD_ROWS:
        return [normalize_row(*r) for r in raw_rows]

    chunksize = max(200, n // (workers * 8))
    with mp.Pool(workers) as pool:
        return list(pool.imap(_normalize_tuple, raw_rows, chunksize=chunksize))


# ---------------- INTELLIGENT CHUNKING (unchanged logic) ----------------
def parse_ts(ts_str: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts_str.strip().replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def group_by_entity(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["sensor"], row["source"]), []).append(row)
    return groups


def split_sessions(rows: list[dict]) -> list[list[dict]]:
    sessions, current, previous = [], [], None
    for row in rows:
        ts = parse_ts(row["timestamp"])
        if previous and ts and (ts - previous).total_seconds() > SESSION_GAP_SECONDS:
            sessions.append(current)
            current = []
        current.append(row)
        if ts:
            previous = ts
    if current:
        sessions.append(current)
    return sessions


def make_chunks(session_rows: list[dict]) -> list[list[dict]]:
    chunks, start = [], 0
    n = len(session_rows)
    overlap_n = max(1, math.floor(MAX_EVENTS_PER_CHUNK * OVERLAP_RATIO)) if n > MAX_EVENTS_PER_CHUNK else 0
    while start < n:
        end = min(start + MAX_EVENTS_PER_CHUNK, n)
        chunks.append(session_rows[start:end])
        if end == n:
            break
        start = end - overlap_n
    return chunks


def build_intelligent_chunks(rows: list[dict]) -> list[dict]:
    chunks = []
    for (sensor, source), group_rows in group_by_entity(rows).items():
        for session_rows in split_sessions(group_rows):
            for chunk_rows in make_chunks(session_rows):
                chunks.append({
                    "sensor": sensor, "source": source,
                    "start_ts": chunk_rows[0]["timestamp"], "end_ts": chunk_rows[-1]["timestamp"],
                    "event_count": len(chunk_rows), "events": chunk_rows,
                })
    return chunks


# ---------------- DETECT (domain -> application), per chunk ----------------
AI_APPLICATIONS = {
    "chatgpt.com": ("ChatGPT", False), "chat.openai.com": ("ChatGPT", False),
    "openai.com": ("ChatGPT", False), "oaistatic.com": ("ChatGPT", False),
    "oaiusercontent.com": ("ChatGPT", False),
    "claude.ai": ("Claude", False), "anthropic.com": ("Claude", False),
    "gemini.google.com": ("Gemini", False),
    "generativelanguage.googleapis.com": ("Gemini", False), "bard.google.com": ("Gemini", False),
    "perplexity.ai": ("Perplexity", False), "cursor.com": ("Cursor", False),
    "huggingface.co": ("Hugging Face", False),
    "cohere.ai": ("Cohere", False), "cohere.com": ("Cohere", False),
    "mistral.ai": ("Mistral", False), "poe.com": ("Poe", False),
    "character.ai": ("Character.AI", False), "deepseek.com": ("DeepSeek", False),
    "x.ai": ("Grok/x.ai", False), "grok.com": ("Grok/x.ai", False),
    "replicate.com": ("Replicate", False), "together.ai": ("Together AI", False),
    "groq.com": ("Groq", False), "fireworks.ai": ("Fireworks AI", False),
    "copilot.microsoft.com": ("Microsoft Copilot", True),
    "githubcopilot.com": ("GitHub Copilot", True),
    "bedrock-runtime": ("AWS Bedrock", False), "sagemaker": ("AWS SageMaker", False),
    "openai.azure.com": ("Azure OpenAI", False),
    "jasper.ai": ("Jasper AI", False), "writesonic.com": ("Writesonic", False),
    "copy.ai": ("Copy.ai", False), "elevenlabs.io": ("ElevenLabs", False),
    "runwayml.com": ("Runway ML", False), "midjourney.com": ("Midjourney", False),
    "stability.ai": ("Stability AI", False),
    "intercom.io": ("Intercom", False), "drift.com": ("Drift", False),
    "zdassets.com": ("Zendesk Chat", False), "zopim.com": ("Zendesk Chat", False),
    "tidio.co": ("Tidio", False), "tidiochat.com": ("Tidio", False),
    "crisp.chat": ("Crisp", False), "livechatinc.com": ("LiveChat", False),
    "freshchat.com": ("Freshchat", False), "manychat.com": ("ManyChat", False),
    "chatfuel.com": ("Chatfuel", False), "landbot.io": ("Landbot", False),
    "kore.ai": ("Kore.ai", False), "yellow.ai": ("Yellow.ai", False),
    "haptik.ai": ("Haptik", False), "ada.cx": ("Ada", False),
    "verloop.io": ("Verloop", False), "mobilemonkey.com": ("MobileMonkey", False),
    "botpress.cloud": ("Botpress", False), "voiceflow.com": ("Voiceflow", False),
    "dialogflow.cloud.goog": ("Dialogflow", False),
    "watsonassistant.watson.cloud.ibm.com": ("Watson Assistant", False),
    "rasa.com": ("Rasa", False), "chatbase.co": ("Chatbase", False),
}


def domain_matches(domain: str, known: str) -> bool:
    if not domain:
        return False
    domain = domain.rstrip(".")
    return domain == known or domain.endswith("." + known)


def detect_ai_in_chunk(chunk: dict) -> dict:
    matched_apps = {}
    text_hint_matches = set()
    unknown_domains = set()
    total_bytes = 0

    for event in chunk["events"]:
        domain = (event.get("domain") or "").lower().strip()
        total_bytes += event.get("bytes", 0)

        matched_this_event = False
        if domain:
            for known, (app, approved) in AI_APPLICATIONS.items():
                if domain_matches(domain, known):
                    matched_apps[app] = approved
                    matched_this_event = True
                    break

        for hint in event.get("text_hints") or []:
            text_hint_matches.add(hint)
            matched_this_event = True

        if domain and not matched_this_event:
            unknown_domains.add(domain)

    chunk["applications"] = matched_apps
    chunk["text_hints"] = sorted(text_hint_matches)
    chunk["unknown_domains"] = sorted(unknown_domains)
    chunk["total_bytes"] = total_bytes
    chunk["is_ai_related"] = bool(matched_apps or text_hint_matches)
    chunk["needs_review"] = bool(unknown_domains) and not chunk["is_ai_related"]
    chunk["match_tier"] = "domain" if matched_apps else ("text_hint" if text_hint_matches else "none")
    return chunk


# ---------------- RAG retrieval — FIX #6: cached by matched app(s) ----------------
@lru_cache(maxsize=512)
def _retrieve_context_cached(apps_key: tuple[str, ...]) -> str:
    apps = ", ".join(apps_key) or "an AI tool"
    queries = [
        f"Is {apps} an approved AI tool and what is the policy on using {apps}?",
        f"Security risk of uploading business data to {apps}",
        "Data classification and DLP rules for external generative AI services",
        f"Severity and recommended response action for shadow AI use of {apps}",
    ][:CONTEXT_QUERIES_PER_EVENT]

    seen, passages = set(), []
    for q in queries:
        res = knowledge.query(query_texts=[q], n_results=RESULTS_PER_QUERY)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for d, m in zip(docs, metas):
            key = d[:100]
            if key not in seen:
                seen.add(key)
                passages.append(f"[source: {m.get('source', 'policy')}] {d}")
    return "\n\n".join(passages)


def retrieve_context(chunk: dict) -> str:
    # Cache key drops the exact byte count (see docstring #6) so chunks that
    # match the same app(s) reuse one cached retrieval instead of re-querying.
    apps_key = tuple(sorted(chunk["applications"].keys())) or tuple(sorted(chunk["text_hints"])) or ("an AI tool",)
    return _retrieve_context_cached(apps_key)


# ---------------- qwen verdict (unchanged) ----------------
SYSTEM_PROMPT = """You are a Shadow AI Security Analyst.

You are given a CHUNK of grouped network/endpoint events (same sensor/source,
one session) plus retrieved security policy, and a match_tier field:
  - match_tier="domain": at least one event matched a KNOWN AI/chatbot domain
    exactly or as a proper subdomain. Treat this as strong evidence.
  - match_tier="text_hint": no domain was confirmed, the chunk only matched a
    generic text pattern (an API path, keyword, or category tag). This is
    weaker evidence — verify from the actual event content that this really
    reflects a conversational AI / chatbot / generative-AI service before
    calling it a positive. If the evidence is thin or could plausibly be
    something else, say so and lower your confidence rather than guessing.

A finding only qualifies as Shadow AI chatbot/assistant usage if the evidence
shows communication with an external conversational AI, AI assistant, or
generative-AI-backed tool — not ordinary web browsing, routine customer-support
ticketing, or generic bot/crawler traffic unrelated to AI.

You must:
1. Identify potential Shadow AI activity, naming the specific platform if identifiable.
2. Determine whether the application(s) are approved (use the retrieved policy).
3. Identify security risks and data exposure (total bytes transferred).
4. Apply the organizational policies provided.
5. Determine severity and recommend an action consistent with policy.
6. Provide evidence taken ONLY from the events in this chunk.

Do not invent facts. Use ONLY the chunk and the retrieved knowledge.
Return ONLY a valid JSON object, no other text, no markdown."""


def analyze_chunk(chunk: dict, rag_context: str) -> dict:
    events_summary = json.dumps(
        [{"timestamp": e["timestamp"], "domain": e.get("domain"),
          "user": e.get("user"), "bytes": e.get("bytes", 0)} for e in chunk["events"]],
        ensure_ascii=False, indent=2,
    )
    prompt = f"""CHUNK (sensor={chunk['sensor']}, source={chunk['source']}, \
{chunk['start_ts']} to {chunk['end_ts']}, {chunk['event_count']} events, \
match_tier={chunk['match_tier']}, \
applications seen={list(chunk['applications'].keys())}, \
text-pattern hints={chunk['text_hints']}):
{events_summary}

RETRIEVED SECURITY POLICY (use this to judge approval, severity, and action):
{rag_context or "(no policy retrieved — flag low confidence)"}

Return JSON with keys:
classification, risk, confidence, application, user, reason, evidence,
policy, severity, recommended_action"""

    response = ollama_client.chat(
        model=ANALYSIS_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0},
    )
    raw = response["message"]["content"].strip()
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"classification": None, "confidence": "low",
                "reason": "LLM returned unparseable output", "parse_error": raw[:300]}


# ---------------- MAIN ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="raw log CSV (timestamp,sensor,source,event)")
    ap.add_argument("--workers", type=int, default=None, help="override CPU worker count")
    args = ap.parse_args()

    if knowledge.count() == 0:
        raise SystemExit("Knowledge base is EMPTY. Run: python build_knowledge_base.py")
    print(f"Knowledge RAG: {knowledge.count()} policy chunks")

    t0 = time.perf_counter()
    rows = load_and_normalize(args.csv, workers=args.workers)
    t1 = time.perf_counter()
    print(f"Normalized {len(rows)} log rows from {args.csv.name} in {t1 - t0:.1f}s")

    chunks = build_intelligent_chunks(rows)
    t2 = time.perf_counter()
    print(f"Built {len(chunks)} intelligent chunks in {t2 - t1:.1f}s")

    chunks = [detect_ai_in_chunk(c) for c in chunks]
    ai_chunks = [c for c in chunks if c["is_ai_related"]]
    review_chunks = [c for c in chunks if c["needs_review"]]
    domain_tier = sum(1 for c in ai_chunks if c["match_tier"] == "domain")
    hint_tier = len(ai_chunks) - domain_tier
    t3 = time.perf_counter()
    print(f"Detection pass: {t3 - t2:.1f}s. {len(ai_chunks)} / {len(chunks)} chunks flagged "
          f"({domain_tier} by domain, {hint_tier} by text-hint only)")
    if review_chunks:
        print(f"{len(review_chunks)} chunks have unrecognized external domains — see 'needs_review' in output\n")
    else:
        print()

    findings = []
    for chunk in ai_chunks:
        context = retrieve_context(chunk)
        result = analyze_chunk(chunk, context)
        result.update({
            "_sensor": chunk["sensor"], "_source": chunk["source"],
            "_start_ts": chunk["start_ts"], "_end_ts": chunk["end_ts"],
            "_applications": list(chunk["applications"].keys()),
            "_approved": chunk["applications"], "_match_tier": chunk["match_tier"],
            "_bytes": chunk["total_bytes"], "_event_count": chunk["event_count"],
        })
        findings.append(result)
        print(f"  {list(chunk['applications'].keys())!s:30} tier={chunk['match_tier']:10} "
              f"{chunk['total_bytes']:>9}B  sev={result.get('severity')}  action={result.get('recommended_action')}")
    t4 = time.perf_counter()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in findings:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nRAG+qwen pass: {t4 - t3:.1f}s. Done. {len(findings)} findings -> {OUTPUT_FILE}")
    print(f"Total: {t4 - t0:.1f}s")

    if review_chunks:
        review_file = "needs_review.jsonl"
        with open(review_file, "w", encoding="utf-8") as f:
            for c in review_chunks:
                f.write(json.dumps({
                    "sensor": c["sensor"], "source": c["source"],
                    "start_ts": c["start_ts"], "end_ts": c["end_ts"],
                    "unknown_domains": c["unknown_domains"], "event_count": c["event_count"],
                }, ensure_ascii=False) + "\n")
        print(f"{len(review_chunks)} chunks with unrecognized domains -> {review_file}")


if __name__ == "__main__":
    main()