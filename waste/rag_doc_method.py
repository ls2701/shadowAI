"""


Pipeline recap:
  upload_to_tower.py  -> Tower chunks + embeds -> download
  ingest_to_chroma.py -> stores LOG chunks in Chroma  (collection: log_collection)
  THIS FILE           -> (a) builds a KNOWLEDGE base of security policy,
                         (b) reads normalized events,
                         (c) enriches each with the AI app it hit,
                         (d) retrieves relevant policy from the knowledge base,
                         (e) asks the analyst LLM for a JSON verdict,
                         (f) writes findings.jsonl

Two Chroma collections, on purpose:
  - log_collection      : the actual log chunks (built by ingest_to_chroma.py)
  - shadow_ai_knowledge : security policy / approved-tool list (seeded by THIS file)
The analyst reasons about a log event USING the retrieved policy. If the knowledge
base is empty, retrieval returns nothing and verdicts are guesses — so this file
seeds it automatically on first run (idempotent upsert; safe to run repeatedly).

SETUP (run once):
    pip install chromadb ollama
    # Tower must be running Ollama with these models pulled:
    #   ollama pull nomic-embed-text-v2-moe:latest
    #   ollama pull qwen3:8b

USAGE:
    python shadow_ai_analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from ollama import Client


# ---------------- CONFIG ----------------
TOWER_IP = "192.168.100.100"        # where Ollama runs. Use "localhost" if it's this PC.
OLLAMA_BASE = f"http://{TOWER_IP}:11434"

CHROMA_PATH = "./chroma_db"         # same folder ingest_to_chroma.py wrote to
KNOWLEDGE_COLLECTION = "shadow_ai_knowledge"

EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
ANALYSIS_MODEL = "qwen3:8b"

EVENT_FILE = "data/normalized/events.jsonl"
OUTPUT_FILE = "findings.jsonl"


# ---------------- CHROMA + EMBEDDING ----------------
client = chromadb.PersistentClient(path=CHROMA_PATH)

embedding_function = OllamaEmbeddingFunction(
    url=f"{OLLAMA_BASE}/api/embeddings",
    model_name=EMBEDDING_MODEL,
)

knowledge = client.get_or_create_collection(
    name=KNOWLEDGE_COLLECTION,
    embedding_function=embedding_function,
)

ollama_client = Client(host=OLLAMA_BASE)


# ---------------- KNOWLEDGE BASE (seeded once) ----------------
# Edit these to match YOUR org's real policy. Each entry becomes one retrievable doc.
POLICY_DOCS = [
    ("policy-approved-tools",
     "Approved AI tools: Microsoft Copilot (enterprise tenant) is approved for internal "
     "use. GitHub Copilot is approved for engineering only. No other generative AI tool "
     "is approved for handling company or customer data.",
     {"type": "policy", "topic": "approved_tools"}),

    ("policy-unapproved-tools",
     "Unapproved / shadow AI tools: ChatGPT (chatgpt.com, chat.openai.com), Claude "
     "(claude.ai), Google Gemini (gemini.google.com), Perplexity (perplexity.ai), and "
     "Cursor (cursor.com) are NOT approved. Any business data sent to these is a policy "
     "violation and a potential data-exposure incident.",
     {"type": "policy", "topic": "unapproved_tools"}),

    ("policy-data-classification",
     "Data handling: source code, customer PII, financial records, and credentials are "
     "Confidential. Confidential data must never be uploaded to external AI services. "
     "Large outbound transfers (>100 KB) to an AI endpoint indicate possible bulk data "
     "exfiltration and must be escalated.",
     {"type": "policy", "topic": "data_classification"}),

    ("policy-severity-rubric",
     "Severity rubric: CRITICAL = confidential data or >1 MB uploaded to an unapproved "
     "AI tool. HIGH = any upload/POST to an unapproved generative AI tool. MEDIUM = "
     "browsing/DNS to an unapproved AI tool with little or no data sent. LOW = use of an "
     "approved tool, or an AI-adjacent domain with no data transfer.",
     {"type": "policy", "topic": "severity"}),

    ("policy-response-actions",
     "Recommended actions: for HIGH/CRITICAL, notify the user's manager and the security "
     "team, capture the session, and block the destination at the proxy. For MEDIUM, warn "
     "the user and log for trend analysis. For LOW, monitor only.",
     {"type": "policy", "topic": "response"}),

    ("kb-chatgpt", "ChatGPT (chatgpt.com) is a generative AI chat tool by OpenAI. The path "
     "/backend-api/conversation carries prompt and file-upload payloads. Unapproved.",
     {"type": "knowledge", "application": "ChatGPT"}),
    ("kb-claude", "Claude (claude.ai, api.anthropic.com) is a generative AI assistant by "
     "Anthropic. The API path /v1/messages carries prompt payloads. Unapproved.",
     {"type": "knowledge", "application": "Claude"}),
    ("kb-gemini", "Google Gemini (gemini.google.com, generativelanguage.googleapis.com) is "
     "a generative AI tool by Google. Unapproved for business data.",
     {"type": "knowledge", "application": "Gemini"}),
    ("kb-perplexity", "Perplexity (perplexity.ai, api.perplexity.ai) is an AI search tool. "
     "The path /chat/completions carries prompt payloads. Unapproved.",
     {"type": "knowledge", "application": "Perplexity"}),
    ("kb-copilot", "Microsoft Copilot (copilot.microsoft.com) is approved on the enterprise "
     "tenant. Personal-account use is not covered by the enterprise data agreement.",
     {"type": "knowledge", "application": "Copilot"}),
    ("kb-bedrock", "AWS Bedrock (bedrock-runtime.*.amazonaws.com) hosts LLMs incl. Claude. "
     "The path /model/*/invoke carries inference payloads. Treat as unapproved AI use "
     "unless the workload is sanctioned.",
     {"type": "knowledge", "application": "AWS Bedrock"}),
]


def seed_knowledge() -> None:
    """Idempotent: upserts policy docs so retrieval always has content."""
    knowledge.upsert(
        ids=[d[0] for d in POLICY_DOCS],
        documents=[d[1] for d in POLICY_DOCS],
        metadatas=[d[2] for d in POLICY_DOCS],
    )


# ---------------- ENRICHMENT (domain -> application) ----------------
# suffix match so api.openai.com, cdn.oaistatic.com, sub.perplexity.ai all resolve.
AI_APPLICATIONS = {
    "chatgpt.com":       ("ChatGPT", "generative_ai", False),
    "chat.openai.com":   ("ChatGPT", "generative_ai", False),
    "openai.com":        ("ChatGPT", "generative_ai", False),
    "oaistatic.com":     ("ChatGPT", "generative_ai", False),
    "claude.ai":         ("Claude", "generative_ai", False),
    "anthropic.com":     ("Claude", "generative_ai", False),
    "gemini.google.com": ("Gemini", "generative_ai", False),
    "generativelanguage.googleapis.com": ("Gemini", "generative_ai", False),
    "perplexity.ai":     ("Perplexity", "ai_search", False),
    "cursor.com":        ("Cursor", "ai_coding", False),
    "huggingface.co":    ("Hugging Face", "ml_platform", False),
    "copilot.microsoft.com": ("Microsoft Copilot", "generative_ai", True),   # approved
    "githubcopilot.com": ("GitHub Copilot", "ai_coding", True),              # approved
    "bedrock-runtime":   ("AWS Bedrock", "ml_inference", False),
}


def enrich_event(event: dict) -> dict:
    """Resolve the AI application from the event's domain (suffix match)."""
    domain = (event.get("domain") or "").lower().strip()
    app, category, approved = "Unknown", "unknown", None
    if domain:
        for known, (a, c, ok) in AI_APPLICATIONS.items():
            if domain == known or domain.endswith("." + known) or known in domain:
                app, category, approved = a, c, ok
                break
    event["application"] = app
    event["category"] = category
    event["approved"] = approved
    return event


# ---------------- 8.3  RETRIEVE CONTEXT ----------------
def retrieve_context(event: dict, n_results: int = 5) -> str:
    """Query the knowledge base for policy relevant to THIS event, per section 8.3."""
    total_bytes = event.get("total_bytes", event.get("bytes", 0))
    query = (
        f"Evaluate security risk of {event.get('application', 'an AI tool')} being used "
        f"to upload {total_bytes} bytes of business data. "
        f"Is this tool approved? What are the relevant policy restrictions?"
    )
    results = knowledge.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [[]])
    return "\n\n".join(docs[0]) if docs and docs[0] else ""


# ---------------- LLM ANALYST ----------------
SYSTEM_PROMPT = """You are a Shadow AI Security Analyst.

You analyze endpoint and network activity involving AI applications. You must:
1. Identify potential Shadow AI activity.
2. Determine whether the AI application is approved.
3. Identify security risks.
4. Consider data exposure.
5. Consider organizational policies.
6. Determine severity.
7. Provide evidence.
8. Recommend an action.

Do not invent facts. Only use evidence supplied in the event and the retrieved
knowledge. Return ONLY a valid JSON object, no other text, no markdown."""


def analyze_event(event: dict, rag_context: str) -> dict:
    prompt = f"""EVENT:
{json.dumps(event, ensure_ascii=False, indent=2)}

RETRIEVED SECURITY KNOWLEDGE:
{rag_context or "(no policy retrieved)"}

Analyze this event. Return JSON with these keys:
classification, risk, confidence, application, user, reason, evidence,
policy, severity, recommended_action"""

    response = ollama_client.chat(
        model=ANALYSIS_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        format="json",                 # force JSON so parsing is reliable
        options={"temperature": 0},
    )
    raw = response["message"]["content"].strip()

    # qwen3 can emit <think>...</think> before the JSON — strip it defensively
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"classification": None, "risk": None, "confidence": None,
                "reason": None, "evidence": None,
                "parse_error": raw[:300]}


# ---------------- MAIN ----------------
def load_events() -> list[dict]:
    path = Path(EVENT_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"{EVENT_FILE} not found. Point EVENT_FILE at your normalized events "
            f".jsonl (one JSON object per line with at least a 'domain' field)."
        )
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def main() -> None:
    print("Seeding knowledge base...")
    seed_knowledge()
    print(f"Knowledge base has {knowledge.count()} policy/knowledge docs\n")

    events = load_events()
    print(f"Loaded {len(events)} events from {EVENT_FILE}\n")

    findings = []
    for event in events:
        event = enrich_event(event)

        if event["application"] == "Unknown":
            continue                       # skip non-AI events

        print(f"Analyzing: {event['application']}"
              f"  (approved={event['approved']})")

        context = retrieve_context(event)
        result = analyze_event(event, context)

        result["_domain"] = event.get("domain")
        result["_application"] = event["application"]
        result["_approved"] = event["approved"]
        result["_timestamp"] = event.get("timestamp")
        findings.append(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in findings:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Analyzed {len(findings)} AI events -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()