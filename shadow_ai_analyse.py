"""


Reads normalized events, detects AI usage by domain (deterministic — nothing
semantic needed on the logs), pulls RICH policy context from the knowledge RAG
(built by build_knowledge_base.py, NOT hardcoded), and asks the analyst LLM for a
JSON verdict grounded in that policy. Writes findings.jsonl.

Prereqs:
  - normalized events at EVENT_FILE (one JSON per line, must have a "domain" field)
  - knowledge RAG already built:  python build_knowledge_base.py
  - Tower running Ollama with nomic + qwen3:8b pulled

    pip install chromadb ollama
    python shadow_ai_analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from ollama import Client


# ---------------- CONFIG ----------------
TOWER_IP = "192.168.100.100"        # where Ollama runs. "localhost" if it's this PC.
OLLAMA_BASE = f"http://{TOWER_IP}:11434"

CHROMA_PATH = "./chroma_db"
KNOWLEDGE_COLLECTION = "shadow_ai_knowledge"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"
ANALYSIS_MODEL = "qwen3:8b"

EVENT_FILE = "data/normalized/events.jsonl"
OUTPUT_FILE = "findings.jsonl"

CONTEXT_QUERIES_PER_EVENT = 4       # how many angles to retrieve policy from
RESULTS_PER_QUERY = 3


# ---------------- CHROMA + OLLAMA ----------------
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


# ---------------- DETECTION (domain -> application) ----------------
# suffix match: api.openai.com, cdn.oaistatic.com, bedrock-runtime.us-east-1.amazonaws.com all resolve.
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


def detect_ai(event: dict) -> dict:
    """Resolve the AI application from the event's domain (suffix match).
    Deterministic: no LLM, no embedding — this is what makes detection reliable."""
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


# ---------------- RAG: rich multi-angle retrieval ----------------
def retrieve_context(event: dict) -> str:
    """Pull policy from several angles and merge unique passages, so the LLM gets
    broad, well-grounded context instead of a single lucky match."""
    app = event.get("application", "an AI tool")
    total_bytes = event.get("total_bytes", event.get("bytes", 0))
    queries = [
        f"Is {app} an approved AI tool and what is the policy on using {app}?",
        f"Security risk of uploading {total_bytes} bytes of business data to {app}",
        "Data classification and DLP rules for external generative AI services",
        f"Severity level and recommended response action for shadow AI use of {app}",
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


# ---------------- LLM ANALYST ----------------
SYSTEM_PROMPT = """You are a Shadow AI Security Analyst.

You analyze endpoint and network activity involving AI applications. You must:
1. Identify potential Shadow AI activity.
2. Determine whether the AI application is approved (use the retrieved policy).
3. Identify security risks.
4. Consider data exposure (bytes transferred).
5. Apply the organizational policies provided.
6. Determine severity.
7. Provide evidence taken only from the event.
8. Recommend an action consistent with the policy.

Do not invent facts. Use ONLY the event and the retrieved knowledge.
Return ONLY a valid JSON object, no other text, no markdown."""


def analyze_event(event: dict, rag_context: str) -> dict:
    prompt = f"""EVENT:
{json.dumps(event, ensure_ascii=False, indent=2)}

RETRIEVED SECURITY POLICY (use this to judge approval, severity, and action):
{rag_context or "(no policy retrieved — flag low confidence)"}

Return JSON with keys:
classification, risk, confidence, application, user, reason, evidence,
policy, severity, recommended_action"""

    response = ollama_client.chat(
        model=ANALYSIS_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": 0},
    )
    raw = response["message"]["content"].strip()
    if "<think>" in raw:                       # qwen3 reasoning block
        raw = raw.split("</think>")[-1].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"classification": None, "risk": None, "confidence": "low",
                "reason": "LLM returned unparseable output",
                "parse_error": raw[:300]}


# ---------------- MAIN ----------------
def load_events() -> list[dict]:
    path = Path(EVENT_FILE)
    if not path.exists():
        raise SystemExit(f"{EVENT_FILE} not found. Point EVENT_FILE at your normalized events.")
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def main() -> None:
    if knowledge.count() == 0:
        raise SystemExit("Knowledge base is EMPTY. Run: python build_knowledge_base.py")
    print(f"Knowledge RAG: {knowledge.count()} policy chunks\n")

    events = load_events()
    print(f"Loaded {len(events)} events\n")

    findings = []
    for event in events:
        event = detect_ai(event)
        if event["application"] == "Unknown":
            continue                            # not AI — skip

        context = retrieve_context(event)       # rich policy grounding
        result = analyze_event(event, context)  # event + policy -> verdict

        result.update({
            "_domain": event.get("domain"),
            "_application": event["application"],
            "_approved": event["approved"],
            "_user": event.get("user"),
            "_timestamp": event.get("timestamp"),
            "_bytes": event.get("total_bytes", event.get("bytes", 0)),
        })
        findings.append(result)
        print(f"  {event['application']:18} sev={result.get('severity')}  "
              f"action={result.get('recommended_action')}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in findings:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(findings)} AI events analyzed -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()