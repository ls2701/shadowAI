"""
DROP-IN replacement for the candidate-selection part of rag_analyze.py
(the block from "seen_ids = set()" down to the end of the SEED_QUERIES loop).

Why: pure semantic search misses AI hits because the signal (chatgpt.com) is a
handful of chars buried in ~700 chars of firewall noise, so the chunk's embedding
is dominated by "generic log" and never ranks near the seed query. Fix = hybrid:

  1. KEYWORD PASS  - substring/regex match on a known-AI list. 100% recall on
     anything named, instant, no embeddings. This is what actually catches ChatGPT.
  2. SEMANTIC PASS - your existing cosine search, kept ONLY to surface unknown
     / shadow-AI traffic that isn't on the keyword list.
  3. Union both, dedupe, send all to qwen.

Paste AI_DOMAINS + select_candidates() into rag_analyze.py and call
select_candidates() where you currently build `candidates`.
"""

import re

# Known AI / LLM services. Add your own as you find them. Lowercased substring match.
AI_DOMAINS = [
    "chatgpt.com", "chat.openai.com", "openai.com", "oaistatic.com", "oaiusercontent.com",
    "api.anthropic.com", "anthropic.com", "claude.ai",
    "gemini.google.com", "generativelanguage.googleapis.com", "bard.google.com",
    "copilot.microsoft.com", "githubcopilot.com", "copilot.github.com",
    "perplexity.ai", "huggingface.co", "cohere.ai", "cohere.com",
    "mistral.ai", "poe.com", "character.ai", "deepseek.com", "x.ai", "grok",
    "bedrock-runtime", "bedrock.", "sagemaker",   # AWS-hosted LLMs (no 'ai' in host)
    "azure-api.net", "openai.azure.com",           # Azure OpenAI
    "replicate.com", "together.ai", "groq.com", "fireworks.ai",
]

# Category/keyword hints that appear in the log text itself (not just domains).
AI_TEXT_HINTS = [
    'category="artificial intelligence"',
    "artificial intelligence",
    "/backend-api/conversation",   # ChatGPT
    "/v1/messages",                # Anthropic
    "/chat/completions",           # OpenAI-compatible APIs
    "/v1/complete",
    "generativelanguage",
    "anthropic.claude",            # Bedrock model id
]

_AI_NEEDLES = [n.lower() for n in AI_DOMAINS + AI_TEXT_HINTS]


def keyword_hit(text: str) -> list[str]:
    """Return which AI needles appear in the chunk text (empty list = no match)."""
    low = text.lower()
    return [n for n in _AI_NEEDLES if n in low]


def select_candidates(chunks, chunk_embeddings, norms, embed_text_fn,
                      seed_queries, top_k_per_query):
    """
    Returns (candidates, reason_by_id) where candidates is a list of chunk dicts
    and reason_by_id maps chunk id -> why it was picked (for the report).
    """
    import numpy as np

    seen = {}
    reason = {}

    # -------- PASS 1: deterministic keyword match (the reliable one) --------
    kw_count = 0
    for c in chunks:
        hits = keyword_hit(c["text"])
        if hits:
            seen[c["id"]] = c
            reason[c["id"]] = f"keyword:{','.join(sorted(set(hits))[:4])}"
            kw_count += 1
    print(f"Keyword pass: {kw_count} chunks matched a known-AI term")

    # -------- PASS 2: semantic sweep (for unknown/shadow AI only) --------
    sem_new = 0
    for query in seed_queries:
        q = embed_text_fn(query)
        sims = (chunk_embeddings @ q) / (norms * (np.linalg.norm(q) + 1e-8))
        for idx in np.argsort(sims)[::-1][:top_k_per_query]:
            c = chunks[idx]
            if c["id"] not in seen:
                seen[c["id"]] = c
                reason[c["id"]] = f"semantic:{sims[idx]:.3f}"
                sem_new += 1
    print(f"Semantic pass: {sem_new} additional chunks (not already keyword-matched)")

    candidates = list(seen.values())
    print(f"Total unique candidates -> {len(candidates)}")
    return candidates, reason