"""
DROP-IN replacement for analyze_chunk() in rag_analyze.py, plus a standalone
smoke test you can run FIRST to prove qwen works at all.

Run the smoke test on its own:
    python this_file.py
It sends one obvious ChatGPT log line to qwen and prints the raw response.
If that comes back clean or errors, your problem is qwen/Ollama, not retrieval.
"""

import json
import requests

TOWER_IP = "192.168.100.100"          # <-- set to Tower's LAN IP
OLLAMA_CHAT_URL = f"http://{TOWER_IP}:11434/api/chat"
ANALYSIS_MODEL = "qwen2.5:14b"

ANALYSIS_SYSTEM_PROMPT = """You are a security analyst reviewing log chunks for evidence of AI tool usage
(e.g. ChatGPT, Claude, Gemini, Copilot, or other AI/LLM services) inside an organization's network.

Given one log chunk, respond ONLY with a JSON object, no other text:
{
  "ai_usage_detected": true or false,
  "service": "name of AI tool/service if identified, else null",
  "evidence": "short quote or description of the specific log lines that support this",
  "confidence": "high" | "medium" | "low",
  "risk_notes": "brief note on why this matters, or null if not applicable"
}"""


def analyze_chunk(chunk_text: str) -> dict:
    """Same contract as before, but distinguishes HTTP failure from parse failure
    so 'model said no' can never be confused with 'the call broke'."""
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": ANALYSIS_MODEL,
                "messages": [
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk_text},
                ],
                "stream": False,
                "options": {"temperature": 0},
                "format": "json",
            },
            timeout=180,
        )
    except requests.RequestException as exc:
        # network / connection refused / timeout — NOT a "clean" verdict
        return {"ai_usage_detected": None, "service": None, "evidence": None,
                "confidence": None, "risk_notes": f"HTTP_ERROR: {exc}"}

    if response.status_code != 200:
        # model missing, bad request, etc. — surfaces instead of silently passing
        return {"ai_usage_detected": None, "service": None, "evidence": None,
                "confidence": None, "risk_notes": f"STATUS_{response.status_code}: {response.text[:300]}"}

    raw = response.json().get("message", {}).get("content", "").strip()
    if not raw:
        return {"ai_usage_detected": None, "service": None, "evidence": None,
                "confidence": None, "risk_notes": "EMPTY_RESPONSE"}

    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"ai_usage_detected": None, "service": None, "evidence": None,
                "confidence": None, "risk_notes": f"PARSE_ERROR: {raw[:300]}"}

    # normalize a stringy "true"/"false" if the model returns text
    v = parsed.get("ai_usage_detected")
    if isinstance(v, str):
        parsed["ai_usage_detected"] = v.strip().lower() in ("true", "yes", "1")
    return parsed


if __name__ == "__main__":
    # ---- SMOKE TEST: one blatant ChatGPT line. qwen MUST flag this. ----
    obvious = (
        "[SENSOR=US Sensor] [SOURCE=192.168.101.1] [EVENT_COUNT=1]\n"
        '2026-08-02T23:04:40+00:00 | id=firewall msg="Web site hit" '
        "src=192.168.101.58:60128 dst=104.18.32.47:443 dstname=chatgpt.com "
        'arg=/backend-api/conversation sent=1842033 Category="Artificial Intelligence"'
    )
    print(f"Testing {ANALYSIS_MODEL} at {OLLAMA_CHAT_URL} ...\n")
    result = analyze_chunk(obvious)
    print(json.dumps(result, indent=2))
    print()
    verdict = result.get("ai_usage_detected")
    if verdict is True:
        print("PASS - qwen works. Your problem is retrieval/seed queries, not the model.")
    elif verdict is None:
        print("BROKEN - the call/parse failed. Read risk_notes above - that's your real error.")
    else:
        print("MODEL SAID NO on a blatant hit - model too weak or prompt issue. Try a stronger/less-quantized model.")