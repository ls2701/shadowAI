"""
ai_log_detector.py

Multi-signal Shadow-AI detector for network / proxy / EDR logs.

Design goals:
    - High recall without making weak signals decisive
    - Explainable detections
    - Provider-aware fingerprints
    - Cloud + local AI detection
    - Network + EDR + proxy + DNS + TLS evidence
    - Robust against malformed/missing fields
    - Avoid excessive double-counting of correlated signals
    - Configurable thresholds
    - Safe handling of arbitrary log input

Expected normalized row examples:

{
    "timestamp": "...",
    "src_ip": "...",
    "dst_ip": "...",
    "dst_port": 443,
    "domain": "api.openai.com",
    "url": "https://api.openai.com/v1/chat/completions",
    "url_path": "/v1/chat/completions",
    "method": "POST",
    "headers": {
        "authorization": "Bearer ...",
        "content-type": "application/json"
    },
    "user_agent": "OpenAI/Python 1.2.3",
    "process": "python.exe",
    "cmd": "python app.py",
    "body": "...",
    "ja3": "...",
    "ja4": "...",
    "dns_query": "api.openai.com",
    "sni": "api.openai.com"
}

Output:

{
    "verdict": "analyze",
    "score": 0.96,
    "confidence": 0.98,
    "severity": "HIGH",
    "providers": ["openai"],
    "categories": ["cloud-api", "llm"],
    "hits": [...],
    "evidence": [...]
}
"""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


# ============================================================
# CONFIG
# ============================================================

ANALYZE_THRESHOLD = 0.72
REVIEW_THRESHOLD = 0.38

MAX_STRING_LENGTH = 64_000
MAX_BODY_LENGTH = 16_000

LOGGER = logging.getLogger("ai_log_detector")


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Detection:
    """
    One piece of evidence.

    weight:
        Contribution before correlation adjustment.

    confidence:
        How trustworthy this particular signal is.

    category:
        cloud-api / local-ai / sdk / mcp / process / etc.

    provider:
        Optional provider such as openai, anthropic, ollama.
    """
    name: str
    weight: float
    severity: str
    reason: str
    category: str
    provider: Optional[str] = None
    confidence: float = 1.0


@dataclass
class DetectionResult:
    verdict: str
    score: float
    confidence: float
    severity: str
    providers: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    hits: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_str(value: Any, limit: int = MAX_STRING_LENGTH) -> str:
    if value is None:
        return ""

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if not isinstance(value, str):
        value = str(value)

    return value[:limit]


def _lower(value: Any) -> str:
    return _safe_str(value).strip().lower()


def _first(row: Dict[str, Any], *keys: str) -> Any:
    """
    Return the first non-empty value.
    """
    for key in keys:
        value = row.get(key)

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        return value

    return None


def _headers(row: Dict[str, Any]) -> Dict[str, str]:
    raw = row.get("headers") or {}

    if isinstance(raw, dict):
        return {
            _lower(k): _safe_str(v)
            for k, v in raw.items()
        }

    # Some proxy logs store headers as a single string.
    if isinstance(raw, str):
        result = {}

        for line in raw.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                result[_lower(key)] = value.strip()

        return result

    return {}


def _header_blob(row: Dict[str, Any]) -> str:
    headers = _headers(row)

    return " ".join(
        f"{key}:{value}"
        for key, value in headers.items()
    )


def _extract_host(value: Any) -> str:
    value = _safe_str(value).strip()

    if not value:
        return ""

    # URL
    if "://" in value:
        try:
            parsed = urlparse(value)
            return (parsed.hostname or "").rstrip(".").lower()
        except ValueError:
            return ""

    # host:port
    if value.count(":") == 1:
        host, port = value.rsplit(":", 1)

        if port.isdigit():
            value = host

    # IPv6
    value = value.strip("[]")

    return value.rstrip(".").lower()


def _host_of(row: Dict[str, Any]) -> str:
    candidates = (
        row.get("domain"),
        row.get("host"),
        row.get("sni"),
        row.get("server_name"),
    )

    for candidate in candidates:
        host = _extract_host(candidate)

        if host:
            return host

    url = _safe_str(row.get("url"))

    if url:
        try:
            return (urlparse(url).hostname or "").rstrip(".").lower()
        except ValueError:
            pass

    return ""


def _path_of(row: Dict[str, Any]) -> str:
    path = _first(row, "url_path", "path")

    if path:
        return _safe_str(path)

    url = _safe_str(row.get("url"))

    if url:
        try:
            return urlparse(url).path or "/"
        except ValueError:
            return ""

    return ""


def _port_of(row: Dict[str, Any]) -> Optional[int]:
    raw = _first(row, "dst_port", "dest_port", "port")

    try:
        if raw is None:
            return None

        port = int(raw)

        if 1 <= port <= 65535:
            return port

    except (TypeError, ValueError):
        pass

    return None


def _is_private_ip(value: Any) -> bool:
    try:
        return ipaddress.ip_address(_safe_str(value)).is_private
    except ValueError:
        return False


# ============================================================
# DOMAIN FINGERPRINTS
# ============================================================

PROVIDER_DOMAINS: Dict[str, set[str]] = {

    "openai": {
        "api.openai.com",
        "chat.openai.com",
        "openai.com",
    },

    "anthropic": {
        "api.anthropic.com",
        "claude.ai",
        "anthropic.com",
    },

    "google-ai": {
        "generativelanguage.googleapis.com",
        "gemini.google.com",
        "aiplatform.googleapis.com",
        "vertexai.googleapis.com",
    },

    "mistral": {
        "api.mistral.ai",
        "mistral.ai",
    },

    "cohere": {
        "api.cohere.ai",
        "api.cohere.com",
        "cohere.com",
    },

    "groq": {
        "api.groq.com",
        "groq.com",
    },

    "perplexity": {
        "api.perplexity.ai",
        "perplexity.ai",
    },

    "together": {
        "api.together.xyz",
        "together.ai",
    },

    "replicate": {
        "api.replicate.com",
        "replicate.com",
    },

    "huggingface": {
        "api.huggingface.co",
        "api-inference.huggingface.co",
        "huggingface.co",
    },

    "ai21": {
        "api.ai21.com",
        "ai21.com",
    },

    "deepseek": {
        "api.deepseek.com",
        "deepseek.com",
    },

    "moonshot": {
        "api.moonshot.cn",
        "moonshot.cn",
    },

    "openrouter": {
        "openrouter.ai",
        "api.openrouter.ai",
    },

    "github-copilot": {
        "copilot.github.com",
        "api.githubcopilot.com",
    },

    "cursor": {
        "cursor.sh",
        "api.cursor.sh",
    },

    "codeium": {
        "codeium.com",
        "api.codeium.com",
        "windsurf.com",
    },

    "sourcegraph": {
        "sourcegraph.com",
    },

    "tabnine": {
        "tabnine.com",
        "api.tabnine.com",
    },

    "supermaven": {
        "supermaven.com",
    },
}


# Local AI endpoints are deliberately separate.
LOCAL_AI_DOMAINS = {
    "localhost",
    "127.0.0.1",
    "::1",
    "host.docker.internal",
}


def _domain_provider(host: str) -> Optional[str]:
    if not host:
        return None

    for provider, domains in PROVIDER_DOMAINS.items():
        for domain in domains:
            if host == domain or host.endswith("." + domain):
                return provider

    return None


# ============================================================
# API PATH DETECTION
# ============================================================

PATH_PATTERNS: List[Tuple[str, re.Pattern[str], float, str]] = [

    (
        "openai-compatible-chat",
        re.compile(r"^/v1/chat/completions(?:/)?$", re.I),
        0.82,
        "llm-api",
    ),

    (
        "openai-compatible-completions",
        re.compile(r"^/v1/completions(?:/)?$", re.I),
        0.75,
        "llm-api",
    ),

    (
        "openai-embeddings",
        re.compile(r"^/v1/embeddings(?:/)?$", re.I),
        0.70,
        "llm-api",
    ),

    (
        "anthropic-messages",
        re.compile(r"^/v1/messages(?:/)?$", re.I),
        0.82,
        "llm-api",
    ),

    (
        "google-generate-content",
        re.compile(
            r"(?:^|/)models/[^/]+:generateContent$",
            re.I,
        ),
        0.82,
        "llm-api",
    ),

    (
        "google-stream-generate",
        re.compile(
            r"(?:^|/)models/[^/]+:streamGenerateContent$",
            re.I,
        ),
        0.85,
        "llm-api",
    ),

    (
        "ollama-chat",
        re.compile(r"^/api/chat/?$", re.I),
        0.85,
        "local-llm",
    ),

    (
        "ollama-generate",
        re.compile(r"^/api/generate/?$", re.I),
        0.85,
        "local-llm",
    ),

    (
        "ollama-tags",
        re.compile(r"^/api/tags/?$", re.I),
        0.60,
        "local-llm",
    ),

    (
        "ollama-embeddings",
        re.compile(r"^/api/(?:embed|embeddings)/?$", re.I),
        0.75,
        "local-llm",
    ),

    (
        "mcp",
        re.compile(r"(?:^|/)mcp(?:/sse)?/?$", re.I),
        0.82,
        "mcp",
    ),

    (
        "sse",
        re.compile(r"^/sse/?$", re.I),
        0.55,
        "mcp",
    ),

    (
        "model-invoke",
        re.compile(r"/model/[^/]+/invoke/?$", re.I),
        0.65,
        "llm-api",
    ),
]


# ============================================================
# HEADER FINGERPRINTS
# ============================================================

HEADER_PATTERNS = [

    (
        "stainless-sdk",
        re.compile(
            r"\bx-stainless-(?:lang|package-version|os|arch|runtime|retry-count)\b",
            re.I,
        ),
        0.80,
        "sdk",
        None,
    ),

    (
        "anthropic-version",
        re.compile(r"\banthropic-version\s*:", re.I),
        0.90,
        "sdk",
        "anthropic",
    ),

    (
        "anthropic-beta",
        re.compile(r"\banthropic-beta\s*:", re.I),
        0.75,
        "sdk",
        "anthropic",
    ),

    (
        "openai-organization",
        re.compile(r"\bopenai-organization\s*:", re.I),
        0.80,
        "sdk",
        "openai",
    ),

    (
        "openai-beta",
        re.compile(r"\bopenai-beta\s*:", re.I),
        0.80,
        "sdk",
        "openai",
    ),

    (
        "api-key-header",
        re.compile(r"\bx-api-key\s*:", re.I),
        0.45,
        "auth",
        None,
    ),

    (
        "bearer-ai-key",
        re.compile(
            r"\bauthorization\s*:\s*bearer\s+"
            r"(?:sk-|sk-ant-|sk-proj-|or-|xai-)",
            re.I,
        ),
        0.82,
        "auth",
        None,
    ),
]


# ============================================================
# USER AGENT / SDK DETECTION
# ============================================================

UA_PATTERNS = [

    ("openai-sdk", re.compile(r"\bopenai(?:[/\s-]|$)", re.I), 0.70, "openai"),
    ("anthropic-sdk", re.compile(r"\banthropic(?:[/\s-]|$)", re.I), 0.75, "anthropic"),

    (
        "google-generative-ai",
        re.compile(
            r"\b(?:google-generativeai|google-genai)\b",
            re.I,
        ),
        0.70,
        "google-ai",
    ),

    ("langchain", re.compile(r"\blangchain\b", re.I), 0.55, None),
    ("llama-index", re.compile(r"\bllama[-_ ]?index\b", re.I), 0.55, None),
    ("llama-cpp", re.compile(r"\bllama[-_]?cpp\b", re.I), 0.65, None),
    ("litellm", re.compile(r"\blitellm\b", re.I), 0.60, None),
    ("autogen", re.compile(r"\b(?:auto)?gen\b", re.I), 0.45, None),
    ("crewai", re.compile(r"\bcrewai\b", re.I), 0.55, None),
    ("smolagents", re.compile(r"\bsmolagents\b", re.I), 0.55, None),
    ("ollama", re.compile(r"\bollama\b", re.I), 0.75, "ollama"),
    ("aider", re.compile(r"\baider\b", re.I), 0.60, None),
    ("cursor", re.compile(r"\bcursor\b", re.I), 0.60, "cursor"),
    ("windsurf", re.compile(r"\bwindsurf\b", re.I), 0.60, "codeium"),
]


# ============================================================
# PROCESS / COMMAND LINE
# ============================================================

PROCESS_PATTERNS = [

    ("ollama", re.compile(r"\bollama(?:\.exe)?\b", re.I), 0.85, "ollama"),

    (
        "llama-server",
        re.compile(r"\bllama-server(?:\.exe)?\b", re.I),
        0.85,
        "local-llm",
    ),

    (
        "llama.cpp",
        re.compile(r"\bllama\.cpp\b", re.I),
        0.75,
        "local-llm",
    ),

    (
        "lm-studio",
        re.compile(r"\b(?:lm[- ]?studio|lmstudio)\b", re.I),
        0.85,
        "local-llm",
    ),

    ("jan", re.compile(r"\bjan(?:\.exe)?\b", re.I), 0.70, "local-llm"),

    (
        "anythingllm",
        re.compile(r"\banythingllm\b", re.I),
        0.80,
        "local-llm",
    ),

    (
        "open-webui",
        re.compile(r"\bopen[-_ ]?webui\b", re.I),
        0.75,
        "local-llm",
    ),

    ("aider", re.compile(r"\baider(?:\.exe)?\b", re.I), 0.65, None),

    ("cline", re.compile(r"\bcline\b", re.I), 0.60, None),

    ("goose", re.compile(r"\bgoose\b", re.I), 0.60, None),

    (
        "gptscript",
        re.compile(r"\bgptscript\b", re.I),
        0.65,
        None,
    ),

    (
        "mcp-server",
        re.compile(r"\bmcp[-_ ]?server\b", re.I),
        0.80,
        "mcp",
    ),

    (
        "fastmcp",
        re.compile(r"\bfastmcp\b", re.I),
        0.75,
        "mcp",
    ),

    (
        "claude-code",
        re.compile(r"\bclaude(?:-code)?\b", re.I),
        0.75,
        "anthropic",
    ),
]


# ============================================================
# MCP JSON-RPC
# ============================================================

MCP_METHODS = {
    "initialize",
    "notifications/initialized",
    "tools/list",
    "tools/call",
    "resources/list",
    "resources/read",
    "prompts/list",
    "prompts/get",
}


def _parse_json_safely(value: Any) -> Optional[Any]:
    value = _safe_str(value, MAX_BODY_LENGTH).strip()

    if not value:
        return None

    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _walk_json_for_mcp(obj: Any) -> Optional[str]:
    """
    Detect MCP JSON-RPC even when body isn't a simple flat string.
    """

    if isinstance(obj, dict):

        if str(obj.get("jsonrpc", "")) == "2.0":

            method = obj.get("method")

            if method in MCP_METHODS:
                return str(method)

        for value in obj.values():
            result = _walk_json_for_mcp(value)

            if result:
                return result

    elif isinstance(obj, list):

        for value in obj:
            result = _walk_json_for_mcp(value)

            if result:
                return result

    return None


# ============================================================
# LOCAL AI PORTS
# ============================================================

# Important:
# Ports are NEVER high-confidence by themselves.

LOCAL_AI_PORTS = {
    11434: ("ollama", 0.65),
    1234: ("lm-studio", 0.55),
    8000: ("local-ai", 0.20),
    8080: ("local-ai", 0.15),
    8001: ("local-ai", 0.15),
}


# ============================================================
# CONTENT / API SEMANTICS
# ============================================================

# These are intentionally weak individually.
#
# They become useful when combined with:
#   - POST
#   - JSON
#   - model parameter
#   - token parameters
#   - streaming
#   - known AI endpoint
#
# This catches unknown/self-hosted OpenAI-compatible servers.

AI_JSON_KEYS = {
    "model",
    "messages",
    "system",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "tools",
    "tool_choice",
    "response_format",
    "embeddings",
}


def _body_json_keys(body: Any) -> set[str]:
    obj = _parse_json_safely(body)

    if not isinstance(obj, dict):
        return set()

    return {
        str(k).lower()
        for k in obj.keys()
    }


def _looks_like_llm_payload(row: Dict[str, Any]) -> Optional[Detection]:
    body = _first(row, "body", "payload", "request_body")

    if not body:
        return None

    keys = _body_json_keys(body)

    if not keys:
        return None

    overlap = keys & AI_JSON_KEYS

    # One generic "model" key isn't enough.
    if len(overlap) < 2:
        return None

    return Detection(
        name="llm-json-payload",
        weight=0.42,
        severity="MEDIUM",
        reason=f"json_keys:{','.join(sorted(overlap))}",
        category="llm-api",
        confidence=0.72,
    )


# ============================================================
# DETECTORS
# ============================================================

def detect_domain(row: Dict[str, Any]) -> Optional[Detection]:

    host = _host_of(row)

    if not host:
        return None

    provider = _domain_provider(host)

    if provider:

        return Detection(
            name="known-ai-domain",
            weight=0.88,
            severity="HIGH",
            reason=f"domain:{host}",
            category="cloud-api",
            provider=provider,
            confidence=0.98,
        )

    # Local endpoint.
    if host in LOCAL_AI_DOMAINS:

        path = _path_of(row)

        if path.lower().startswith(
            (
                "/api/chat",
                "/api/generate",
                "/api/tags",
                "/v1/chat/completions",
                "/v1/completions",
            )
        ):

            return Detection(
                name="local-ai-host",
                weight=0.55,
                severity="MEDIUM",
                reason=f"local-host:{host}",
                category="local-llm",
                confidence=0.90,
            )

    return None


def detect_sni(row: Dict[str, Any]) -> Optional[Detection]:

    sni = _extract_host(
        _first(row, "sni", "tls_sni", "server_name")
    )

    if not sni:
        return None

    provider = _domain_provider(sni)

    if provider:

        return Detection(
            name="known-ai-sni",
            weight=0.78,
            severity="HIGH",
            reason=f"sni:{sni}",
            category="cloud-api",
            provider=provider,
            confidence=0.96,
        )

    return None


def detect_dns(row: Dict[str, Any]) -> Optional[Detection]:

    query = _extract_host(
        _first(row, "dns_query", "query_name", "qname")
    )

    if not query:
        return None

    provider = _domain_provider(query)

    if provider:

        return Detection(
            name="known-ai-dns",
            weight=0.72,
            severity="HIGH",
            reason=f"dns:{query}",
            category="cloud-api",
            provider=provider,
            confidence=0.94,
        )

    return None


def detect_path(row: Dict[str, Any]) -> Optional[Detection]:

    path = _path_of(row)

    if not path:
        return None

    for name, pattern, weight, category in PATH_PATTERNS:

        if pattern.search(path):

            return Detection(
                name=name,
                weight=weight,
                severity="HIGH",
                reason=f"path:{path}",
                category=category,
                confidence=0.93,
            )

    return None


def detect_headers(row: Dict[str, Any]) -> List[Detection]:

    blob = _header_blob(row)

    if not blob:
        return []

    detections = []

    for name, pattern, weight, category, provider in HEADER_PATTERNS:

        match = pattern.search(blob)

        if match:

            detections.append(
                Detection(
                    name=name,
                    weight=weight,
                    severity="HIGH",
                    reason=f"header:{match.group(0)[:100]}",
                    category=category,
                    provider=provider,
                    confidence=0.92,
                )
            )

    return detections


def detect_user_agent(row: Dict[str, Any]) -> List[Detection]:

    ua = _safe_str(
        _first(row, "user_agent", "user-agent", "ua")
    )

    if not ua:
        return []

    detections = []

    for name, pattern, weight, provider in UA_PATTERNS:

        match = pattern.search(ua)

        if match:

            detections.append(
                Detection(
                    name=name,
                    weight=weight,
                    severity="MEDIUM",
                    reason=f"ua:{match.group(0)}",
                    category="sdk",
                    provider=provider,
                    confidence=0.82,
                )
            )

    return detections


def detect_process(row: Dict[str, Any]) -> List[Detection]:

    proc = _safe_str(
        _first(
            row,
            "process",
            "process_name",
            "image",
            "cmd",
            "command_line",
        )
    )

    if not proc:
        return []

    detections = []

    for name, pattern, weight, category in PROCESS_PATTERNS:

        match = pattern.search(proc)

        if match:

            detections.append(
                Detection(
                    name=name,
                    weight=weight,
                    severity="HIGH",
                    reason=f"process:{match.group(0)}",
                    category=category or "ai-tool",
                    provider=category if category in PROVIDER_DOMAINS else None,
                    confidence=0.90,
                )
            )

    return detections


def detect_mcp(row: Dict[str, Any]) -> Optional[Detection]:

    body = _first(
        row,
        "body",
        "payload",
        "request_body",
        "response_body",
    )

    if not body:
        return None

    parsed = _parse_json_safely(body)

    if parsed is not None:

        method = _walk_json_for_mcp(parsed)

        if method:

            return Detection(
                name="mcp-json-rpc",
                weight=0.90,
                severity="HIGH",
                reason=f"mcp:{method}",
                category="mcp",
                confidence=0.98,
            )

    # Fallback for logs where body isn't valid JSON.
    body_text = _safe_str(body, MAX_BODY_LENGTH)

    if re.search(
        r'"jsonrpc"\s*:\s*"2\.0"',
        body_text,
        re.I,
    ) and re.search(
        r'"method"\s*:\s*"(?:initialize|tools/call|tools/list|resources/read)"',
        body_text,
        re.I,
    ):

        return Detection(
            name="mcp-json-rpc",
            weight=0.88,
            severity="HIGH",
            reason="mcp:jsonrpc",
            category="mcp",
            confidence=0.95,
        )

    return None


def detect_port(row: Dict[str, Any]) -> Optional[Detection]:

    port = _port_of(row)

    if port is None:
        return None

    if port in LOCAL_AI_PORTS:

        provider, weight = LOCAL_AI_PORTS[port]

        return Detection(
            name="local-ai-port",
            weight=weight,
            severity="LOW" if weight < 0.5 else "MEDIUM",
            reason=f"port:{port}",
            category="local-llm",
            provider=provider,
            confidence=0.55,
        )

    return None


# ============================================================
# TLS FINGERPRINTS
# ============================================================

KNOWN_AI_JA3 = {
    # Populate only from controlled lab captures.
    #
    # "hash": "openai-python",
}

KNOWN_AI_JA4 = {
    # Populate from controlled captures.
    #
    # "hash": "some-ai-client",
}


def detect_tls(row: Dict[str, Any]) -> List[Detection]:

    results = []

    ja3 = _lower(
        _first(row, "ja3", "ja3_hash")
    )

    if ja3 and ja3 in KNOWN_AI_JA3:

        results.append(
            Detection(
                name="known-ja3",
                weight=0.62,
                severity="HIGH",
                reason=f"ja3:{KNOWN_AI_JA3[ja3]}",
                category="tls",
                confidence=0.80,
            )
        )

    ja4 = _lower(
        _first(row, "ja4", "ja4_hash")
    )

    if ja4 and ja4 in KNOWN_AI_JA4:

        results.append(
            Detection(
                name="known-ja4",
                weight=0.65,
                severity="HIGH",
                reason=f"ja4:{KNOWN_AI_JA4[ja4]}",
                category="tls",
                confidence=0.82,
            )
        )

    return results


# ============================================================
# HTTP METHOD / CONTENT TYPE
# ============================================================

def detect_http_context(row: Dict[str, Any]) -> List[Detection]:

    results = []

    method = _lower(
        _first(row, "method", "http_method")
    )

    headers = _headers(row)

    content_type = _lower(
        headers.get("content-type")
    )

    if method == "post":

        results.append(
            Detection(
                name="http-post",
                weight=0.08,
                severity="LOW",
                reason="http:POST",
                category="http-context",
                confidence=0.80,
            )
        )

    if "application/json" in content_type:

        results.append(
            Detection(
                name="json-content",
                weight=0.08,
                severity="LOW",
                reason="content-type:json",
                category="http-context",
                confidence=0.85,
            )
        )

    return results


# ============================================================
# ALL ROW DETECTION
# ============================================================

def detect_row(row: Dict[str, Any]) -> List[Detection]:

    detections: List[Detection] = []

    # Strong network identity.
    for detector in (
        detect_domain,
        detect_sni,
        detect_dns,
        detect_path,
        detect_mcp,
        detect_port,
    ):

        try:
            result = detector(row)

            if result:
                detections.append(result)

        except Exception:
            LOGGER.exception(
                "Detector failed: %s",
                detector.__name__,
            )

    # Multi-result detectors.
    for detector in (
        detect_headers,
        detect_user_agent,
        detect_process,
        detect_tls,
        detect_http_context,
    ):

        try:
            detections.extend(detector(row))

        except Exception:
            LOGGER.exception(
                "Detector failed: %s",
                detector.__name__,
            )

    semantic = _looks_like_llm_payload(row)

    if semantic:
        detections.append(semantic)

    return detections


# ============================================================
# CORRELATION
# ============================================================

def _unique_detections(
    detections: Sequence[Detection],
) -> List[Detection]:

    seen = set()
    result = []

    for d in detections:

        key = (
            d.name,
            d.provider,
            d.reason,
        )

        if key not in seen:

            seen.add(key)
            result.append(d)

    return result


def _correlation_bonus(
    detections: Sequence[Detection],
) -> float:

    names = {d.name for d in detections}
    categories = {d.category for d in detections}

    bonus = 0.0

    # Known domain + API endpoint.
    if (
        "known-ai-domain" in names
        and any(
            d.category in {"llm-api", "local-llm"}
            for d in detections
        )
    ):
        bonus += 0.12

    # SDK + known destination.
    if (
        "known-ai-domain" in names
        and "sdk" in categories
    ):
        bonus += 0.10

    # Local AI process + local API.
    if (
        "local-llm" in categories
        and "process" not in names
    ):
        bonus += 0.0

    # MCP protocol + MCP endpoint/tool.
    if "mcp-json-rpc" in names and "mcp" in categories:
        bonus += 0.15

    # LLM payload + LLM API path.
    if (
        "llm-json-payload" in names
        and any(d.category == "llm-api" for d in detections)
    ):
        bonus += 0.10

    return bonus


# ============================================================
# SCORE
# ============================================================

def score_detections(
    detections: Sequence[Detection],
) -> Tuple[float, float]:

    detections = _unique_detections(detections)

    if not detections:
        return 0.0, 0.0

    # Group by category to reduce double counting.
    category_scores: Dict[str, float] = {}

    for d in detections:

        contribution = (
            d.weight
            * d.confidence
        )

        category_scores[d.category] = max(
            category_scores.get(d.category, 0.0),
            contribution,
        )

    # Strongest evidence in each independent category.
    score = sum(category_scores.values())

    score += _correlation_bonus(detections)

    score = min(score, 1.0)

    # Confidence is not identical to score.
    #
    # Example:
    #   port 8080 + JSON body
    #
    # may have moderate score but low confidence.
    #
    # Domain + API + SDK should have high confidence.

    high_confidence = sum(
        1
        for d in detections
        if d.confidence >= 0.9
    )

    independent_categories = len(category_scores)

    confidence = (
        0.35
        + min(high_confidence, 3) * 0.15
        + min(independent_categories, 4) * 0.08
    )

    confidence = min(confidence, 0.98)

    return score, confidence


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(
    score: float,
    detections: Sequence[Detection],
) -> str:

    if any(
        d.severity == "HIGH"
        and d.confidence >= 0.9
        for d in detections
    ):
        if score >= 0.72:
            return "HIGH"

    if score >= 0.55:
        return "MEDIUM"

    if score >= 0.25:
        return "LOW"

    return "LOW"


# ============================================================
# ROW SCORING
# ============================================================

def score_row(
    row: Dict[str, Any],
) -> DetectionResult:

    detections = detect_row(row)

    score, confidence = score_detections(detections)

    severity = calculate_severity(
        score,
        detections,
    )

    providers = sorted({
        d.provider
        for d in detections
        if d.provider
    })

    categories = sorted({
        d.category
        for d in detections
    })

    hits = [
        d.reason
        for d in detections
    ]

    evidence = [
        {
            "signal": d.name,
            "reason": d.reason,
            "weight": round(d.weight, 3),
            "confidence": round(d.confidence, 3),
            "severity": d.severity,
            "category": d.category,
            "provider": d.provider,
        }
        for d in detections
    ]

    if score >= ANALYZE_THRESHOLD:
        verdict = "analyze"

    elif score >= REVIEW_THRESHOLD:
        verdict = "review"

    else:
        verdict = "audit"

    return DetectionResult(
        verdict=verdict,
        score=round(score, 4),
        confidence=round(confidence, 4),
        severity=severity,
        providers=providers,
        categories=categories,
        hits=hits,
        evidence=evidence,
    )


# ============================================================
# CHUNK / SESSION SCORING
# ============================================================

def score_chunk(
    rows: Sequence[Dict[str, Any]],
) -> DetectionResult:

    if not rows:
        return DetectionResult(
            verdict="audit",
            score=0.0,
            confidence=0.0,
            severity="LOW",
        )

    row_results = [
        score_row(row)
        for row in rows
    ]

    # Strongest row.
    best = max(
        row_results,
        key=lambda result: result.score,
    )

    # --------------------------------------------------------
    # Session-level correlation.
    #
    # This catches cases where no individual event is decisive:
    #
    # DNS -> CONNECT -> TLS -> POST -> API path
    # --------------------------------------------------------

    all_detections = []

    for result in row_results:

        for evidence in result.evidence:

            all_detections.append(
                Detection(
                    name=evidence["signal"],
                    weight=evidence["weight"],
                    severity=evidence["severity"],
                    reason=evidence["reason"],
                    category=evidence["category"],
                    provider=evidence["provider"],
                    confidence=evidence["confidence"],
                )
            )

    session_score, session_confidence = score_detections(
        all_detections
    )

    # Don't allow session aggregation to become absurdly high
    # simply because the same fingerprint appeared 100 times.
    final_score = max(
        best.score,
        min(session_score, 1.0),
    )

    final_confidence = max(
        best.confidence,
        session_confidence,
    )

    severity = calculate_severity(
        final_score,
        all_detections,
    )

    providers = sorted({
        d.provider
        for d in all_detections
        if d.provider
    })

    categories = sorted({
        d.category
        for d in all_detections
    })

    hits = list(dict.fromkeys(
        d.reason
        for d in all_detections
    ))

    evidence = [
        {
            "signal": d.name,
            "reason": d.reason,
            "weight": round(d.weight, 3),
            "confidence": round(d.confidence, 3),
            "severity": d.severity,
            "category": d.category,
            "provider": d.provider,
        }
        for d in _unique_detections(all_detections)
    ]

    if final_score >= ANALYZE_THRESHOLD:
        verdict = "analyze"

    elif final_score >= REVIEW_THRESHOLD:
        verdict = "review"

    else:
        verdict = "audit"

    return DetectionResult(
        verdict=verdict,
        score=round(final_score, 4),
        confidence=round(final_confidence, 4),
        severity=severity,
        providers=providers,
        categories=categories,
        hits=hits,
        evidence=evidence,
    )


# ============================================================
# JSON SERIALIZATION
# ============================================================

def result_to_dict(
    result: DetectionResult,
) -> Dict[str, Any]:

    return {
        "verdict": result.verdict,
        "score": result.score,
        "confidence": result.confidence,
        "severity": result.severity,
        "providers": result.providers,
        "categories": result.categories,
        "hits": result.hits,
        "evidence": result.evidence,
    }


# ============================================================
# CSV DOMAIN LOADER
# ============================================================

def load_extra_domains_csv(
    *paths: str,
) -> None:

    for path in paths:

        with open(
            path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:

            reader = csv.DictReader(handle)

            for row in reader:

                domain = _extract_host(
                    row.get("Domain")
                    or row.get("domain")
                    or ""
                )

                if not domain:
                    continue

                # Put additional domains into a generic provider.
                PROVIDER_DOMAINS.setdefault(
                    "custom",
                    set(),
                ).add(domain)


# ============================================================
# STREAMING CLI
# ============================================================

# def process_file(
#     input_path: str,
#     near_miss_path: str = "near_miss.jsonl",
# ) -> None:

#     with open(
#         input_path,
#         "r",
#         encoding="utf-8",
#     ) as source, open(
#         near_miss_path,
#         "a",
#         encoding="utf-8",
#     ) as near_miss:

#         for line_number, line in enumerate(
#             source,
#             start=1,
#         ):

#             line = line.strip()

#             if not line:
#                 continue

#             try:
#                 rows = json.loads(line)

#             except json.JSONDecodeError as exc:

#                 LOGGER.warning(
#                     "Invalid JSON on line %d: %s",
#                     line_number,
#                     exc,
#                 )

#                 continue

#             if not isinstance(rows, list):

#                 LOGGER.warning(
#                     "Expected list on line %d",
#                     line_number,
#                 )

#                 continue

#             rows = [
#                 row
#                 for row in rows
#                 if isinstance(row, dict)
#             ]

#             result = score_chunk(rows)

#             output = result_to_dict(result)

#             output["line"] = line_number

#             encoded = json.dumps(
#                 output,
#                 ensure_ascii=False,
#             )

#             if result.verdict == "audit":
#                 near_miss.write(encoded + "\n")
#             else:
#                 print(encoded)



# ============================================================
# MAIN
# ============================================================

# def main() -> int:

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    if len(sys.argv) < 2:

        print(
            "usage: python ai_log_detector.py "
            "chunks.jsonl [domains.csv ...]",
            file=sys.stderr,
        )

        return 2

    input_path = sys.argv[1]

    if len(sys.argv) > 2:

        try:
            load_extra_domains_csv(
                *sys.argv[2:]
            )

        except OSError as exc:

            print(
                f"Failed loading domains: {exc}",
                file=sys.stderr,
            )

            return 2

    process_file(input_path)

    return 0

# ============================================================
# CSV INPUT SUPPORT
# ============================================================

def _normalize_csv_value(value: Any) -> Any:
    """
    Convert CSV strings into useful Python values.

    Examples:
        "443"        -> 443
        "true"       -> True
        "false"      -> False
        '{"a": 1}'   -> {"a": 1}
        "hello"      -> "hello"
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    # Integer
    try:
        if re.fullmatch(r"-?\d+", value):
            return int(value)
    except Exception:
        pass

    # Float
    try:
        if re.fullmatch(r"-?\d+\.\d+", value):
            return float(value)
    except Exception:
        pass

    # Boolean
    if value.lower() == "true":
        return True

    if value.lower() == "false":
        return False

    # JSON object / array
    if value.startswith("{") or value.startswith("["):

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

    return value


def normalize_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize one CSV row into the format expected by the detector.

    Handles common CSV column names such as:

        src_ip
        source_ip
        dst_ip
        destination_ip
        dst_port
        domain
        host
        url
        path
        url_path
        method
        headers
        user_agent
        process
        command_line
        body
        ja3
        ja4
        dns_query
        sni
    """

    normalized = {}

    for key, value in row.items():

        if key is None:
            continue

        clean_key = str(key).strip().lower()

        # Normalize spaces / hyphens.
        clean_key = clean_key.replace(" ", "_")
        clean_key = clean_key.replace("-", "_")

        normalized[clean_key] = _normalize_csv_value(value)

    # --------------------------------------------------------
    # Common column aliases
    # --------------------------------------------------------

    aliases = {
        "source_ip": "src_ip",
        "src": "src_ip",

        "destination_ip": "dst_ip",
        "dest_ip": "dst_ip",
        "destination": "dst_ip",
        "dest": "dst_ip",

        "destination_port": "dst_port",
        "dest_port": "dst_port",

        "hostname": "host",

        "uri": "url",
        "request_url": "url",

        "request_path": "url_path",
        "uri_path": "url_path",

        "http_method": "method",

        "ua": "user_agent",

        "process_name": "process",
        "processname": "process",

        "command": "cmd",
        "commandline": "cmd",
        "command_line": "cmd",

        "request_body": "body",
        "payload": "body",

        "query_name": "dns_query",
        "qname": "dns_query",

        "tls_sni": "sni",
        "server_name": "sni",

        "ja3_hash": "ja3",
        "ja4_hash": "ja4",
    }

    for old_key, new_key in aliases.items():

        if (
            old_key in normalized
            and new_key not in normalized
        ):
            normalized[new_key] = normalized[old_key]

    return normalized


def load_csv_rows(
    input_path: str,
) -> Iterable[Dict[str, Any]]:
    """
    Stream rows from a CSV file.

    Each CSV row becomes one normalized detector row.
    """

    with open(
        input_path,
        "r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError(
                "CSV file does not contain a header row."
            )

        LOGGER.info(
            "CSV columns: %s",
            ", ".join(reader.fieldnames),
        )

        for row in reader:

            if not row:
                continue

            try:
                yield normalize_csv_row(row)

            except Exception:
                LOGGER.exception(
                    "Failed normalizing CSV row"
                )


# ============================================================
# CSV PROCESSOR
# ============================================================

def process_csv_file(
    input_path: str,
    output_path: str = "ai_detections.jsonl",
    near_miss_path: str = "near_miss.jsonl",
) -> None:
    """
    Analyze a CSV file row-by-row.

    Every CSV row is treated as one network / EDR / proxy event.
    """

    total = 0
    analyzed = 0
    reviewed = 0
    audited = 0
    errors = 0

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output, open(
        near_miss_path,
        "w",
        encoding="utf-8",
    ) as near_miss:

        for row_number, row in enumerate(
            load_csv_rows(input_path),
            start=1,
        ):

            total += 1

            try:

                result = score_row(row)

                result_dict = result_to_dict(result)

                result_dict["row"] = row_number

                encoded = json.dumps(
                    result_dict,
                    ensure_ascii=False,
                )

                if result.verdict == "analyze":

                    analyzed += 1

                    output.write(
                        encoded + "\n"
                    )

                    print(
                        f"[ANALYZE] row={row_number} "
                        f"score={result.score:.2f} "
                        f"confidence={result.confidence:.2f} "
                        f"providers={','.join(result.providers)}"
                    )

                elif result.verdict == "review":

                    reviewed += 1

                    output.write(
                        encoded + "\n"
                    )

                    print(
                        f"[REVIEW]  row={row_number} "
                        f"score={result.score:.2f} "
                        f"confidence={result.confidence:.2f} "
                        f"providers={','.join(result.providers)}"
                    )

                else:

                    audited += 1

                    near_miss.write(
                        encoded + "\n"
                    )

            except Exception as exc:

                errors += 1

                LOGGER.exception(
                    "Failed processing CSV row %d: %s",
                    row_number,
                    exc,
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("AI LOG DETECTION SUMMARY")
    print("=" * 60)

    print(f"Total rows      : {total}")
    print(f"Analyze         : {analyzed}")
    print(f"Review          : {reviewed}")
    print(f"Audit           : {audited}")
    print(f"Errors          : {errors}")

    print()
    print(f"Detection file  : {output_path}")
    print(f"Near-miss file  : {near_miss_path}")
    print("=" * 60)


# ============================================================
# AUTO INPUT PROCESSOR
# ============================================================

def process_input_file(
    input_path: str,
    output_path: str = "ai_detections.jsonl",
    near_miss_path: str = "near_miss.jsonl",
) -> None:
    """
    Automatically determine whether the input is CSV or JSONL.
    """

    extension = input_path.lower()

    if extension.endswith(".csv"):

        process_csv_file(
            input_path=input_path,
            output_path=output_path,
            near_miss_path=near_miss_path,
        )

        return

    if extension.endswith(
        (".jsonl", ".ndjson", ".json")
    ):

        process_file(
            input_path=input_path,
            near_miss_path=near_miss_path,
        )

        return

    raise ValueError(
        "Unsupported input format. "
        "Use .csv, .jsonl, .ndjson, or .json"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    if len(sys.argv) < 2:

        print(
            "Usage:",
            file=sys.stderr,
        )

        print(
            "  python ai_log_detector.py logs.csv",
            file=sys.stderr,
        )

        print(
            "  python ai_log_detector.py chunks.jsonl",
            file=sys.stderr,
        )

        print(
            "  python ai_log_detector.py logs.csv domains.csv",
            file=sys.stderr,
        )

        return 2

    input_path = sys.argv[1]

    # --------------------------------------------------------
    # Load optional additional domains.
    # --------------------------------------------------------

    if len(sys.argv) > 2:

        try:

            load_extra_domains_csv(
                *sys.argv[2:]
            )

        except OSError as exc:

            print(
                f"Failed loading domains: {exc}",
                file=sys.stderr,
            )

            return 2

    # --------------------------------------------------------
    # Process input.
    # --------------------------------------------------------

    try:

        process_input_file(
            input_path
        )

    except FileNotFoundError:

        print(
            f"File not found: {input_path}",
            file=sys.stderr,
        )

        return 2

    except ValueError as exc:

        print(
            f"Input error: {exc}",
            file=sys.stderr,
        )

        return 2

    except Exception as exc:

        LOGGER.exception(
            "Fatal processing error"
        )

        print(
            f"Failed processing file: {exc}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# if __name__ == "__main__":
#     raise SystemExit(main())