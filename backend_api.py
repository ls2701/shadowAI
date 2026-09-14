"""
Shadow AI Report API + Live Dashboard
=====================================
Serves findings.jsonl, needs_review.jsonl, dynamic_domain_cache.json and
streams pipeline_events.jsonl over SSE.

Run with:
    uvicorn backend_api:app --host 0.0.0.0 --port 8000 --reload

Dashboard:  http://localhost:8000/
Docs:       http://localhost:8000/docs

Required packages:
    fastapi
    uvicorn[standard]
    python-multipart      ← needed for /upload (File/Form)
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

# ── FastAPI imports (single, unified line) ──────────────────────
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Startup sanity check ────────────────────────────────────────
try:
    import multipart  # noqa: F401
except ImportError:
    print(
        "WARNING: python-multipart is not installed.\n"
        "         /upload will fail with an empty response.\n"
        "         Fix:  pip install python-multipart\n"
    )

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
FINDINGS_FILE = BASE_DIR / "findings.jsonl"
REVIEW_FILE = BASE_DIR / "needs_review.jsonl"
CACHE_FILE = BASE_DIR / "dynamic_domain_cache.json"
EVENTS_FILE = BASE_DIR / "pipeline_events.jsonl"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ANALYSER_SCRIPT = BASE_DIR / "rag_analyser3.py"

# Global handle so we can reject a second upload while one is running.
_run_lock = threading.Lock()
_current_run: dict | None = None

_jsonl_cache: dict[str, tuple[float, list[dict]]] = {}

# ------------------------------------------------------------
# App + middleware
# ------------------------------------------------------------
app = FastAPI(
    title="Shadow AI Report API",
    version="1.2.0",
    description="Live dashboard + report API for the Shadow AI analyzer.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------------------------------------
# Global exception handler — always returns JSON
# ------------------------------------------------------------
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    import traceback
    tb = traceback.format_exc()
    print(f"[unhandled] {request.method} {request.url.path}: {exc}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    key = str(path.resolve())
    cached = _jsonl_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    _jsonl_cache[key] = (mtime, records)
    return records


def _load_cache_file() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        with CACHE_FILE.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return raw if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _finding_id(rec: dict, idx: int) -> str:
    return str(rec.get("_chunk_id", idx))


def _emit_event(kind: str, **fields) -> None:
    """Append a line to pipeline_events.jsonl."""
    try:
        with EVENTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"ts": time.time(), "kind": kind, **fields},
                ensure_ascii=False,
            ) + "\n")
    except OSError:
        pass


# ------------------------------------------------------------
# Health
# ------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "files": {
            "findings": {"path": str(FINDINGS_FILE), "exists": FINDINGS_FILE.exists()},
            "review":   {"path": str(REVIEW_FILE),   "exists": REVIEW_FILE.exists()},
            "cache":    {"path": str(CACHE_FILE),    "exists": CACHE_FILE.exists()},
            "events":   {"path": str(EVENTS_FILE),   "exists": EVENTS_FILE.exists()},
            "analyser": {"path": str(ANALYSER_SCRIPT), "exists": ANALYSER_SCRIPT.exists()},
        },
        "server_time": time.time(),
    }


# ------------------------------------------------------------
# Findings
# ------------------------------------------------------------
@app.get("/findings")
def list_findings(
    severity: Optional[str] = Query(None),
    classification: Optional[str] = Query(None),
    application: Optional[str] = Query(None),
    user: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    match_tier: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    records = _load_jsonl(FINDINGS_FILE)
    filtered = []
    for idx, r in enumerate(records):
        if severity and (r.get("severity") or "").lower() != severity.lower():
            continue
        if classification and (r.get("classification") or "").lower() != classification.lower():
            continue
        if application and (r.get("application") or "").lower() != application.lower():
            continue
        if match_tier and (r.get("_match_tier") or "").lower() != match_tier.lower():
            continue
        if user and user.lower() not in (r.get("_user") or "").lower():
            continue
        if min_confidence is not None:
            c = r.get("confidence")
            if not isinstance(c, (int, float)) or c < min_confidence:
                continue
        out = dict(r)
        out["_finding_id"] = _finding_id(r, idx)
        filtered.append(out)

    total = len(filtered)
    page = filtered[offset: offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "count": len(page), "items": page}


@app.get("/findings/{chunk_id}")
def get_finding(chunk_id: str) -> dict:
    records = _load_jsonl(FINDINGS_FILE)
    for idx, r in enumerate(records):
        if _finding_id(r, idx) == str(chunk_id):
            out = dict(r)
            out["_finding_id"] = _finding_id(r, idx)
            return out
    raise HTTPException(status_code=404, detail=f"Finding chunk_id={chunk_id} not found")


@app.get("/findings/{chunk_id}/evidence")
def get_finding_evidence(chunk_id: str) -> dict:
    records = _load_jsonl(FINDINGS_FILE)
    for idx, r in enumerate(records):
        if _finding_id(r, idx) == str(chunk_id):
            ev = r.get("_evidence")
            if not ev:
                raise HTTPException(status_code=404, detail="No structured evidence for this finding")
            return ev
    raise HTTPException(status_code=404, detail=f"Finding chunk_id={chunk_id} not found")


# ------------------------------------------------------------
# Review queue
# ------------------------------------------------------------
@app.get("/review")
def list_review(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    records = _load_jsonl(REVIEW_FILE)
    total = len(records)
    page = records[offset: offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "count": len(page), "items": page}


# ------------------------------------------------------------
# Aggregations
# ------------------------------------------------------------
@app.get("/stats")
def stats() -> dict:
    findings = _load_jsonl(FINDINGS_FILE)
    review = _load_jsonl(REVIEW_FILE)

    severity_counts: Counter = Counter()
    classification_counts: Counter = Counter()
    app_counts: Counter = Counter()
    tier_counts: Counter = Counter()
    user_counts: Counter = Counter()
    total_bytes = 0
    total_events = 0

    for r in findings:
        severity_counts[r.get("severity") or "Unknown"] += 1
        classification_counts[r.get("classification") or "Unknown"] += 1
        if r.get("application"):
            app_counts[r["application"]] += 1
        tier_counts[r.get("_match_tier") or "unknown"] += 1
        if r.get("_user"):
            user_counts[r["_user"]] += 1
        if isinstance(r.get("_bytes"), int):
            total_bytes += r["_bytes"]
        if isinstance(r.get("_event_count"), int):
            total_events += r["_event_count"]

    return {
        "findings_total": len(findings),
        "review_total": len(review),
        "severity_counts": dict(severity_counts),
        "classification_counts": dict(classification_counts),
        "application_counts": dict(app_counts.most_common(20)),
        "match_tier_counts": dict(tier_counts),
        "top_users": dict(user_counts.most_common(20)),
        "total_bytes_observed": total_bytes,
        "total_events_observed": total_events,
        "generated_at": time.time(),
    }


@app.get("/applications")
def list_applications() -> dict:
    findings = _load_jsonl(FINDINGS_FILE)
    per_app: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "count": 0, "severities": Counter(), "users": set(),
        "total_bytes": 0, "approved": None,
    })
    for r in findings:
        app = r.get("application")
        if not app:
            continue
        b = per_app[app]
        b["count"] += 1
        b["severities"][r.get("severity") or "Unknown"] += 1
        if r.get("_user"):
            b["users"].add(r["_user"])
        if isinstance(r.get("_bytes"), int):
            b["total_bytes"] += r["_bytes"]
        approved_map = r.get("_approved") or {}
        if isinstance(approved_map, dict) and app in approved_map:
            b["approved"] = bool(approved_map[app])

    out = []
    for app, b in per_app.items():
        out.append({
            "application": app,
            "count": b["count"],
            "severities": dict(b["severities"]),
            "users": sorted(b["users"]),
            "total_bytes": b["total_bytes"],
            "approved": b["approved"],
        })
    out.sort(key=lambda x: -x["count"])
    return {"total": len(out), "items": out}


@app.get("/users")
def list_users() -> dict:
    findings = _load_jsonl(FINDINGS_FILE)
    per_user: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "count": 0, "severities": Counter(), "applications": Counter(), "total_bytes": 0,
    })
    for r in findings:
        u = r.get("_user")
        if not u:
            continue
        b = per_user[u]
        b["count"] += 1
        b["severities"][r.get("severity") or "Unknown"] += 1
        if r.get("application"):
            b["applications"][r["application"]] += 1
        if isinstance(r.get("_bytes"), int):
            b["total_bytes"] += r["_bytes"]

    out = [
        {
            "user": u,
            "count": b["count"],
            "severities": dict(b["severities"]),
            "applications": dict(b["applications"].most_common(10)),
            "total_bytes": b["total_bytes"],
        }
        for u, b in per_user.items()
    ]
    out.sort(key=lambda x: -x["count"])
    return {"total": len(out), "items": out}


@app.get("/domains")
def list_domains(
    status: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> dict:
    cache = _load_cache_file()
    rows = []
    for domain, v in cache.items():
        if not isinstance(v, dict):
            continue
        if status and v.get("status") != status:
            continue
        rows.append({
            "domain": domain,
            "status": v.get("status"),
            "app": v.get("app"),
            "approved": bool(v.get("approved", False)),
            "confidence": float(v.get("confidence", 0.0) or 0.0),
            "last_attempt": v.get("last_attempt", 0),
        })
    rows.sort(key=lambda x: (-x["confidence"], x["domain"]))
    total = len(rows)
    return {
        "total": total, "limit": limit, "offset": offset,
        "count": len(rows[offset: offset + limit]),
        "items": rows[offset: offset + limit],
    }


# ------------------------------------------------------------
# Upload & Run
# ------------------------------------------------------------
@app.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    classify_model: str = Form("qwen2.5:14b"),
    reset_cache: bool = Form(False),
    log_year: Optional[int] = Form(None),
):
    """
    Save the uploaded CSV, launch the analyser as a subprocess, return
    immediately. Progress streams to the browser over /stream.
    """
    global _current_run

    # ---- 1) Guard against concurrent runs ----
    with _run_lock:
        if _current_run and _current_run["proc"].poll() is None:
            raise HTTPException(
                status_code=409,
                detail=f"a run is already in progress (csv={_current_run['csv_name']})",
            )

    # ---- 2) Sanity checks ----
    if not ANALYSER_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"analyser script not found at {ANALYSER_SCRIPT}",
        )
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="please upload a .csv file")

    # ---- 3) Save file ----
    safe_name = Path(file.filename).name
    dest = UPLOAD_DIR / safe_name
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not save upload: {exc}")

    if dest.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    # ---- 4) Build command ----
    cmd = [
        sys.executable, str(ANALYSER_SCRIPT), str(dest),
        "--classify-model", classify_model,
    ]
    if reset_cache:
        cmd.append("--reset-cache")
    if log_year is not None:
        cmd += ["--log-year", str(log_year)]

    # ---- 5) Spawn subprocess ----
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not start analyser: {exc}")

    # ---- 6) Pump stdout into the events file ----
    def _pump():
        try:
            for line in proc.stdout:
                _emit_event("stdout", line=line.rstrip())
        except Exception:
            pass

    threading.Thread(target=_pump, daemon=True).start()

    with _run_lock:
        _current_run = {
            "proc": proc,
            "csv_name": safe_name,
            "csv_path": str(dest),
            "started_at": time.time(),
            "cmd": cmd,
        }

    return {
        "status": "started",
        "csv": safe_name,
        "csv_path": str(dest),
        "command": " ".join(cmd),
        "pid": proc.pid,
    }


@app.get("/run-status")
def run_status() -> dict:
    """Whether an analysis subprocess is currently running."""
    with _run_lock:
        if not _current_run:
            return {"running": False}
        proc = _current_run["proc"]
        return {
            "running": proc.poll() is None,
            "pid": proc.pid,
            "csv": _current_run["csv_name"],
            "started_at": _current_run["started_at"],
            "exit_code": proc.poll(),
        }


# ------------------------------------------------------------
# SSE: live event stream
# ------------------------------------------------------------
@app.get("/stream")
async def stream_events():
    async def gen():
        yield ": connected\n\n"
        offset = 0
        last_heartbeat = 0.0
        while True:
            try:
                if EVENTS_FILE.exists():
                    size = EVENTS_FILE.stat().st_size
                    if size < offset:
                        offset = 0
                    if size > offset:
                        with EVENTS_FILE.open("r", encoding="utf-8") as f:
                            f.seek(offset)
                            for line in f:
                                line = line.strip()
                                if line:
                                    yield f"data: {line}\n\n"
                            offset = f.tell()
            except OSError:
                pass

            now = asyncio.get_event_loop().time()
            if now - last_heartbeat > 15:
                yield ": ping\n\n"
                last_heartbeat = now

            await asyncio.sleep(0.4)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ------------------------------------------------------------
# Dashboard root
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h1>Dashboard missing</h1><p>Create <code>static/index.html</code>.</p>",
            status_code=404,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))