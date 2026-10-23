#!/usr/bin/env python3
"""Agent proxy on port 8090 — type: devops — Primary: Polza AI, Fallback: Ollama."""
import asyncio, json, sqlite3, sys, time as _time
from datetime import datetime, timezone
from pathlib import Path
import httpx
from fastapi import FastAPI, Request, Response
import uvicorn

# ── Upstreams ──────────────────────────────────────────────────────
UPSTREAM_PRIMARY = 'http://127.0.0.1:8084'
UPSTREAM_FALLBACK = "http://127.0.0.1:11434/v1"  # Local Ollama
API_KEY_PRIMARY = None  # Polza proxy doesn't need auth, it has its own key
API_KEY_FALLBACK = "ollama"
FALLBACK_MODEL = "qwen3-hermes"
UPSTREAM_TIMEOUT = 180

# ── Rest unchanged (DB, logging, disconnect handling) ─────────────
DB_PATH = Path.home() / "token-stats" / "data" / "tokens.db"
AGENT_TYPE = "backup"
DB_TIMEOUT = 5.0
STALE_SECONDS = 300
AGENT_MAX_TOKENS = 2048

app = FastAPI()
_client = None

def get_client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    return _client

async def replace_client():
    global _client
    old = _client
    _client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    if old is not None:
        try: await old.aclose()
        except Exception: pass

_active_tasks: dict[int, asyncio.Task] = {}
_next_id = 0
_task_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_db():
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute("PRAGMA journal_mode=WAL")

def _mark_stale():
    try:
        with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
            rows = conn.execute("SELECT id, ts FROM token_log WHERE agent_name=? AND status='running'", (AGENT_TYPE,)).fetchall()
            for row in rows:
                try:
                    ts = row["ts"].replace("Z", "+00:00")
                    then = datetime.fromisoformat(ts)
                    now = datetime.now(timezone.utc)
                    if (now - then).total_seconds() > STALE_SECONDS:
                        diff = int((now - then).total_seconds())
                        conn.execute("UPDATE token_log SET status='timeout', error_msg=? WHERE id=?", (f"Stale after {diff}s", row["id"]))
                except Exception: pass
            conn.commit()
    except Exception: pass

def _phase1(preview):
    ts = datetime.now(timezone.utc).isoformat()
    _mark_stale()
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        cur = conn.execute("INSERT INTO token_log (ts, agent_name, provider_type, model, prompt_tokens, completion_tokens, cost, status, task_preview) VALUES (?,?,?,?,0,0,0,'running',?)", (ts, AGENT_TYPE, 'polza_api', '', preview))
        conn.commit()
        return cur.lastrowid or 0

def _phase2_ok(row_id, model, pt, ct, dur, prov_type="polza_api", is_fb=0, status_code=200):
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        tot = pt + ct
        conn.execute("UPDATE token_log SET status='completed', provider_type=?, is_fallback=?, model=?, prompt_tokens=?, completion_tokens=?, total_tokens=?, cost=0, duration_ms=?, status_code=? WHERE id=? AND status='running'", (prov_type, is_fb, model, pt, ct, tot, dur, status_code, row_id))
        conn.commit()

def _phase2_err(row_id, err, dur=0, status_code=500):
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute("UPDATE token_log SET status='error', error_msg=?, duration_ms=?, status_code=? WHERE id=? AND status='running'", (err[:500], dur, status_code, row_id))
        conn.commit()

def _phase2_tool(row_id, dur=0):
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute("UPDATE token_log SET status='tool_call', duration_ms=?, status_code=200 WHERE id=? AND status='running'", (dur, row_id))
        conn.commit()

def _phase2_cancelled(row_id, dur):
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute("UPDATE token_log SET status='cancelled', error_msg=?, duration_ms=?, status_code=499 WHERE id=? AND status='running'", ("cancelled by client", dur, row_id))
        conn.commit()

async def _monitor_disconnect(request: Request, disconnect: asyncio.Event):
    while not disconnect.is_set():
        if await request.is_disconnected(): disconnect.set(); return
        await asyncio.sleep(0.3)

async def _silence_task(t: asyncio.Task):
    try: await t
    except Exception: pass

async def _upstream_call(method, url, body, headers, client):
    req = await client.request(method=method, url=url, content=body, headers=headers)
    return req

async def _try_upstream_with_fallback(method, url_path, body, headers, disconnect):
    """Try primary (Polza), fallback to Ollama on any error."""
    client = get_client()
    primary_url = f"{UPSTREAM_PRIMARY}/{url_path}"
    if '?' in str(body):
        pass  # query params handled by caller

    # Try primary
    used_fallback = False
    for attempt, (upstream, api_key, override_model) in enumerate([
        (UPSTREAM_PRIMARY, API_KEY_PRIMARY, None),
        (UPSTREAM_FALLBACK, API_KEY_FALLBACK, FALLBACK_MODEL),
    ]):
        if disconnect.is_set():
            raise asyncio.CancelledError("client disconnected")
        
        url = f"{upstream}/{url_path}"
        req_headers = dict(headers)
        req_headers.pop("host", None)
        req_headers["X-Subagent"] = "8092"
        if api_key:
            req_headers["Authorization"] = f"Bearer {api_key}"

        req_body = body
        if override_model and b"chat/completions" in url.encode() and body:
            try:
                req_json = json.loads(body)
                req_json["model"] = override_model
                req_body = json.dumps(req_json).encode()
                req_headers["Content-Length"] = str(len(req_body))
            except Exception: pass

        try:
            resp = await client.request(method=method, url=url, content=req_body, headers=req_headers)
            if resp.status_code >= 500 or resp.status_code in (429, 402, 401):
                if attempt == 0:
                    used_fallback = True
                    print(f"[{AGENT_TYPE}] Polza returned {resp.status_code}, falling back to Ollama", flush=True)
                    continue
            return resp, used_fallback
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError):
            if attempt == 0:
                used_fallback = True
                print(f"[{AGENT_TYPE}] Polza connection failed, falling back to Ollama", flush=True)
                continue
            raise
    # Both failed
    raise httpx.ConnectError("Both Polza and Ollama upstreams failed")

@app.post("/cancel")
async def cancel_active():
    killed = 0
    async with _task_lock:
        for tid in list(_active_tasks.keys()):
            t = _active_tasks.get(tid)
            if t is not None and not t.done(): t.cancel(); killed += 1
    return {"cancelled": killed}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(request: Request, path: str):
    global _next_id
    body = await request.body()
    headers = dict(request.headers)
    url_path = path
    if request.url.query: url_path += f"?{request.url.query}"
    print(f"[{AGENT_TYPE}] {request.method} /{url_path}", flush=True)

    row_id = None
    preview = ""
    is_chat = path.endswith("/chat/completions") and request.method == "POST"
    t_start = datetime.now(timezone.utc)

    if is_chat and body:
        try:
            req_json = json.loads(body)
            for key in ("max_tokens", "max_completion_tokens"):
                val = req_json.get(key, AGENT_MAX_TOKENS)
                if val is None or val > AGENT_MAX_TOKENS: req_json[key] = AGENT_MAX_TOKENS
            body = json.dumps(req_json).encode()
            headers.pop("Content-Length", None)
            headers.pop("content-length", None)
            headers["Content-Length"] = str(len(body))
            for msg in req_json.get("messages", []):
                c = msg.get("content", "")
                if isinstance(c, str) and c.strip(): preview = c[:120]; break
        except Exception: pass
        row_id = _phase1(preview)

    disconnect = asyncio.Event()
    async with _task_lock: _next_id += 1; task_id = _next_id

    monitor = asyncio.create_task(_monitor_disconnect(request, disconnect))
    upstream_task = asyncio.create_task(_try_upstream_with_fallback(request.method, url_path, body, headers, disconnect))

    async with _task_lock: _active_tasks[task_id] = upstream_task

    try:
        finished, _ = await asyncio.wait([upstream_task, monitor], return_when=asyncio.FIRST_COMPLETED)
        for ft in finished:
            if ft is upstream_task: monitor.cancel()
            else: upstream_task.cancel(); asyncio.create_task(_silence_task(upstream_task))

        elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)

        if upstream_task in finished and not upstream_task.cancelled():
            try: resp, used_fallback = upstream_task.result()
            except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError):
                if row_id: _phase2_cancelled(row_id, elapsed)
                await replace_client()
                return Response(content=json.dumps({"error": "cancelled", "detail": "upstream disconnected"}).encode(), status_code=499, media_type="application/json", headers={"Connection": "close"})
        else:
            if row_id: _phase2_cancelled(row_id, elapsed)
            await replace_client()
            return Response(content=json.dumps({"error": "cancelled", "detail": "client disconnected"}).encode(), status_code=499, media_type="application/json", headers={"Connection": "close"})

        if row_id and resp.status_code >= 400: _phase2_err(row_id, f"HTTP {resp.status_code}", elapsed)
        elif row_id:
            try:
                data = resp.json()
                u = data.get("usage") or {}
                pt = u.get("prompt_tokens", 0); ct = u.get("completion_tokens", 0)
                model = data.get("model", "unknown")
                if model in ("unknown", "", "-") and pt == 0 and ct == 0: _phase2_tool(row_id, elapsed)
                else: _phase2_ok(row_id, model, pt, ct, elapsed, "local_moe" if used_fallback else "polza_api", 1 if used_fallback else 0)
            except Exception: _phase2_tool(row_id, elapsed)

        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))

    except asyncio.CancelledError:
        if row_id:
            elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
            _phase2_cancelled(row_id, elapsed)
        await replace_client()
        return Response(content=json.dumps({"error": "cancelled"}).encode(), status_code=499, media_type="application/json", headers={"Connection": "close"})
    finally:
        async with _task_lock: _active_tasks.pop(task_id, None)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8092, log_level="warning")
