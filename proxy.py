#!/usr/bin/env python3
"""Logging proxy for Strata — client-disconnect cancellation + max_tokens cap."""
import asyncio, json, sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
import uvicorn

UPSTREAM = "http://127.0.0.1:8083"
API_KEY = "strata-local"

DB_DIR = Path.home() / "token-stats" / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "tokens.db"

DB_TIMEOUT = 5.0
STALE_SECONDS = 300
CODER_TIMEOUT = 120          # был 300
CODER_MAX_TOKENS = 8192      # жёсткий потолок
COMPRESSOR_TIMEOUT = 180     # долгий таймаут для сжатия
COMPRESSION_PROMPT_MARKER = "You are a summarization agent creating a context checkpoint"
COMPRESSOR_MAX_CHARS = 20000 # обрезать вход до ~20k символов

app = FastAPI()

# ── client management: one per proxy, re-created after a force-close ────
_client = None

def get_client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=CODER_TIMEOUT)
    return _client

async def replace_client():
    global _client
    old = _client
    _client = httpx.AsyncClient(timeout=CODER_TIMEOUT)
    if old is not None:
        try:
            await old.aclose()
        except Exception:
            pass

# ── compressor client (long timeout) ────────────────────────────────────
_compressor_client = None

def get_compressor_client():
    global _compressor_client
    if _compressor_client is None:
        _compressor_client = httpx.AsyncClient(timeout=COMPRESSOR_TIMEOUT)
    return _compressor_client

async def replace_compressor_client():
    global _compressor_client
    old = _compressor_client
    _compressor_client = httpx.AsyncClient(timeout=COMPRESSOR_TIMEOUT)
    if old is not None:
        try:
            await old.aclose()
        except Exception:
            pass

# ── active-task tracking for /cancel ────────────────────────────────────
_active_tasks: dict[int, asyncio.Task] = {}
_next_id = 0
_task_lock = asyncio.Lock()
# ────────────────────────────────────────────────────────────────────────


def init_db():
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS token_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            type TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0,
            endpoint_target TEXT DEFAULT '',
            status TEXT DEFAULT 'completed',
            task_preview TEXT DEFAULT '',
            duration_ms INTEGER DEFAULT 0,
            error_msg TEXT DEFAULT ''
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON token_log(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_type_ts ON token_log(type, ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON token_log(status)")
        conn.commit()


def _mark_stale_as_timeout():
    try:
        with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
            rows = conn.execute(
                "SELECT id, ts FROM token_log WHERE status='running'"
            ).fetchall()
            for row in rows:
                try:
                    ts = row["ts"].replace("Z", "+00:00")
                    then = datetime.fromisoformat(ts)
                    now = datetime.now(timezone.utc)
                    diff = (now - then).total_seconds()
                    if diff > STALE_SECONDS:
                        conn.execute(
                            "UPDATE token_log SET status='timeout', error_msg=? WHERE id=?",
                            (f"Stale after {int(diff)}s", row["id"]),
                        )
                except Exception:
                    pass
            conn.commit()
    except Exception:
        pass


def _phase1_insert(agent_type: str, task_preview: str) -> int:
    ts = datetime.utcnow().isoformat() + "Z"
    _mark_stale_as_timeout()
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        cur = conn.execute(
            "INSERT INTO token_log (ts, type, model, prompt_tokens, completion_tokens, cost, status, task_preview) VALUES (?, ?, ?, 0, 0, 0, 'running', ?)",
            (ts, agent_type, "", task_preview),
        )
        conn.commit()
        return cur.lastrowid or 0


def _phase2_complete(row_id: int, model: str, pt: int, ct: int, duration_ms: int):
    cost = 0.0
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute(
            "UPDATE token_log SET status='completed', model=?, prompt_tokens=?, completion_tokens=?, cost=?, duration_ms=? WHERE id=? AND status='running'",
            (model, pt, ct, cost, duration_ms, row_id),
        )
        conn.commit()


def _phase2_error(row_id: int, error_msg: str, duration_ms: int = 0):
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute(
            "UPDATE token_log SET status='error', error_msg=?, duration_ms=? WHERE id=? AND status='running'",
            (error_msg[:500], duration_ms, row_id),
        )
        conn.commit()


def _phase2_cancelled(row_id: int, duration_ms: int):
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute(
            "UPDATE token_log SET status='cancelled', error_msg=?, duration_ms=? WHERE id=? AND status='running'",
            ("cancelled by client", duration_ms, row_id),
        )
        conn.commit()


# ── disconnect watcher ──────────────────────────────────────────────────
async def _monitor_disconnect(request: Request, disconnect: asyncio.Event):
    """Poll request.is_disconnected() every 0.3s; set Event when True."""
    while not disconnect.is_set():
        if await request.is_disconnected():
            disconnect.set()
            return
        await asyncio.sleep(0.3)


async def _silence_task(t: asyncio.Task):
    """Await a failed/cancelled task silently to suppress unhandled exception warnings."""
    try:
        await t
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────────────


# ── upstream call with disconnect guard ─────────────────────────────────
async def _upstream_guarded(
    method: str, url: str, body: bytes, headers: dict,
    disconnect: asyncio.Event,
    use_compressor: bool = False,
) -> httpx.Response:
    """Call upstream; abort on disconnect. Optionally use compressor client (longer timeout)."""
    client = get_compressor_client() if use_compressor else get_client()
    req = asyncio.create_task(
        client.request(method=method, url=url, content=body, headers=headers)
    )
    disc = asyncio.create_task(disconnect.wait())

    done, pending = await asyncio.wait(
        [req, disc],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()

    if disconnect.is_set():
        req.cancel()
        try:
            await req
        except (asyncio.CancelledError, Exception):
            pass
        raise asyncio.CancelledError("client disconnected")

    disc.cancel()
    return await req
# ────────────────────────────────────────────────────────────────────────


# ── POST /cancel endpoint ──────────────────────────────────────────────
@app.post("/cancel")
async def cancel_active():
    """Cancel all in-flight upstream tasks immediately."""
    killed = 0
    async with _task_lock:
        ids = list(_active_tasks.keys())
        for tid in ids:
            t = _active_tasks.get(tid)
            if t is not None and not t.done():
                t.cancel()
                killed += 1
    return {"cancelled": killed}
# ────────────────────────────────────────────────────────────────────────


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(request: Request, path: str):
    global _next_id
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers["Authorization"] = f"Bearer {API_KEY}"

    url = f"{UPSTREAM}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    print(f"[proxy] {request.method} {url}", flush=True)

    row_id = None
    preview = ""
    is_chat = path.endswith("/chat/completions") and request.method == "POST"
    t_start = datetime.now(timezone.utc)

    # ── Phase 1: detect compression, cap max_tokens, truncate ────
    is_compressor = False
    if is_chat and body:
        try:
            req_json = json.loads(body)
            prompt_text = " ".join(
                m.get("content", "") if isinstance(m.get("content"), str) else ""
                for m in req_json.get("messages", [])
            )
            is_compressor = COMPRESSION_PROMPT_MARKER in prompt_text

            if is_compressor:
                # ── Compressor: truncate input, disable thinking, long timeout ──
                msgs = req_json.get("messages", [])
                total_chars = sum(
                    len(m.get("content", "")) for m in msgs
                    if isinstance(m.get("content"), str)
                )
                if total_chars > COMPRESSOR_MAX_CHARS:
                    # Keep system prompt + last messages up to COMPRESSOR_MAX_CHARS
                    new_msgs = []
                    budget = COMPRESSOR_MAX_CHARS
                    for m in msgs:
                        content = m.get("content", "")
                        if isinstance(content, str):
                            if budget >= len(content):
                                new_msgs.append(m)
                                budget -= len(content)
                            else:
                                m["content"] = content[:budget]
                                new_msgs.append(m)
                                budget = 0
                                break
                    req_json["messages"] = new_msgs

                # Disable thinking for local model
                req_json.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
                for key in ("max_tokens", "max_completion_tokens"):
                    val = req_json.get(key, 4096)
                    if val is None or val > 8192:
                        req_json[key] = 8192
                body = json.dumps(req_json).encode()
                headers.pop("Content-Length", None)
                headers.pop("content-length", None)
                headers["Content-Length"] = str(len(body))
                for msg in req_json.get("messages", []):
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        preview = content[:120]
                        break
                row_id = _phase1_insert("compressor", preview)
            else:
                # ── Normal coder: cap max_tokens ──
                for key in ("max_tokens", "max_completion_tokens"):
                    val = req_json.get(key, CODER_MAX_TOKENS)
                    if val is None or val > CODER_MAX_TOKENS:
                        req_json[key] = CODER_MAX_TOKENS
                body = json.dumps(req_json).encode()
                headers.pop("Content-Length", None)
                headers.pop("content-length", None)
                headers["Content-Length"] = str(len(body))
                msgs = req_json.get("messages", [])
                for msg in msgs:
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        preview = content[:120]
                        break
                row_id = _phase1_insert("coder", preview)
        except Exception:
            pass
    # ─────────────────────────────────────────────────────────────

    disconnect = asyncio.Event()

    async with _task_lock:
        _next_id += 1
        task_id = _next_id

    monitor = asyncio.create_task(_monitor_disconnect(request, disconnect))
    upstream_task = asyncio.create_task(
        _upstream_guarded(request.method, url, body, headers, disconnect, use_compressor=is_compressor)
    )

    async with _task_lock:
        _active_tasks[task_id] = upstream_task

    try:
        finished, _ = await asyncio.wait(
            [upstream_task, monitor],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for ft in finished:
            if ft is upstream_task:
                monitor.cancel()
            else:
                upstream_task.cancel()
                # Consume the cancellation exception silently
                asyncio.create_task(_silence_task(upstream_task))

        elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)

        if upstream_task in finished and not upstream_task.cancelled():
            try:
                resp = upstream_task.result()
            except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError):
                # Upstream cancelled mid-read — treat as client disconnect
                if row_id:
                    _phase2_cancelled(row_id, elapsed)
                await replace_client()
                return Response(
                    content=json.dumps({"error": "cancelled", "detail": "upstream disconnected"}).encode(),
                    status_code=499,
                    media_type="application/json",
                    headers={"Connection": "close"},
                )
        else:
            # ── Client disconnected ──────────────────────────
            if row_id:
                _phase2_cancelled(row_id, elapsed)
            # Force-close the client to send RST to Strata engine
            await replace_client()
            return Response(
                content=json.dumps({"error": "cancelled", "detail": "client disconnected"}).encode(),
                status_code=499,
                media_type="application/json",
                headers={"Connection": "close"},
            )

        # ── Normal response ───────────────────────────────────
        if row_id and resp.status_code >= 400:
            _phase2_error(row_id, f"HTTP {resp.status_code}: {resp.text[:300]}", elapsed)
        elif row_id:
            try:
                data = resp.json()
                u = data.get("usage") or {}
                pt = u.get("prompt_tokens", 0)
                ct = u.get("completion_tokens", 0)
                _phase2_complete(row_id, data.get("model", "unknown"), pt, ct, elapsed)
            except Exception:
                _phase2_complete(row_id, "unknown", 0, 0, elapsed)

        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))

    except asyncio.CancelledError:
        if row_id:
            elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
            _phase2_cancelled(row_id, elapsed)
        await replace_client()
        return Response(
            content=json.dumps({"error": "cancelled"}).encode(),
            status_code=499,
            media_type="application/json",
            headers={"Connection": "close"},
        )
    finally:
        async with _task_lock:
            _active_tasks.pop(task_id, None)


if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")