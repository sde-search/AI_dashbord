#!/usr/bin/env python3
"""
Outbound Proxy — единая точка выхода главного агента к внешним LLM API.
Логирует токены, стоимость и статус в единую SQLite.
Двухфазное логирование: running → completed/error.
"""
import json, os, sqlite3, re, asyncio
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

# ── Config ──────────────────────────────────────────────────────────

TARGET_URL = os.environ.get("LLM_PROXY_TARGET", "https://polza.ai/api/v1")
TARGET_API_KEY = os.environ.get("LLM_PROXY_API_KEY", "")

PROXY_LISTEN = os.environ.get("LLM_PROXY_LISTEN", "0.0.0.0")
PROXY_PORT = int(os.environ.get("LLM_PROXY_PORT", "8084"))

# ── Compression interception ────────────────────────────────────────────
# Detect compression summarization requests and redirect to local Strata
# (free) instead of the paid upstream API.
COMPRESSOR_MODEL = os.environ.get("LLM_COMPRESSOR_MODEL", "qwen3.8-flash-next-coder-iq1_m")
COMPRESSOR_URL = os.environ.get("LLM_COMPRESSOR_URL", "http://127.0.0.1:8083/v1/chat/completions")
COMPRESSOR_API_KEY = os.environ.get("LLM_COMPRESSOR_API_KEY", "strata-local")
COMPRESSION_PROMPT_MARKER = "You are a summarization agent creating a context checkpoint. Treat the conversation turns below as source material for a compact record of prior work."

# ── Security Reviewer interception ────────────────────────────────────
# Redirect Hermes' built-in security reviewer to local Strata (free)
SECURITY_REVIEW_MARKER = "You are a security reviewer for an AI coding agent"

DB_DIR = Path.home() / "token-stats" / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "tokens.db"

DB_TIMEOUT = 5.0
STALE_SECONDS = 300

app = FastAPI()
client = httpx.AsyncClient(timeout=300)

# ── Pricing Dictionary ──────────────────────────────────────────────
# Цена за 1M токенов: (input, output)
# $0 = бесплатно/локально, None = unknown (не считаем)

PRICING: dict[str, tuple[float | None, float | None]] = {
    "gpt-4o":                        (5.0, 15.0),
    "gpt-4o-":                       (2.5, 10.0),
    "gpt-4.1":                       (2.0, 8.0),
    "gpt-4.1-mini":                  (0.4, 1.6),
    "gpt-4.1-nano":                  (0.1, 0.4),
    "gpt-5":                         (10.0, 40.0),
    "gpt-5-codex":                   (10.0, 40.0),
    "gpt-5-mini":                    (1.5, 6.0),
    "gpt-5.1":                       (10.0, 40.0),
    "gpt-5.2":                       (10.0, 40.0),
    "gpt-5.3":                       (10.0, 40.0),
    "gpt-5.4":                       (10.0, 40.0),
    "gpt-5.5":                       (15.0, 60.0),
    "gpt-5.6":                       (15.0, 60.0),
    "gpt-6":                         (20.0, 80.0),
    "o1":                            (15.0, 60.0),
    "o1-mini":                       (1.1, 4.4),
    "o3":                            (10.0, 40.0),
    "o3-mini":                       (1.1, 4.4),
    "o4-mini":                       (1.1, 4.4),
    "claude-opus-4":                 (15.0, 75.0),
    "claude-opus-4.":                (15.0, 75.0),
    "claude-sonnet-4":               (3.0, 15.0),
    "claude-sonnet-5":               (3.0, 15.0),
    "claude-haiku-4":                (0.8, 4.0),
    "claude-fable":                  (3.0, 15.0),
    "gemini-3":                      (0.5, 2.0),
    "gemini-3.1":                    (0.5, 2.0),
    "gemini-3.5":                    (0.5, 2.0),
    "gemini-3.6":                    (0.5, 2.0),
    "gemini-3.7":                    (0.5, 2.0),
    "gemini-3.8":                    (0.5, 2.0),
    "gemini-2.5":                    (1.25, 10.0),
    "gemma-4":                       (0.0, 0.0),
    "deepseek/deepseek-v4":          (2.0, 6.0),
    "deepseek/deepseek-v3":          (1.0, 4.0),
    "deepseek/deepseek-chat":        (0.5, 2.0),
    "deepseek/deepseek-r1":          (0.5, 2.0),
    "qwen/qwen3":                    (0.0, 0.0),
    "qwen/qwen3.5":                  (0.0, 0.0),
    "qwen/qwen3.6":                  (0.0, 0.0),
    "qwen/qwen3.7":                  (0.0, 0.0),
    "qwen/qwen3.8":                  (0.0, 0.0),
    "qwen/qwen-plus":                (0.0, 0.0),
    "qwen/qwen-2.5":                 (0.0, 0.0),
    "qwen/qwen3-coder":              (0.0, 0.0),
    "moonshotai/kimi":               (0.5, 2.0),
    "minimax/minimax-m":             (0.5, 2.0),
    "z-ai/glm-5":                    (1.0, 4.0),
    "z-ai/glm-4":                    (0.5, 2.0),
    "meta-llama/llama-4":            (0.0, 0.0),
    "meta-llama/llama-3":            (0.0, 0.0),
    "mistralai/mistral":             (0.5, 2.0),
    "mistralai/codestral":           (0.5, 2.0),
    "x-ai/grok-4":                   (5.0, 15.0),
    "x-ai/grok-3":                   (3.0, 10.0),
    "aion-labs/aion":                (0.0, 0.0),
    "perplexity/sonar":              (5.0, 15.0),
}

DEFAULT_INPUT_RATE = 5.0
DEFAULT_OUTPUT_RATE = 15.0


def get_pricing(model: str) -> tuple[float, float]:
    model_lower = model.lower().strip()
    for prefix, rates in sorted(PRICING.items(), key=lambda x: -len(x[0])):
        if model_lower.startswith(prefix):
            i, o = rates
            if i is None or o is None:
                continue
            return (i, o)
    return (DEFAULT_INPUT_RATE, DEFAULT_OUTPUT_RATE)


# ── DB ──────────────────────────────────────────────────────────────

def init_db():
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        # Columns already exist in DB - skipping ALTER TABLE
        pass  # Table already exists in DB with agent_name column
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_agent ON token_log(ts, agent_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON token_log(status)")
        conn.commit()


def _mark_stale_as_timeout():
    cutoff = datetime.utcnow().isoformat() + "Z"
    try:
        with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
            rows = conn.execute(
                "SELECT id, ts FROM token_log WHERE agent_name='main' AND status='running'"
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


def _phase1_insert(agent_type, task_preview: str) -> int:
    ts = datetime.utcnow().isoformat() + "Z"
    _mark_stale_as_timeout()
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        cur = conn.execute(
            "INSERT INTO token_log (ts, agent_name, model, prompt_tokens, completion_tokens, cost, status, task_preview) VALUES (?, ?, ?, 0, 0, 0, 'running', ?)",
            (ts, agent_type, "", task_preview),
        )
        conn.commit()
        return cur.lastrowid or 0


def _phase2_complete(row_id: int, model: str, pt: int, ct: int, cost: float, duration_ms: int, target: str = ""):
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute(
            "UPDATE token_log SET status='completed', model=?, prompt_tokens=?, completion_tokens=?, cost=?, duration_ms=?, endpoint_target=? WHERE id=? AND status='running'",
            (model, pt, ct, cost, duration_ms, target, row_id),
        )
        conn.commit()


def _phase2_error(row_id: int, error_msg: str, duration_ms: int = 0):
    with sqlite3.connect(str(DB_PATH), timeout=DB_TIMEOUT) as conn:
        conn.execute(
            "UPDATE token_log SET status='error', error_msg=?, duration_ms=? WHERE id=? AND status='running'",
            (error_msg[:500], duration_ms, row_id),
        )
        conn.commit()


def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Приблизительный подсчёт токенов без tiktoken (4 симв → 1 токен)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Proxy routes ────────────────────────────────────────────────────

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(request: Request, path: str):
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    # Detect subagent requests (8090-8094) - skip local DB logging, they log themselves
    _is_subagent = any(f"127.0.0.1:{p}" in str(request.headers.get("x-forwarded-for", "")) 
                       for p in [8090, 8091, 8092, 8093, 8094])
    # Also detect by checking if a request came from localhost subagent ports
    _from_subagent = "8090" in str(headers) or "8091" in str(headers) or "8092" in str(headers) or "8093" in str(headers) or "8094" in str(headers)

    if TARGET_API_KEY and "authorization" not in headers:
        headers["Authorization"] = f"Bearer {TARGET_API_KEY}"

    url = f"{TARGET_URL.rstrip('/')}/{path}"
    if TARGET_URL.rstrip('/').endswith('/v1') and path.startswith('v1/'):
        url = f"{TARGET_URL.rstrip('/')}/{path[3:]}"
    if request.url.query:
        url += f"?{request.url.query}"

    is_stream = False
    model_name = "unknown"
    prompt_text = ""
    task_preview = ""


    # ── Provider block injection for Polza AI ──────────────────────────
    # Ensure subagent requests (8090-8094) include the 4-provider filter + sort:price
    # (injected at the correct position below, before streaming/non-streaming dispatch)

    # ФАЗА 1: извлекаем превью, вставляем running-запись
    if body and request.method == "POST":
        try:
            req_json = json.loads(body)
            model_name = req_json.get("model", "unknown")
            is_stream = req_json.get("stream", False)
            msgs = req_json.get("messages", [])
            for msg in msgs:
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    task_preview = content[:120]
                    break
            prompt_text = " ".join(
                m.get("content", "") if isinstance(m.get("content"), str) else ""
                for m in msgs
            )
        except (json.JSONDecodeError, TypeError):
            pass

    t_start = datetime.now(timezone.utc)

    # ── Detect subagent requests by X-Subagent header ──────────
    _is_from_subagent = headers.get("x-subagent", "") in ["8090", "8091", "8092", "8093", "8094"]

    # ── Compression detector (non-streaming only) ──────────────
    _is_compressor = False
    if body and not is_stream and COMPRESSION_PROMPT_MARKER in prompt_text:
        _is_compressor = True
        try:
            req_json = json.loads(body)
            req_json["model"] = COMPRESSOR_MODEL
            req_json["max_tokens"] = req_json.pop("max_tokens", 2048)
            req_json.pop("stream", None)
            req_json.pop("stream_options", None)
            # Add chat_template_kwargs: disable thinking for local models
            req_json.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
            body = json.dumps(req_json).encode()
            # Fix Content-Length after body was modified
            headers.pop("Content-Length", None)
            headers.pop("content-length", None)
        except Exception:
            _is_compressor = False  # on parse error, let it fall through normally

    # ── Security Reviewer detector ────────────────────────────
    _is_security_review = False
    if body and not is_stream and SECURITY_REVIEW_MARKER in prompt_text:
        _is_security_review = True
        try:
            req_json = json.loads(body)
            req_json["model"] = COMPRESSOR_MODEL
            req_json["max_tokens"] = 256
            req_json.pop("stream", None)
            req_json.pop("stream_options", None)
            req_json.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
            body = json.dumps(req_json).encode()
            headers.pop("Content-Length", None)
            headers.pop("content-length", None)
        except Exception:
            _is_security_review = False

    # Sub-agent proxies (8090-8094) and 11435 log themselves
    if _is_from_subagent:
        row_id = 0  # Skip logging, subagent logs itself
    else:
        row_id = _phase1_insert(
            "compressor" if _is_compressor else
            "codereview" if _is_security_review else
            "main",
            task_preview
        )

    if _is_compressor:
        # Redirect to local Strata — free, no cost
        comp_headers = dict(headers)
        comp_headers.pop("authorization", None)
        if COMPRESSOR_API_KEY:
            comp_headers["Authorization"] = f"Bearer {COMPRESSOR_API_KEY}"
        comp_url = COMPRESSOR_URL
        try:
            async with client.stream("POST", comp_url, content=body, headers=comp_headers) as resp:
                content = await resp.aread()
        except Exception as e:
            elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
            _phase2_error(row_id, f"Compressor upstream: {str(e)[:300]}", elapsed)
            return Response(content=json.dumps({"error": str(e)}).encode(),
                            status_code=502, media_type="application/json")
        elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
        if resp.status_code >= 400:
            _phase2_error(row_id, f"HTTP {resp.status_code}: {resp.text[:300]}", elapsed)
        else:
            try:
                data = json.loads(content)
                usage = data.get("usage") or {}
                pt = abs(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
                ct = abs(usage.get("completion_tokens", usage.get("output_tokens", 0)))
                resp_model = data.get("model", COMPRESSOR_MODEL)
                if not pt and not ct:
                    completion_text = ""
                    for choice in data.get("choices", []):
                        msg = choice.get("message", {})
                        completion_text += msg.get("content", "") or ""
                    pt = _count_tokens(prompt_text, resp_model)
                    ct = _count_tokens(completion_text, resp_model)
                # compressor: cost=0.0 always
                _phase2_complete(row_id, resp_model, pt, ct, 0.0, elapsed, "compressor")
            except (json.JSONDecodeError, TypeError) as e:
                _phase2_error(row_id, f"Compressor parse error: {str(e)[:300]}", elapsed)
        return Response(content=content, status_code=resp.status_code, headers=dict(resp.headers))

    # ── Provider block injection for Polza AI ──────────────────────────
    # Inject 4-provider filter + sort:price if not already present
    if body and request.method == "POST" and not _is_compressor and not _is_security_review:
        try:
            req_json2 = json.loads(body)
            if "provider" not in req_json2.get("extra_body", {}):
                extras = req_json2.get("extra_body", {})
                if isinstance(extras, dict):
                    extras["provider"] = {
                        "only": ["gmicloud/fp8", "streamlake/fp8", "venice", "digitalocean"],
                        "sort": "price",
                        "allow_fallbacks": False
                    }
                    req_json2["extra_body"] = extras
                    body = json.dumps(req_json2).encode()
                    headers.pop("Content-Length", None)
                    headers.pop("content-length", None)
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Security Reviewer redirect ─────────────────────────────────
    if _is_security_review:
        sec_headers = dict(headers)
        sec_headers.pop("authorization", None)
        if COMPRESSOR_API_KEY:
            sec_headers["Authorization"] = f"Bearer {COMPRESSOR_API_KEY}"
        sec_url = COMPRESSOR_URL
        try:
            async with client.stream("POST", sec_url, content=body, headers=sec_headers) as resp:
                content = await resp.aread()
        except Exception as e:
            elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
            _phase2_error(row_id, f"SecurityReview upstream: {str(e)[:300]}", elapsed)
            return Response(content=json.dumps({"error": str(e)}).encode(),
                            status_code=502, media_type="application/json")
        elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
        if resp.status_code >= 400:
            _phase2_error(row_id, f"HTTP {resp.status_code}: {resp.text[:300]}", elapsed)
        else:
            try:
                data = json.loads(content)
                usage = data.get("usage") or {}
                pt = abs(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
                ct = abs(usage.get("completion_tokens", usage.get("output_tokens", 0)))
                resp_model = data.get("model", COMPRESSOR_MODEL)
                if not pt and not ct:
                    completion_text = ""
                    for choice in data.get("choices", []):
                        msg = choice.get("message", {})
                        completion_text += msg.get("content", "") or ""
                    pt = _count_tokens(prompt_text, resp_model)
                    ct = _count_tokens(completion_text, resp_model)
                # codereview: cost=0.0 always (free local)
                _phase2_complete(row_id, resp_model, pt, ct, 0.0, elapsed, "codereview")
            except (json.JSONDecodeError, TypeError) as e:
                _phase2_error(row_id, f"SecurityReview parse error: {str(e)[:300]}", elapsed)
        return Response(content=content, status_code=resp.status_code, headers=dict(resp.headers))

    # ── Streaming ───────────────────────────────────────────────
    if is_stream:
        async def stream_and_log(row_id=row_id, t_start=t_start, model_name=model_name,
                                  prompt_text=prompt_text, path=path,
                                  body=body, headers=headers, url=url):
            collected_chunks = []
            usage_found = None
            completion_text = ""

            try:
                async with client.stream("POST", url, content=body, headers=headers) as resp:
                    async for chunk in resp.aiter_bytes():
                        collected_chunks.append(chunk)
                        yield chunk

                full = b"".join(collected_chunks).decode("utf-8", errors="replace")
                for line in full.split("\n"):
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "usage" in data:
                                usage_found = data["usage"]
                                break
                            delta = (data.get("choices", [{}])[0]
                                     .get("delta", {})
                                     .get("content", ""))
                            if delta:
                                completion_text += delta
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass

                elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)

                if usage_found:
                    pt = usage_found.get("prompt_tokens", 0)
                    ct = usage_found.get("completion_tokens", 0)
                else:
                    pt = _count_tokens(prompt_text, model_name)
                    ct = len(completion_text) // 4
                    if ct < 1:
                        ct = _count_tokens(completion_text, model_name)

                cost = _calc_cost(model_name, pt, ct)
                _phase2_complete(row_id, model_name, pt, ct, cost, elapsed, path)

            except Exception as e:
                elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
                _phase2_error(row_id, f"Stream error: {str(e)[:300]}", elapsed)

        return StreamingResponse(stream_and_log(), media_type="text/event-stream")

    # ── Non-streaming ─────────────────────────────────────────────
    try:
        async with client.stream("POST" if body else request.method,
                                  url, content=body, headers=headers) as resp:
            content = await resp.aread()
    except Exception as e:
        elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
        _phase2_error(row_id, f"Upstream request failed: {str(e)[:300]}", elapsed)
        return Response(content=json.dumps({"error": str(e)}).encode(),
                        status_code=502, media_type="application/json")

    elapsed = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)

    if resp.status_code >= 400:
        _phase2_error(row_id, f"HTTP {resp.status_code}: {resp.text[:300]}", elapsed)
    else:
        try:
            data = json.loads(content)
            usage = data.get("usage") or {}
            pt = abs(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            ct = abs(usage.get("completion_tokens", usage.get("output_tokens", 0)))
            resp_model = data.get("model", model_name)
            if pt or ct:
                cost = _calc_cost(resp_model, pt, ct)
                _phase2_complete(row_id, resp_model, pt, ct, cost, elapsed, path)
            else:
                completion_text = ""
                for choice in data.get("choices", []):
                    msg = choice.get("message", {})
                    completion_text += msg.get("content", "") or ""
                pt = _count_tokens(prompt_text, resp_model)
                ct = _count_tokens(completion_text, resp_model)
                cost = _calc_cost(resp_model, pt, ct)
                _phase2_complete(row_id, resp_model, pt, ct, cost, elapsed, path)
        except (json.JSONDecodeError, TypeError) as e:
            _phase2_error(row_id, f"Parse error: {str(e)[:300]}", elapsed)

    return Response(content=content, status_code=resp.status_code, headers=dict(resp.headers))


def _calc_cost(model: str, pt: int, ct: int) -> float:
    in_rate, out_rate = get_pricing(model)
    return (pt * in_rate / 1_000_000) + (ct * out_rate / 1_000_000)


# ── Health ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "pricing_rules": len(PRICING)}


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print(f"[proxy] Target: {TARGET_URL}")
    print(f"[proxy] Listening on {PROXY_LISTEN}:{PROXY_PORT}")
    print(f"[proxy] DB: {DB_PATH}")
    uvicorn.run(app, host=PROXY_LISTEN, port=PROXY_PORT, log_level="info")