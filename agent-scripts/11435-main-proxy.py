#!/usr/bin/env python3
"""Ollama Proxy on 11435 — logs main agent tokens. Handles streaming (SSE)."""
import asyncio, json, sqlite3
from datetime import datetime, timezone
from pathlib import Path
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

OLLAMA_UPSTREAM = "http://127.0.0.1:11434"
DB_PATH = Path.home() / "token-stats" / "data" / "tokens.db"
PROXY_PORT = 11435

app = FastAPI()
client = httpx.AsyncClient(timeout=300)

def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def set_status(agent, status, model="", provider=""):
    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO agent_status (agent_name,status,model,provider,last_call,updated_at) VALUES(?,?,?,?,datetime('now'),datetime('now')) "
                "ON CONFLICT(agent_name) DO UPDATE SET status=?,model=?,provider=?,last_call=datetime('now'),updated_at=datetime('now')",
                (agent, status, model, provider, status, model, provider))
            db.commit()
    except Exception:
        pass

def log_token(agent, prov_type, model, is_fb, pt, ct, tt, dur, code, streamed=0):
    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO token_log (agent_name,provider_type,model,is_fallback,prompt_tokens,completion_tokens,total_tokens,duration_ms,status_code,status,streamed) "
                "VALUES(?,?,?,?,?,?,?,?,?,'completed',?)",
                (agent, prov_type, model, 1 if is_fb else 0, pt, ct, tt, dur, code, streamed))
            db.commit()
    except Exception:
        pass

@app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE","PATCH","HEAD","OPTIONS"])
async def proxy(request: Request, path: str):
    t_start = datetime.now(timezone.utc)
    body = await request.body()
    url = f"{OLLAMA_UPSTREAM}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    is_chat = path.rstrip("/").endswith("chat/completions") and request.method == "POST"

    if not is_chat:
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        headers.pop("Content-Length", None)
        try:
            resp = await client.request(request.method, url, content=body, headers=headers)
        except Exception as e:
            return Response(json.dumps({"error": str(e)}).encode(), status_code=502)
        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))

    # ── Chat completions ────────────────────────────────────────────
    is_stream = False
    req_json = None
    if body:
        try:
            req_json = json.loads(body)
            is_stream = req_json.get("stream", False)
            if is_stream and "stream_options" not in req_json:
                req_json["stream_options"] = {"include_usage": True}
                body = json.dumps(req_json).encode()
        except Exception:
            pass

    set_status("main", "active", "qwen3-hermes", "local_moe")

    if is_stream:
        async def stream_and_log():
            pt = ct = tt = 0
            model_name = "qwen3-hermes"
            code = 200
            try:
                async with client.stream("POST", url, content=body) as resp:
                    code = resp.status_code
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                        try:
                            for line in chunk.decode().split("\n"):
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    data = json.loads(line[6:])
                                    usage = data.get("usage") or {}
                                    if usage:
                                        pt = abs(usage.get("prompt_tokens", usage.get("prompt_eval_count", 0)))
                                        ct = abs(usage.get("completion_tokens", usage.get("eval_count", 0)))
                                        tt = abs(usage.get("total_tokens", pt + ct))
                                    model_name = data.get("model", model_name)
                        except Exception:
                            pass
            except Exception as e:
                code = 502
            finally:
                dur = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
                if not pt or not ct:
                    if not pt: pt = max(1, len(body) // 4) if body else 80
                    if not ct: ct = 5
                    tt = pt + ct
                log_token("main", "local_moe", model_name, False, pt, ct, tt, dur, code, 1)
                set_status("main", "idle", model_name, "local_moe")
        return StreamingResponse(stream_and_log(), media_type="text/event-stream")

    else:
        try:
            resp = await client.request("POST", url, content=body)
        except Exception as e:
            set_status("main", "idle")
            return Response(json.dumps({"error": str(e)}).encode(), status_code=502)

        dur = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
        pt = ct = tt = 0
        model_name = "qwen3-hermes"
        try:
            data = resp.json()
            usage = data.get("usage") or {}
            pt = abs(usage.get("prompt_tokens", usage.get("prompt_eval_count", 0)))
            ct = abs(usage.get("completion_tokens", usage.get("eval_count", 0)))
            tt = abs(usage.get("total_tokens", pt + ct))
            model_name = data.get("model", model_name)
            if pt == 0 and ct == 0:
                # Fallback: approximate by text length
                pt = max(1, len(body) // 4) if body else 80
                content = ""
                for ch in (data.get("choices") or []):
                    msg = ch.get("message") or {}
                    content += msg.get("content") or ""
                ct = max(1, len(content) // 4) if content else 5
                tt = pt + ct
        except Exception:
            pass

        log_token("main", "local_moe", model_name, False, pt, ct, tt, dur, resp.status_code, 0)
        set_status("main", "idle", model_name, "local_moe")
        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PROXY_PORT, log_level="warning")