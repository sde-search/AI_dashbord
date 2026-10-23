"""Ask subagent module — calls proxy ports (8090-8094), returns response text.
Usage: from ask_subagent import ask_devops, ask_network, ask_backup, ask_fs, ask_codereview
"""

import json, urllib.request, urllib.error

PROXY_BASE = "http://127.0.0.1"
MODEL = "deepseek/deepseek-v4-flash-0731"  # routed through 8090->8084->Polza AI
TIMEOUT = 120

def _ask(port: int, prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1536,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{PROXY_BASE}:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read())
        c = data["choices"][0]["message"]
        return c.get("content") or c.get("reasoning_content", "") or "[empty]"
    except Exception as e:
        return f"[ERROR] {e}"

def ask_devops(prompt: str) -> str:
    return _ask(8090, prompt)

def ask_network(prompt: str) -> str:
    return _ask(8091, prompt)

def ask_backup(prompt: str) -> str:
    return _ask(8092, prompt)

def ask_fs(prompt: str) -> str:
    return _ask(8093, prompt)

def ask_codereview(prompt: str) -> str:
    return _ask(8094, prompt)