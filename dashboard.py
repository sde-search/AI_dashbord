#!/usr/bin/env python3
"""Token usage dashboard + hardware telemetry + emergency controls — port 8500."""
import sqlite3, json, os, subprocess, time, threading
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

DB = Path.home() / "token-stats" / "data" / "tokens.db"
AGENT_META = {
    "main":       { "label": "Main (Local MoE)",  "color": "#58a6ff", "bg": "#1f6feb33",
                    "desc": "qwen3-hermes через port 11435" },
    "devops":     { "label": "DevOps (8090)",      "color": "#d2a8ff", "bg": "#d2a8ff33",
                    "desc": "Polza API -> Fallback Local MoE" },
    "network":    { "label": "Network (8091)",     "color": "#79c0ff", "bg": "#79c0ff33",
                    "desc": "Polza API -> Fallback Local MoE" },
    "backup":     { "label": "Backup (8092)",      "color": "#7ee787", "bg": "#7ee78733",
                    "desc": "Polza API -> Fallback Local MoE" },
    "fs":         { "label": "FS (8093)",          "color": "#ffa657", "bg": "#ffa65733",
                    "desc": "Polza API -> Fallback Local MoE" },
    "codereview": { "label": "CodeReview (8094)",  "color": "#ff7b72", "bg": "#ff7b7233",
                    "desc": "Polza API -> Fallback Local MoE" },
    "compression":{ "label": "Compression (Polza)","color": "#f0883e", "bg": "#f0883e33",
                    "desc": "Compression via Polza DeepSeek" },
}
ORDER = ["main","devops","network","backup","fs","codereview","compression"]

app = FastAPI()

@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try: yield conn
    finally: conn.close()

def to_int(v, defv=0):
    try: return int(v)
    except: return defv

# ── API: Stats ───────────────────────────────────────────────────
@app.get("/api/stats")
def api_stats(since: int = Query(3600)):
    cutoff = datetime.now(timezone.utc).isoformat()
    # fallback: use ts-based filter
    with get_db() as db:
        rows = db.execute(
            "SELECT agent_name, provider_type, model, is_fallback, prompt_tokens, completion_tokens, total_tokens, duration_ms, status_code "
            "FROM token_log WHERE status='completed' ORDER BY ts DESC LIMIT 1000"
        ).fetchall()

    totals = {}
    daily = {}
    local_tokens = 0
    polza_tokens = 0
    fallback_count = 0

    for r in rows:
        a = r["agent_name"]
        pt = to_int(r["prompt_tokens"])
        ct = to_int(r["completion_tokens"])
        tt = to_int(r["total_tokens"], pt + ct)
        prov = r["provider_type"] or "unknown"
        is_fb = to_int(r["is_fallback"])

        totals.setdefault(a, {"calls":0,"prompt":0,"completion":0,"total":0})
        totals[a]["calls"] += 1
        totals[a]["prompt"] += pt
        totals[a]["completion"] += ct
        totals[a]["total"] += tt

        if prov == "local_moe":
            local_tokens += tt
            if is_fb:
                fallback_count += 1
        else:
            polza_tokens += tt

    grand = {"calls": sum(v["calls"] for v in totals.values()),
             "prompt": sum(v["prompt"] for v in totals.values()),
             "completion": sum(v["completion"] for v in totals.values()),
             "total": sum(v["total"] for v in totals.values()),
             "local_tokens": local_tokens, "polza_tokens": polza_tokens,
             "fallback_count": fallback_count}

    return {"totals": totals, "grand": grand}

# ── API: Logs ────────────────────────────────────────────────────
@app.get("/api/logs")
def api_logs(limit: int = Query(50)):
    with get_db() as db:
        rows = db.execute(
            "SELECT ts, agent_name, provider_type, model, is_fallback, prompt_tokens, completion_tokens, total_tokens, duration_ms, status_code, status "
            "FROM token_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"logs": [dict(r) for r in rows]}

# ── API: Agent Status ────────────────────────────────────────────
@app.get("/api/status")
def api_status():
    with get_db() as db:
        rows = db.execute("SELECT agent_name, status, model, provider, last_call, updated_at FROM agent_status").fetchall()
    return {"agents": [dict(r) for r in rows]}

# ── API: Hardware Telemetry ─────────────────────────────────────
@app.get("/api/hardware")
def api_hardware():
    data = {"cpu": 0, "ram_used_gb": 0, "ram_total_gb": 31,
            "gpu_util": 0, "vram_used_mib": 0, "vram_total_mib": 12288, "gpu_temp": 0,
            "loaded_model": ""}
    try:
        l1 = open("/proc/loadavg").read().split()[0]
        data["cpu"] = round(float(l1) * 100 / os.cpu_count(), 1)
    except: pass
    try:
        m = open("/proc/meminfo").read()
        total = int([l for l in m.split("\n") if "MemTotal" in l][0].split()[1]) // 1024 // 1024
        avail = int([l for l in m.split("\n") if "MemAvailable" in l][0].split()[1]) // 1024 // 1024
        data["ram_total_gb"] = total
        data["ram_used_gb"] = round(total - avail, 1)
    except: pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], timeout=5, stderr=subprocess.DEVNULL).decode().strip()
        parts = out.split(",")
        data["gpu_util"] = float(parts[0].strip())
        data["vram_used_mib"] = float(parts[1].strip())
        data["vram_total_mib"] = float(parts[2].strip())
        data["gpu_temp"] = float(parts[3].strip())
    except: pass
    try:
        out = subprocess.check_output(["ollama","ps"], timeout=5, stderr=subprocess.DEVNULL).decode().strip().split("\n")
        if len(out) > 1:
            parts = out[1].split()
            data["loaded_model"] = parts[0] if parts else ""
    except: pass
    return data

# ── API: Emergency Controls ─────────────────────────────────────
@app.post("/api/control/emergency-stop")
def emergency_stop():
    os.system("pkill -f 'apt|dpkg' 2>/dev/null")
    os.system("sudo pkill -f 'apt|dpkg' 2>/dev/null")
    with get_db() as db:
        db.execute("UPDATE agent_status SET status='idle', updated_at=datetime('now')")
        db.commit()
    os.system("sudo systemctl restart hermes-gateway 2>/dev/null")
    return {"status": "ok", "message": "Agents reset, gateway restarted"}

@app.post("/api/control/unload-vram")
def unload_vram():
    os.system("curl -s http://127.0.0.1:11434/api/generate -d '{\"model\":\"qwen3-hermes\",\"keep_alive\":0}' >/dev/null 2>&1")
    return {"status": "ok", "message": "VRAM unloaded (keep_alive=0)"}

@app.post("/api/control/restart-proxies")
def restart_proxies():
    for p in ["11435-main-proxy.py","outbound-proxy.py","8090-proxy.py","8091-proxy.py","8092-proxy.py","8093-proxy.py","8094-proxy.py"]:
        os.system(f"pkill -f '{p}' 2>/dev/null")
    time.sleep(1)
    for p, port in [("11435-main-proxy.py",11435),("outbound-proxy.py",8084),
                    ("8090-proxy.py",8090),("8091-proxy.py",8091),("8092-proxy.py",8092),
                    ("8093-proxy.py",8093),("8094-proxy.py",8094)]:
        os.system(f"cd /home/hermes/token-stats/agent-scripts && nohup python3 {p} > /dev/null 2>&1 &")
    return {"status": "ok", "message": "All proxies restarted"}

# ── API: Download agent log ────────────────────────────────────
@app.get("/api/log/download")
def download_log(tail: int = Query(0)):
    log_path = Path.home() / ".hermes" / "logs" / "agent.log"
    if not log_path.exists():
        return {"error": "Log file not found"}
    if tail > 0:
        from fastapi.responses import PlainTextResponse
        lines = subprocess.check_output(["tail", "-" + str(tail), str(log_path)]).decode()
        return PlainTextResponse(lines)
    from fastapi.responses import FileResponse
    return FileResponse(str(log_path), media_type="text/plain", filename="agent.log")

# ── HTML Dashboard ──────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token Stats Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;padding:20px}
h2{font-size:18px;margin:20px 0 10px;color:#f0f6fc}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.stat{display:flex;justify-content:space-between;padding:2px 0}
.l{color:#8b949e}
.v{color:#f0f6fc;font-weight:600;font-family:'SF Mono','Fira Code',monospace}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}
.badge-green{background:#23863633;color:#3fb950;border:1px solid #3fb95055}
.badge-blue{background:#1f6feb33;color:#58a6ff;border:1px solid #58a6ff55}
.badge-orange{background:#d2992233;color:#d29922;border:1px solid #d2992255}
.badge-red{background:#da363333;color:#f85149;border:1px solid #f8514955}
.kpi{font-size:28px;font-weight:700;margin:4px 0}
.kpi-label{font-size:11px;color:#8b949e;text-transform:uppercase}
.progress-bar{height:8px;background:#21262d;border-radius:4px;overflow:hidden;margin:4px 0}
.progress-fill{height:100%;background:#58a6ff;border-radius:4px;transition:width .5s}
canvas{width:100%!important;height:120px!important;border-radius:4px;background:#0d1117}
button{padding:8px 14px;border:1px solid #30363d;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500;background:#21262d;color:#c9d1d9;transition:.2s}
button:hover{background:#30363d;border-color:#8b949e}
button.danger{background:#da363333;border-color:#f8514955;color:#f85149}
button.danger:hover{background:#da363366}
.toast{position:fixed;bottom:20px;right:20px;background:#1f6feb;color:#fff;padding:10px 20px;border-radius:8px;font-size:14px;z-index:999;opacity:0;transition:opacity .3s}
.toast.show{opacity:1}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{padding:6px 8px;text-align:left;border-bottom:1px solid #21262d;white-space:nowrap}
th{color:#8b949e;font-weight:500}
.status-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.status-active{background:#3fb950;box-shadow:0 0 6px #3fb95066}
.status-idle{background:#484f58}
.control-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
</style>
</head>
<body>
<div id="app"></div>
<div id="toast" class="toast"></div>
<script>
const n = v => { if(v===undefined||v===null) return '0'; const s=Math.abs(v).toString(); return (v<0?'-':'')+s.replace(/\B(?=(\d{3})+(?!\d))/g,' '); };
const toast = m => { const t=document.getElementById('toast'); t.textContent=m; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),3000); };

async function load(url) { const r=await fetch(url); return r.json(); }
async function post(url) { const r=await fetch(url,{method:'POST'}); const d=await r.json(); toast(d.message||d.status); return d; }

const meta = """ + json.dumps(AGENT_META) + r""";
const order = """ + json.dumps(ORDER) + r""";

function renderKPI(grand) {
  const localPct = grand.total ? Math.round(grand.local_tokens/grand.total*100) : 0;
  return `<div class="grid">
    <div class="card"><div class="kpi-label">Local MoE (Free)</div><div class="kpi" style="color:#3fb950">${n(grand.local_tokens)}</div><div class="kpi-label">tokens</div></div>
    <div class="card"><div class="kpi-label">Polza API (Paid)</div><div class="kpi" style="color:#58a6ff">${n(grand.polza_tokens)}</div><div class="kpi-label">tokens</div></div>
    <div class="card"><div class="kpi-label">Local Share</div><div class="kpi">${localPct}%</div><div class="progress-bar"><div class="progress-fill" style="width:${localPct}%"></div></div></div>
    <div class="card"><div class="kpi-label">Fallbacks</div><div class="kpi" style="color:#d29922">${n(grand.fallback_count)}</div><div class="kpi-label">sub-agents saved by MoE</div></div>
  </div>`;
}

function renderAgentStatus(agents, totals) {
  const m = {};
  (agents||[]).forEach(a => m[a.agent_name] = a);
  return order.map(k => {
    const meta_k = meta[k]||{label:k,color:'#fff',bg:'#333'};
    const a = m[k]||{status:'idle',model:'',provider:'',last_call:'-'};
    const t = (totals||{})[k]||{calls:0,total:0};
    const active = a.status === 'active';
    return `<div class="card" style="border-left:3px solid ${meta_k.color}">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <strong style="color:${meta_k.color}">${meta_k.label}</strong>
        <span><span class="status-dot ${active?'status-active':'status-idle'}"></span>${active?'ACTIVE':'IDLE'}</span>
      </div>
      <div style="margin-top:6px;font-size:12px;color:#8b949e">${meta_k.desc}</div>
      <div class="grid" style="margin-top:6px;grid-template-columns:1fr 1fr">
        <div><span class="l">Model:</span><span class="v"> ${a.model||'-'}</span></div>
        <div><span class="l">Provider:</span><span class="v"> ${a.provider||'-'}</span></div>
        <div><span class="l">Calls:</span><span class="v"> ${n(t.calls)}</span></div>
        <div><span class="l">Tokens:</span><span class="v"> ${n(t.total)}</span></div>
      </div>
    </div>`;
  }).join('');
}

function renderLogs(logs) {
  if(!logs||!logs.length) return '<div class="card" style="color:#8b949e">No requests yet — send a message in Telegram</div>';
  return `<table><tr><th>Time</th><th>Agent</th><th>Provider</th><th>Model</th><th>Prompt</th><th>Comp</th><th>Total</th><th>ms</th></tr>
    ${logs.map(r => {
      const provType = r.provider_type||'unknown';
      const isFb = r.is_fallback;
      const badge = provType==='local_moe' ? (isFb?'badge-orange':'badge-green') : 'badge-blue';
      const badgeText = provType==='local_moe' ? (isFb?'FALLBACK -> MoE':'LOCAL MoE') : 'POLZA API';
      const meta_k = meta[r.agent_name]||{color:'#fff'};
      return `<tr>
        <td style="color:#8b949e">${(r.ts||'').slice(11,19)}</td>
        <td><span class="badge" style="background:${meta_k.bg};color:${meta_k.color};border:1px solid ${meta_k.color}55">${(r.agent_name||'?').toUpperCase()}</span></td>
        <td><span class="badge ${badge}">${badgeText}</span></td>
        <td>${r.model||'-'}</td>
        <td style="font-family:monospace">${n(r.prompt_tokens)}</td>
        <td style="font-family:monospace">${n(r.completion_tokens)}</td>
        <td style="font-family:monospace;font-weight:600">${n(r.total_tokens)}</td>
        <td style="font-family:monospace">${n(r.duration_ms)}</td>
      </tr>`;
    }).join('')}</table>`;
}

function renderHardware(h) {
  const vramPct = h.vram_total_mib ? Math.round(h.vram_used_mib/h.vram_total_mib*100) : 0;
  const ramPct = h.ram_total_gb ? Math.round(h.ram_used_gb/h.ram_total_gb*100) : 0;
  return `<div class="grid">
    <div class="card"><div class="kpi-label">GPU ${n(h.gpu_util)}%</div><div class="progress-bar"><div class="progress-fill" style="width:${h.gpu_util}%;background:#3fb950"></div></div>
      <div class="stat"><span class="l">VRAM</span><span class="v">${n(h.vram_used_mib)} / ${n(h.vram_total_mib)} MiB</span></div>
      <div class="progress-bar"><div class="progress-fill" style="width:${vramPct}%;background:#58a6ff"></div></div>
      <div class="stat"><span class="l">Temp</span><span class="v">${n(h.gpu_temp)}°C</span></div></div>
    <div class="card"><div class="kpi-label">CPU ${n(h.cpu)}%</div><div class="progress-bar"><div class="progress-fill" style="width:${Math.min(h.cpu,100)}%"></div></div>
      <div class="stat"><span class="l">RAM</span><span class="v">${h.ram_used_gb} / ${h.ram_total_gb} GB</span></div>
      <div class="progress-bar"><div class="progress-fill" style="width:${ramPct}%"></div></div>
      <div class="stat"><span class="l">Model</span><span class="v">${h.loaded_model||'-'}</span></div></div>
  </div>`;
}

function render() {
  Promise.all([load('/api/stats'), load('/api/logs?limit=30'), load('/api/status'), load('/api/hardware')])
    .then(([stats, logs, status, hw]) => {
      document.getElementById('app').innerHTML = `
        <h2>Token Stats — Live Monitor</h2>
        ${renderKPI(stats.grand)}
        <h2>Hardware Telemetry</h2>
        ${renderHardware(hw)}
        <h2>Emergency Controls</h2>
        <div class="control-row">
          <button class="danger" onclick="post('/api/control/emergency-stop').then(render)">🛑 Emergency Stop & Restart</button>
          <button onclick="post('/api/control/unload-vram').then(render)">🧹 Unload VRAM</button>
          <button onclick="post('/api/control/restart-proxies').then(render)">🔄 Restart Proxies</button>
          <button onclick="window.open('/api/log/download','_blank')">📥 Download Agent Log</button>
          <button onclick="window.open('/api/log/download?tail=200','_blank')">📋 Last 200 Lines</button>
        </div>
        <h2>Agent Status</h2>
        <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(280px,1fr))">${renderAgentStatus(status.agents, stats.totals)}</div>
        <h2>Live Request Log</h2>
        ${renderLogs(logs.logs)}
        <div style="text-align:center;color:#484f58;font-size:11px;margin:20px">Auto-refresh every 4s</div>
      `;
    });
  setTimeout(render, 4000);
}
render();
</script>
</body>
</html>"""

@app.get("/")
def index():
    return HTMLResponse(HTML_PAGE)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8500, log_level="warning")