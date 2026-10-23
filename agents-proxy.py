#!/usr/bin/env python3
"""Multi-port logging proxy for Hermes sub-agents — generates per-port proxy scripts from a reference template."""
import json, sqlite3, os, subprocess, sys, signal, time
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM = "http://127.0.0.1:8083"
API_KEY = "strata-local"
DB_DIR = Path.home() / "token-stats" / "data"
DB_PATH = DB_DIR / "tokens.db"

AGENTS = {
    8090: "devops",
    8091: "network",
    8092: "backup",
    8093: "fs",
    8094: "codereview",
}

# Use the already-patched 8091-proxy.py as authoritative reference
REF_PORT = 8091

if __name__ == "__main__":
    scripts_dir = Path.home() / "token-stats" / "agent-scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    ref_path = scripts_dir / f"{REF_PORT}-proxy.py"
    if not ref_path.exists():
        print(f"[agents-proxy] ERROR: reference {ref_path} not found")
        sys.exit(1)
    ref_content = ref_path.read_text()
    for port, agent_type in AGENTS.items():
        script_path = scripts_dir / f"{port}-proxy.py"
        # Replace port number and type placeholder
        content = ref_content \
            .replace(str(REF_PORT), str(port)) \
            .replace(f'"agent-proxy@{REF_PORT}', f'"agent-proxy@{port}') \
            .replace(AGENTS[REF_PORT], agent_type)
        script_path.write_text(content)
        script_path.chmod(0o755)
        print(f"[agents-proxy] written: {script_path.name}")