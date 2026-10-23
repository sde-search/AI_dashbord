---
name: force-subagent
description: "Force subagent delegation for infra, network, code, backup."
version: 1.0.0
author: Gnoll, Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [subagent, delegation, force]
    related_skills: []
---

# Force Subagent Delegation

Use when: main agent MUST delegate to subagent proxies (8090-8094) and NOT handle tasks directly.

## Strict Rules

1. ALWAYS call subagent via execute_code. NEVER do the task yourself.
2. NEVER report success without calling. If you write "subagent checked" but didn't call — task is FAILED.
3. Exact call pattern:
   from ask_subagent import ask_codereview
   result = ask_codereview("task description")
   print(result)

## Port mapping

| Task type | Port | Function |
|---|---|---|
| DevOps/services/systemd | 8090 | ask_devops |
| Network/ports/connectivity | 8091 | ask_network |
| Backup policies/scripts | 8092 | ask_backup |
| FS/disk/files | 8093 | ask_fs |
| Code review/security | 8094 | ask_codereview |

## Enforcement

- If ask_subagent import fails: curl to port directly
- If subagent returns error: tell user, no fabricating
- If subagent returns empty: still log, no making up
