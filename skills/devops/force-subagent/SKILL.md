---
name: force-subagent
description: Force subagent delegation for infra, network, code, backup.
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

Use the conditions below to decide delegation.

## Delegate if:
1. User says: delegate, ask subagent, let subagent do it
2. Task: code review, code check, security check
3. Task: full server audit (all 5 subagents in parallel)
4. Task needs external API, internet search, package install

## Do NOT delegate if:
- Quick answer he knows himself
- Write/edit a file, email, text
- Read a file, run one terminal command

## How:
  from ask_subagent import ask_codereview
  result = ask_codereview(task)
  print(result)

## Important:
If user says delegate or ask subagent — YOU MUST call. No exception.
