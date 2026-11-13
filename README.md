# persistent-agent

<p align="center">
  <em>A single-operator agent with a body on disk.</em>
</p>

<p align="center">
  <img alt="version" src="https://img.shields.io/badge/1.0.0-persistent--agent-111111?style=for-the-badge">
  <img alt="python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="license" src="https://img.shields.io/badge/license-PolyForm--Shield-C41E3A?style=for-the-badge">
</p>

<p align="center">
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-bare_metal-00C7B7?logo=langchain&logoColor=white">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-local_LLM-000000">
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-OpenAI--compatible-FF6F00">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-uvicorn-009688?logo=fastapi&logoColor=white">
  <img alt="Telegram" src="https://img.shields.io/badge/Telegram-bot-26A5E4?logo=telegram&logoColor=white">
  <img alt="SearXNG" src="https://img.shields.io/badge/SearXNG-loopback-5B5B5B">
  <img alt="Playwright" src="https://img.shields.io/badge/Playwright-Chromium-2EAD33?logo=playwright&logoColor=white">
</p>

**persistent-agent** is an R&D project: a working lab for building an agent with persistent
vector memory. The question it chases is narrow and stubborn — what does an agent need so that
it still knows you after the process dies, the model is swapped, and the context window is
gone? The findings from this repository are applied in
[**CHA0S_AOE**](https://github.com/cha0skvlt/CHA0S_AOE), the agent environment built on top of
them.

> **What has been demonstrated.** An agent was grown inside `persistent-agent` on local Qwen
> large local models over three months, accumulating a large body of world knowledge and
> a long record of reflections about itself. Its phylactery — the database — was then moved
> into [**CHA0S_AOE**](https://github.com/cha0skvlt/CHA0S_AOE) and integrated there as a
> first-class agent. On its first launch in the new product it read its identity from the
> phylactery and came up with full access to its memory.
>
> **Scope of the claim.** What is confirmed is the transfer of the database between two
> products: the same rows, the same identity anchor, the same memory, read by a different
> process. This is not a claim that every deployment is portable, nor that the agent behaves
> identically under a different model — only that the state survives the move.
>
> The architectural point stands: **the model is swappable brains, not the agent.** The weights
> can change; the personality and the memory live in the phylactery and survive the swap.

Concretely, it is a personal assistant for one operator. Not a shared bot, not a
multi-tenant service: one person, one agent, one database. PostgreSQL holds world facts, a
first-person diary, causal lessons, a knowledge graph, the dialogue checkpoint, and signed key
material. You can kill the API process. You can swap the model behind the endpoint. The
database stays. This is not a chat that forgets past its context window.

Wording comes from the model weights and
[`prompts/agent/system_prompt.yml`](prompts/agent/system_prompt.yml). The code stores, retrieves,
binds tools, stops stalls, and delivers the reply. One operator message — one graph pass — one
visible answer. Deferred work (session summary, memory extraction, diary) runs after the reply,
not instead of it.

You can write from Telegram, the CLI, the REPL, or HTTP `POST /chat`. Every live turn funnels
through one runner. Telegram is the main channel: long-polling inside the API, a closed chat
allowlist, and `/make sleep` / `/make wake`. While asleep, Telegram turns and the background
mind stay quiet and Heart holds outgoing messages. REST, CLI, and REPL ignore sleep. Heart is a
separate process: signed vitals, encrypted database backup, and delivery of the outgoing queue.

The prompt carries no persona and no hardcoded name — identity lives in the phylactery, not in
the system prompt. If the phylactery is empty, the name has not been given yet. The prompt is
English-only.

---

## A live turn

On every operator message the agent first reads the clock and plans retrieval: is the web
needed, is this about the world. Then it pulls semantic memory from Postgres, plus diary lines
and lessons when relevant. If the operator asks for a web lookup, the page or the SearXNG result
set arrives in the same turn alongside personal memory, not instead of it.

The model then calls tools itself: save a fact, correct a row by id, forget, search by time,
open a URL, page through a site in Chromium, schedule a reminder, record a lesson.
Twenty-three tools. The graph does not dictate what to say. The brakes are a wall clock, a repeated tool
batch, and a round cap.

After the reply, reflect may extract a world fact into `memories` and a short diary entry. A
fact with no grounding in the operator's words or in this turn's tool output does not reach the
database. A `USER:` prefix is not taken from a web page alone.

---

## Background loops

While the operator is silent, the API runs reverie. It first claims a due Pulse intent. If none
is ready and solitary is enabled (`SOLITARY_ENABLED`, off by default), it runs a solitary pass.
The default interval is five minutes since the last pass. A live turn always wins: background
work is cancelled at the LLM, and solitary does not start while the operator has just spoken.

Solitary is the same full graph, not a smaller model. The prompt differs. Automatic `[MEM]`
injection is off on background passes: the agent calls `search_memory`, `search_web`,
`save_memory` itself. Mind Hub — the recent-events timeline — stays, so the pass does not lie
about what just happened.

Each solitary pass gets a lane. Lanes rotate.

**Discover (P1).** An external topic: world, place, craft, science, culture, tools. `search_web`
first. If a page or thread is worth it, `fetch_url` or `browse_web`. It reads the full text from
tool output. A new fact is written with `save_memory` in the same pass, not "some day". Three
flavours of the lane: broad world, deep page, practice and technique.

**Hygiene (P2).** Database sanitising. `search_memory` and `search_by_time`, take ids,
`forget_memory` or `memory_replace` on the false and the stale, `save_memory` with the correct
slice. This is not a "who am I" loop. When contradictions pile up, hygiene debt (`hygiene_debt`)
forces the pass onto this lane until the agent actually fixes a row. Search without mutation
does not clear the debt. Once the debt is paid, retrieve trust drifts back toward its baseline.

**Intents (P3).** Rhythm. `list_intents` is mandatory in the pass. Set or cancel something
near-term: minutes, hours, today, at most tomorrow. Solitary does not plan a distant horizon.

Background writes are short: a fact, a lesson, a hypothesis with a confidence. An essay does not
pass into memory. The write must overlap with what a tool already returned in this pass. An
invented `USER:` line with no such line in the tool output is rejected. Diary entries from
solitary are short too. Solitary never writes to Telegram on its own: only an explicit
`message_user` call. If a live turn interrupts a pass, the next tick resumes the same thread
instead of starting over.

---

## Memory

The database is not a log dump. Retrieve on a live turn takes cosine similarity to the query,
drops the weak, then applies one distrust multiplier: unverified rows and old facts are demoted.
Half-life and unverified weight live in `policy_knobs`; the agent's own history moves them, not
the prompt.

On write, `save_memory` looks for a near conflict. The default policy logs the pile
(`log_pile`): the old row's importance is cut, hygiene debt rises, and the next solitary pass
must clean up. Alternatives are replace, mark unverified, or skip. Diary writes do not feed that
pile. A repeated thought is caught by cosine plus fuzzy text dedup. Lessons dedup on the
`change` / `because` pair.

After the reply, reflect may not write from the model's own answer: only overlap with the
operator's words and this turn's tools counts. On background passes a second LLM writes no world
facts at all — only tools plus the filter. The deferred diary does not store nearly the same
sentence twice.

Lessons ("stopped X because Y") raise tool curiosity on later solitary passes. Initiative to
message the operator is a separate number: too low and `message_user` refuses. Attention to the
live channel (`attention_reserve`) shortens and postpones solitary when the operator was just in
the chat.

Operator scripts (`make repair-memory`, dedup, checkpoint pruning) run by hand. The hot path
never uses them.

All of it lives in one PostgreSQL database — the **phylactery**. It holds the identity anchor
and the memory together, so a restored dump brings the agent back with the same memory and the
same identity, on another machine or under another model. The process is disposable; the
phylactery is not.

---

## Scheduling and outgoing messages

The operator, or the agent itself, sets an intent in a human phrase: in two hours, tomorrow
evening, every Monday, every N minutes (no shorter than half an hour). There is no cron. Heart
marks overdue rows. Reverie claims one ready row and runs the same graph in intent mode.
Recurring intents move to the next `run_at`. One-shots go `done`. A failure backs off
exponentially; after five attempts it goes `failed` and the owner gets a Telegram notice. An
intent turn does not inject foreign memory into SYS.

The agent can only speak first from the background: `message_user` puts a draft into `outbox`,
and Heart sends it to Telegram in order. On a live turn the tool is hidden — the reply is already
going to the channel. The queue is capped. Disabling outreach blocks both enqueue and delivery.

Mind Hub records one timeline: operator input, agent output, solitary pass, outreach queued and
sent, intent closed. That timeline is visible in the system context of later turns.

---

## Web, keys, hardware

Web is a local SearXNG, then fallback scrapes, then the page through `fetch_url` or a live
Chromium. Private addresses are blocked. Search needs no cloud keys.

Signing is Ed25519: response footers when `SIGN_RESPONSES=1`, and Heart vitals. Keys live in the
`agent_identity` table. `install.sh` creates them (`make init-identity` is the manual path).

The stack runs bare-metal on purpose. No Docker, no containers, no orchestration layer: the
agent talks to the GPU, the disk, and the local network directly. That is what makes a large
model on a single machine practical, and what lets the phylactery stay a plain PostgreSQL
database you can copy, inspect, and move.

Stack on the host: Python 3.12, LangGraph, PostgreSQL + pgvector, FastAPI, SearXNG on loopback,
Playwright. The model is not part of the stack — it is an endpoint, local or remote. Behaviour
is defined by the code. The reference is [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Providers

**The agent does not install or manage a model runtime.** It owns the database and the
process; the model is an external endpoint you start yourself — Ollama, vLLM, OpenAI, or any
OpenAI-compatible server. `install.sh` never installs Ollama and never pulls a model.

Chat and embeddings are independent channels. Both default to local Ollama; either can point
at any OpenAI-compatible server.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `LLM_BASE_URL` | empty | Chat endpoint. Empty = `OLLAMA_BASE_URL`; required for `openai` |
| `LLM_API_KEY` | empty | Bearer token |
| `LLM_EXTRA_BODY` | empty | JSON merged into the request body (vLLM: `{"top_k": 40, "min_p": 0.0}`) |
| `EMBED_PROVIDER` | `ollama` | `ollama` or `openai` |
| `EMBED_BASE_URL` | empty | Falls back to `LLM_BASE_URL`, then `OLLAMA_BASE_URL` |
| `EMBED_API_KEY` | empty | Falls back to `LLM_API_KEY` |
| `EMBED_DIM` | `768` | Must match the `vector(N)` columns |

Cloud chat, local embeddings:

```bash
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
REASONING_MODEL=gpt-4o-mini
```

vLLM on a weak machine:

```bash
LLM_PROVIDER=openai
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_API_KEY=not-needed
REASONING_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_EXTRA_BODY={"top_k": 40, "min_p": 0.0}
```

`EMBED_DIM` is not cosmetic: the schema stores `vector(768)`. Switching to a 1536-d model is a
schema migration plus a full reindex, not just a config change. A 768-d embedding model on vLLM
(`BAAI/bge-base-en-v1.5`) avoids that entirely.

`install.sh` takes the same choices as flags: `--provider`, `--base-url`, `--api-key`,
`--model`, `--embed-provider`, `--embed-base-url`, `--embed-api-key`, `--embed-model`,
`--embed-dim`. It writes them to `.env` and probes both endpoints; it installs nothing for them.

Models are not distributed with this repository. Each one carries its own license, set by its
publisher, not by this project.

### Model requirements

The agent needs a model that supports tool calling and follows a directive prompt. The golden
standard is **`qwen3.8:27b`** — a dense 27B model that fills tool arguments correctly.

A model that only narrates its intentions will answer without calling tools. The `system`
block carries the rule that prevents this — do not remove it. The same applies to the evidence
seeding in `agent_node`: without it the live `save_memory` provenance gate rejects every save.

---

## Launch

```bash
# 1. Start the model server yourself (pick one)
ollama serve
# or
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 2. Install the agent
./install.sh

# 3. Run
./start.sh
```

`./install.sh` installs what the agent owns:

1. Python 3.12 venv and `requirements.txt`
2. PostgreSQL + the `vector` extension (`CREATE EXTENSION` is verified, not assumed)
3. A database and a role with a generated password, written to `.env`
4. Provider settings in `.env`, then a reachability probe of the chat and embedding endpoints
5. `alembic upgrade head`, then an import smoke and an env-parity check
6. An Ed25519 identity — a random `IDENTITY_PASSPHRASE` is generated and kept in `.env` (mode
   600) for recovery, while Keychain holds the working copy
7. A service manager so it survives reboot: launchd on macOS, systemd on Linux
   (`--no-daemon` opts out; `make systemd-install` / `make daemon-install` run it alone)

It does **not** install Ollama, vLLM, or any model. Point it at an endpoint that is already
running:

```bash
./install.sh --provider openai --base-url http://127.0.0.1:8000/v1 \
             --model Qwen/Qwen2.5-7B-Instruct
```

It is idempotent and works on macOS (Homebrew), Debian, Fedora, and Arch. Non-interactive:
`./install.sh --yes`. Skip PostgreSQL with `--skip-postgres`. Other flags: `--api-key`,
`--embed-provider`, `--embed-base-url`, `--embed-api-key`, `--embed-model`, `--embed-dim`,
`--dev`, `--no-daemon`.

### Cloud Debian + vLLM

A fresh Debian host with vLLM already serving a model needs one command:

```bash
./install.sh --yes \
  --provider openai --base-url http://127.0.0.1:8000/v1 --api-key not-needed \
  --model Qwen/Qwen2.5-7B-Instruct \
  --embed-model BAAI/bge-base-en-v1.5 --embed-dim 768
```

What it does on that host: `apt-get` for Python, PostgreSQL and pgvector (package, or built
from source when Debian main has no `pgvector`), a role and database, `alembic upgrade head`,
the Ed25519 identity, and `systemd/persistent-agent-{api,heart}.service` installed and enabled.
`make systemd-status` shows both units; `make systemd-restart` restarts them.

`./start.sh` is the REPL. `make daemon-restart` starts API, Telegram, and Heart. Solitary:
`SOLITARY_ENABLED=1`. Other targets: `make help`.

---

## CLI for another agent

The CLI is the machine-facing surface: one turn in, one reply out, JSON on demand. Another
agent can drive it without a browser, a Telegram account, or the HTTP API.

```bash
# one turn, plain reply
agent chat "What do you remember about the deploy?"

# JSON for a caller that parses
agent chat "Save this fact: the test server runs on 8081." --json
# {"reply": "Saved.", "thread_id": "cli_session_v1"}

# pipe-friendly: message from stdin
echo "What is the deploy status?" | agent chat --json

# a stable thread across calls
agent chat "Remember: the deploy is on Friday." -t deploy_check
agent chat "When is the deploy?" -t deploy_check

# status without touching the graph
agent status --json
```

`make cli` prints the help. The CLI runs the same `run_agent_turn_detailed` as the REPL, the
API, and Telegram — same graph, same memory, same tools. It is the recommended way to test the
agent from another process: no channel setup, no polling, and `--json` gives a stable contract.

---

## License

[PolyForm Shield 1.0.0](LICENSE) — read, learn, fork, build on it. The one restriction is
competition: you may not use it to provide a product that competes with this software or with
the licensor's products built on it.

Required Notice: Copyright 2026 Eugene Grossbart / piniq.ai (https://piniq.ai)

Licensor Line of Business: AI agent software with persistent vector memory, and agent
environments built on it (https://piniq.ai)
