# Contributing

This is a personal project published for reading and learning. Issues and pull requests are
welcome, but there is no roadmap committee and no guarantee of response time.

## Before you start

Read [`ARCHITECTURE.md`](ARCHITECTURE.md). It is the reference for behaviour; the code is the
source of truth when the two disagree. If you change behaviour, change the document in the same
pull request.

## Setup

```bash
./install.sh          # Python venv, PostgreSQL + pgvector, database, .env, migrations
./start.sh            # REPL
make help             # all targets
```

`install.sh` is idempotent and verifies what it installs: the `vector` extension, the embedding
dimension against the schema, and an import smoke at the end. It also creates the Ed25519
identity (random passphrase kept in `.env`) and, on macOS, the launchd daemons. It does **not**
install a model runtime — start Ollama or vLLM yourself and point `LLM_BASE_URL` at it. If you
already have PostgreSQL, pass `--skip-postgres`.

Tests do not need a live database:

```bash
make test-unit        # fast, no Postgres, no Ollama
make ci               # lint + env-check + typecheck + PR smoke
```

DB-marked tests need a disposable database and skip without one:

```bash
TEST_DATABASE_URL=postgresql://localhost:5432/agent_test make test
```

Never point `TEST_DATABASE_URL` at your working database. The fixture refuses to run when it
matches `DATABASE_URL`.

## Testing with a model

The agent needs a model that supports tool calling and follows a directive prompt. The golden
standard is **`qwen3.8:27b`** (dense 27B).

A model that only narrates its intentions will answer without calling tools. The `system`
block's "act, do not narrate" rule is what prevents that; do not remove it. If you change the
prompt, re-run a write scenario (`save this fact: …`) and confirm the row lands in `memories`.

The CLI is the fastest way to drive the agent from another process — no Telegram, no HTTP:

```bash
agent chat "Save this fact: the test server runs on 8081." --json
agent chat "What do you remember?" -t test_thread
```

`--json` returns `{"reply", "thread_id"}`; the message can come from stdin.

## Pull requests

- One logical concern per pull request. Prefer under 400 changed lines.
- Keep the diff readable: no drive-by reformatting of untouched files.
- For a change to the graph, a tool, or configuration, state in the description: the name, the
  default, why it exists, and how to roll it back.
- Add or update a test in the matching tier: `tests/unit` for pure logic, `tests/integration`
  for graph and multi-module flows, `tests/system` for API, heartbeat, and import smoke.
- Run `make ci` before pushing. `make hooks` installs the pre-push hook.

## Code conventions

- Python 3.12, type hints on public functions.
- Ruff for lint and format, mypy and pyright for types. `make lint` and `make typecheck`.
- No new dependency without a reason in the pull request description.
- Configuration goes through `src/config.py` and `.env.example`. `make env-check` enforces
  parity between the two; a key that exists in one and not the other fails CI.

## Language

The project is English-only. Code, comments, prompts, documentation, fixtures, and test data
are English. Do not add another language anywhere in the tree.

## What will be declined

- A persona, a character name, or a voice contract for the agent.
- A content filter or a refusal layer.
- A cloud dependency where a local one already works.
- A test that reaches the network or a real database without a marker.

## Security

Do not open a public issue for a vulnerability. See [`SECURITY.md`](SECURITY.md).

## License

Contributions are accepted under [PolyForm Shield 1.0.0](LICENSE).
