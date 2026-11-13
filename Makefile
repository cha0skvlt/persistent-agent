# Agent Core — developer commands. Run `make` or `make help` for a grouped list.
SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= venv/bin/python3
export PYTHONPATH ?= .
# Tests only — do not export CHECKPOINTER globally (would override .env on make api).
TEST_ENV := CHECKPOINTER=memory

.PHONY: help ops-help host-install install install-dev migrate db-init \
	run api cli chat studio log \
	daemon-install daemon-uninstall daemon-status daemon-restart sleep wake \
	systemd-install systemd-uninstall systemd-status systemd-restart \
	api-install api-uninstall api-status api-tail \
	heart-run heart-install heart-uninstall heart-status heart-tail \
	test test-pr-smoke test-unit test-integration test-system test-live test-cov \
	lint format typecheck env-check ci hooks \
	health selftest selftest-json \
	seed-self forget forget-ids prune-checkpoints \
	repair-memory memory-dedupe kg-backfill dpo-export ollama-tune stabilize-kg \
	debug-memory e2e-memory \
	init-identity show-identity export-identity store-backup-passphrase \
	backup phylactery-backup phylactery-restore phylactery-health emergency-exit

# --- Help ---

help: ## Show grouped targets (default)
	@echo "Agent Core — make targets"
	@if [ -z "$$SHOW_OPS" ]; then echo "(ops-heavy targets hidden; run 'make ops-help')"; fi
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
	  /^##@ / { section=$$0; gsub(/^##@ /,"",section); hidden=(section ~ /Daemon|Seeds|Memory|Identity/); if (!hidden || ENVIRON["SHOW_OPS"]=="1") printf "\n%s\n", section; next } \
	  /^[a-zA-Z0-9_-]+:.*##/ { if (!hidden || ENVIRON["SHOW_OPS"]=="1") printf "  %-24s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

ops-help: ## Show operational/maintenance targets hidden from default help
	@SHOW_OPS=1 $(MAKE) help

##@ Setup

host-install: ## Interactive host setup (venv, DB, models, .env)
	./install.sh

install: ## Runtime deps (requirements.txt)
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Runtime + dev deps (requirements-dev.txt)
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

migrate: ## Alembic upgrade head
	venv/bin/alembic upgrade head

db-init: ## Apply db_schema.sql (fresh DB only; prefer migrate)
	psql "$$DATABASE_URL" -f db_schema.sql

##@ Run

run: ## REPL (./start.sh)
	./start.sh

api: ## FastAPI + Telegram polling (foreground)
	$(PYTHON) -c "import uvicorn; from src.config import settings; uvicorn.run('src.api.app:app', host=settings.api_host, port=settings.api_port)"

cli: ## Typer CLI help
	$(PYTHON) -m src.cli --help

studio: ## LangGraph Studio — open Studio URL in Chrome (no --tunnel)
	$(PYTHON) -m langgraph_cli dev --no-browser

chat: ## One-shot CLI chat (MSG='…')
	@if [ -z "$(MSG)" ]; then echo "Usage: make chat MSG='your message'" >&2; exit 1; fi
	$(PYTHON) -m src.cli chat "$(MSG)"

log: ## Live agent dialogue + graph node timings (tail API log)
	@log="$$HOME/Library/Logs/agent-api.log"; \
	if [[ ! -f "$$log" ]]; then \
		echo "No API log at $$log — start with: make api-install   or   make api"; \
		exit 1; \
	fi; \
	echo "Following $$log (Ctrl-C to stop)"; \
	tail -f "$$log"

##@ Daemon (launchd)

daemon-install: api-install heart-install ## Full resident bundle (API+Telegram, Heart)
	@echo "=== daemon verify ==="
	@brew services list | grep -E "postgresql|ollama" || echo "WARN: check brew postgres/ollama"
	@$(MAKE) selftest || echo "WARN: selftest issues — see above"
	@echo "Daemon installed. Starts at boot, survives reboot."
	@echo "  status: make daemon-status"

daemon-uninstall: api-uninstall heart-uninstall ## Remove full resident bundle
	@echo "Daemon removed. Postgres/Ollama (brew) untouched."

daemon-status: ## launchctl snapshot (api + heart + brew)
	@echo "=== api ===" && launchctl list | grep agent.api || echo "(not loaded)"
	@echo "=== heart ===" && launchctl list | grep agent.heart || echo "(not loaded)"
	@echo "=== postgres/ollama ===" && brew services list | grep -E "postgresql|ollama" || echo "(check brew)"

daemon-restart: ## kickstart api (reinstall plist on failure)
	@launchctl kickstart -k "gui/$$(id -u)/cha0skvlt.persistent-agent.api" || $(MAKE) api-uninstall api-install
	@echo "api restarted"

##@ Daemon (systemd, Linux)

systemd-install: ## Linux: install API + Heart as systemd services
	bash scripts/install-systemd.sh

systemd-uninstall: ## Linux: remove the systemd services
	bash scripts/uninstall-systemd.sh

systemd-status: ## Linux: systemctl status for both services
	@systemctl --no-pager --lines=5 status persistent-agent-api.service persistent-agent-heart.service || true

systemd-restart: ## Linux: restart both services
	@sudo systemctl restart persistent-agent-api.service persistent-agent-heart.service

sleep: ## Stop API+Heart; unload Ollama models (free GPU/RAM)
	bash scripts/sleep.sh

wake: ## Reload API+Heart from existing launchd plists (after sleep)
	@plist="$$HOME/Library/LaunchAgents/cha0skvlt.persistent-agent.api.plist"; \
	if [[ -f "$$plist" ]]; then launchctl load -w "$$plist" && echo "API loaded."; \
	else echo "No API plist — run: make api-install"; fi
	@plist="$$HOME/Library/LaunchAgents/cha0skvlt.persistent-agent.heart.plist"; \
	if [[ -f "$$plist" ]]; then launchctl load -w "$$plist" && echo "Heart loaded."; \
	else echo "No heart plist — run: make heart-install"; fi

api-install: ## launchd API unit (Telegram at boot)
	bash scripts/install-api.sh

api-uninstall: ## Remove launchd API unit
	bash scripts/uninstall-api.sh

api-status: ## launchd + last log lines
	@echo "--- launchd status ---"
	@launchctl list | grep agent.api || echo "(not loaded)"
	@echo "--- last log lines ---"
	@tail -n 5 "$$HOME/Library/Logs/agent-api.log" 2>/dev/null || echo "(log not found)"

api-tail: ## follow raw ~/Library/Logs/agent-api.log
	tail -f "$$HOME/Library/Logs/agent-api.log"

heart-run: ## Heartbeat foreground (vitals loop)
	$(PYTHON) -m src.heart.heartbeat

heart-install: ## launchd Heart unit
	bash scripts/install-heart.sh

heart-uninstall: ## Remove launchd Heart unit
	bash scripts/uninstall-heart.sh

heart-status: ## launchd + last vitals tick
	@echo "--- launchd status ---"
	@launchctl list | grep agent.heart || echo "(not loaded)"
	@echo "--- last vitals tick ---"
	@$(PYTHON) -c "\
from src.config import settings; import psycopg;\
url = settings.database_url.replace('postgresql+psycopg://','postgresql://');\
conn = psycopg.connect(url);\
row = conn.execute('SELECT tick_at, phylactery_status, heart_pid, body_metrics FROM vitals ORDER BY tick_at DESC LIMIT 1').fetchone();\
conn.close();\
print(row) if row else print('(no ticks yet)')" 2>/dev/null || echo "(DB unavailable)"
	@echo "--- last 5 log lines ---"
	@tail -n 5 "$$HOME/Library/Logs/agent-heart.log" 2>/dev/null || echo "(log not found)"

heart-tail: ## follow raw ~/Library/Logs/agent-heart.log
	tail -f "$$HOME/Library/Logs/agent-heart.log"

##@ Tests & quality

test-pr-smoke: ## PR gate: unit + import/graph smoke
	$(TEST_ENV) $(PYTHON) -m pytest -q tests/unit \
	  tests/system/test_main_import.py \
	  tests/system/test_smoke.py \
	  tests/integration/test_workflow_import.py

test: ## Unit + integration + system (non-live)
	$(TEST_ENV) $(PYTHON) -m pytest -q tests/unit tests/integration tests/system -m "not live"

test-unit: ## tests/unit
	$(TEST_ENV) $(PYTHON) -m pytest -q tests/unit

test-integration: ## tests/integration
	$(TEST_ENV) $(PYTHON) -m pytest -q tests/integration

test-system: ## tests/system (non-live)
	$(TEST_ENV) $(PYTHON) -m pytest -q tests/system -m "not live"

test-live: ## @pytest.mark.live (needs Ollama)
	$(TEST_ENV) $(PYTHON) -m pytest -q tests/system -m live

test-cov: ## Coverage: unit + integration + system (non-live)
	$(TEST_ENV) $(PYTHON) -m pytest -q tests/unit tests/integration tests/system -m "not live" \
	  --cov=src --cov-report=term-missing

lint: ## ruff + black + isort (check)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .
	$(PYTHON) -m isort --check --profile black .

format: ## ruff --fix + black + isort
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .
	$(PYTHON) -m isort --profile black .

typecheck: ## mypy + pyright (src/)
	$(PYTHON) -m mypy src
	PYRIGHT_PYTHON_GLOBAL_NODE=0 $(PYTHON) -m pyright src

env-check: ## Settings fields vs .env.example
	@$(PYTHON) scripts/check_env_sync.py

ci: lint env-check typecheck test-pr-smoke ## PR gate (matches .github/workflows/ci.yml)

hooks: ## Install pre-commit + pre-push (make ci)
	venv/bin/pre-commit install
	venv/bin/pre-commit install --hook-type pre-push
	$(MAKE) format
	venv/bin/pre-commit run -a

##@ Health

health: ## PG + Ollama embed probe (scripts/health.py)
	@$(PYTHON) scripts/health.py

selftest: ## Read-only stack check (ARCH-selftest)
	$(PYTHON) -m src.selftest

selftest-json: ## Selftest JSON report
	$(PYTHON) -m src.selftest --json

backup: phylactery-backup ## Encrypted pg_dump of agent DB (Keychain: make store-backup-passphrase first)

##@ Seeds

seed-self: ## Seed baseline self-reflection row
	$(PYTHON) scripts/seed_self.py

forget: ## Forget script (session / memory utility)
	$(PYTHON) scripts/forget.py

forget-ids: ## Archive+delete by id; APPLY=1 to execute
	@if [ -z "$(IDS)" ]; then echo "Usage: make forget-ids IDS='uuid1,uuid2' [APPLY=1]" >&2; exit 1; fi
	$(PYTHON) scripts/forget.py --ids "$(IDS)" $(if $(APPLY),--apply,)

prune-checkpoints: ## Prune old LangGraph checkpoints + aged dpo_gold_pairs
	$(PYTHON) scripts/prune_checkpoints.py $(if $(DRY),--dry-run,)

##@ Memory & KG

repair-memory: ## Collision audit (dry-run); APPLY=1 to fix
	$(PYTHON) scripts/repair_memory_collisions.py $(if $(APPLY),--apply,)

memory-dedupe: ## Near-duplicate scan; APPLY=1 to merge
	$(PYTHON) scripts/dedupe_memories.py $(if $(APPLY),--apply,)

kg-backfill: ## Backfill KG from memories; APPLY=1 to write
	$(PYTHON) scripts/backfill_kg_from_memories.py $(if $(APPLY),--apply,)

dpo-export: ## Export DPO gold pairs JSONL (DAYS=7, OUT=path required)
	@test -n "$(OUT)" || (echo "OUT=path required (e.g. OUT=/tmp/dpo_gold.jsonl)"; exit 1)
	$(PYTHON) scripts/export_dpo_dataset.py --days $(or $(DAYS),7) -o $(OUT)

ollama-tune: ## Ollama parallel=2 + max_loaded_models=2
	chmod +x scripts/ollama_tune_parallel.sh
	./scripts/ollama_tune_parallel.sh

stabilize-kg: ## Normalize legacy agent KG aliases (--apply)
	$(PYTHON) scripts/stabilize_kg_agent.py --apply

debug-memory: ## Memory debug REPL; LIVE=1 for live search
	$(if $(LIVE),LOG_LEVEL=DEBUG $(PYTHON) scripts/debug_memory.py --live,LOG_LEVEL=DEBUG $(PYTHON) scripts/debug_memory.py)

e2e-memory: ## E2E memory session script
	$(PYTHON) scripts/e2e_memory_sessions.py

##@ Identity & phylactery

init-identity: ## Generate Ed25519 agent key
	$(PYTHON) -m src.identity.cli init

show-identity: ## Print agent public key
	$(PYTHON) -m src.identity.cli show

export-identity: ## FILE=/path/to/identity.json
	@if [ -z "$(FILE)" ]; then echo "Usage: make export-identity FILE=/path/to/identity.json" >&2; exit 1; fi
	$(PYTHON) -m src.identity.cli export "$(FILE)"

store-backup-passphrase: ## Store backup encryption passphrase in Keychain (once)
	$(PYTHON) -m src.identity.cli store-backup-passphrase

phylactery-backup: ## Encrypted pg_dump → ~/Backups/agent (PHYLACTERY_BACKUP_DIR)
	$(PYTHON) scripts/phylactery_backup.py

phylactery-restore: ## Decrypt+restore; FILE=/path/to/agent-….dump.enc
	@if [ -z "$(FILE)" ]; then echo "Usage: make phylactery-restore FILE=/path/to/backup.dump.enc" >&2; exit 1; fi
	$(PYTHON) scripts/phylactery_restore.py "$(FILE)"

phylactery-health: ## Latest vitals + phylactery counts
	$(PYTHON) scripts/phylactery_health.py

emergency-exit: ## Manual emergency vitals tick
	$(PYTHON) -m src.heart.emergency_exit
