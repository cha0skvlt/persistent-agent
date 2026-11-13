#!/bin/bash
set -euo pipefail

PY="python3"
PIP="python3 -m pip"
if [ -x "venv/bin/python3" ]; then
  PY="venv/bin/python3"
  PIP="venv/bin/python3 -m pip"
elif [ -f "venv/bin/activate" ]; then
  # fallback (older venv layouts)
  source venv/bin/activate
fi

_agent_log_info() {
  local msg="$1"
  local line
  line="$(date '+%H:%M:%S') | INFO | ${msg}"
  if [ -t 2 ]; then
    printf '\033[90m\033[2m%s\033[0m\n' "${line}" >&2
  else
    printf '%s\n' "${line}" >&2
  fi
}

_agent_log_warning() {
  local msg="$1"
  local line
  line="$(date '+%H:%M:%S') | WARNING | ${msg}"
  if [ -t 2 ]; then
    printf '\033[90m%s\033[0m\n' "${line}" >&2
  else
    printf '%s\n' "${line}" >&2
  fi
}

# Defaults from src.config.Settings (single source of truth; avoids duplicating literals here).
_agent_runtime_env() {
  PYTHONPATH="." "$PY" <<'PY' 2>/dev/null || true
from src.config import settings as s

print(s.database_host)
print(s.database_port)
print(s.ollama_base_url.rstrip("/"))
print(s.embed_model)
print(s.postgres_service)
print(s.reasoning_model)
PY
}

INSTALL_DEPS="${INSTALL_DEPS:-0}"
if [ "${INSTALL_DEPS}" = "1" ]; then
  ${PIP} install -r requirements.txt
fi

_RT="$(_agent_runtime_env)"
AGENT_DB_HOST_DEFAULT="$(printf '%s\n' "${_RT}" | sed -n '1p')"
AGENT_DB_PORT_DEFAULT="$(printf '%s\n' "${_RT}" | sed -n '2p')"
OLLAMA_BASE_URL_DEFAULT="$(printf '%s\n' "${_RT}" | sed -n '3p')"
EMBED_MODEL_DEFAULT="$(printf '%s\n' "${_RT}" | sed -n '4p')"
POSTGRES_SERVICE_DEFAULT="$(printf '%s\n' "${_RT}" | sed -n '5p')"
REASONING_MODEL_DEFAULT="$(printf '%s\n' "${_RT}" | sed -n '6p')"

AGENT_DB_HOST="${AGENT_DB_HOST:-${AGENT_DB_HOST_DEFAULT}}"
AGENT_DB_PORT="${AGENT_DB_PORT:-${AGENT_DB_PORT_DEFAULT}}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-${OLLAMA_BASE_URL_DEFAULT}}"
EMBED_MODEL="${EMBED_MODEL:-${EMBED_MODEL_DEFAULT}}"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-${POSTGRES_SERVICE_DEFAULT}}"
REASONING_MODEL="${REASONING_MODEL:-${REASONING_MODEL_DEFAULT}}"

# PostgreSQL + Ollama: start via Homebrew when idle (no redundant restart).
if command -v brew >/dev/null 2>&1; then
  postgres_ok=false
  if [ -n "${AGENT_DB_HOST}" ] && [ -n "${AGENT_DB_PORT}" ]; then
    if command -v pg_isready >/dev/null 2>&1 && pg_isready -h "${AGENT_DB_HOST}" -p "${AGENT_DB_PORT}" >/dev/null 2>&1; then
      postgres_ok=true
    elif command -v nc >/dev/null 2>&1 && nc -z "${AGENT_DB_HOST}" "${AGENT_DB_PORT}" >/dev/null 2>&1; then
      postgres_ok=true
    fi
  fi
  if [ "${postgres_ok}" != true ]; then
    if [ -n "${POSTGRES_SERVICE}" ]; then
      brew services start "${POSTGRES_SERVICE}"
    else
      echo "POSTGRES_SERVICE unset and Postgres not reachable — start PostgreSQL manually." >&2
    fi
  fi

  if [ -n "${OLLAMA_BASE_URL}" ] && ! curl -sf --max-time 2 "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
    brew services start ollama
  fi
else
  echo "brew not found in PATH — start PostgreSQL and Ollama manually." >&2
fi

# Wait for Ollama HTTP API (~30s) and pull missing models (embed + LLM).
if command -v ollama >/dev/null 2>&1 && [ -n "${OLLAMA_BASE_URL}" ]; then
  for _ in $(seq 1 30); do
    if curl -sf --max-time 2 "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if curl -sf --max-time 2 "${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
    if [ -n "${EMBED_MODEL}" ] \
      && ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${EMBED_MODEL}" \
      && ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${EMBED_MODEL}:latest"; then
      _agent_log_info "Ollama: pulling embed model ${EMBED_MODEL} (required for vector memory)..."
      ollama pull "${EMBED_MODEL}" || _agent_log_warning "Ollama: pull failed for ${EMBED_MODEL} — vector memory disabled."
    fi
    if [ -n "${REASONING_MODEL}" ] \
      && ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${REASONING_MODEL}" \
      && ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${REASONING_MODEL}:latest"; then
      _agent_log_info "Ollama: pulling chat model ${REASONING_MODEL}..."
      ollama pull "${REASONING_MODEL}" || _agent_log_warning "Ollama: pull failed for ${REASONING_MODEL} — set REASONING_MODEL in .env or run ollama pull manually."
    fi
  else
    _agent_log_warning "Ollama not responding at ${OLLAMA_BASE_URL} — vector memory disabled."
  fi
fi

# Run LangGraph agent from repo root (PYTHONPATH=.); `src` is a package from root.
PYTHONPATH="." ${PY} -m src.main
