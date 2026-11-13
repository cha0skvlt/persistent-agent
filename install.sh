#!/usr/bin/env bash
# persistent-agent installer.
#
# Installs what the agent owns: a Python venv, PostgreSQL + pgvector, the database,
# and a .env that points at a model endpoint. It does NOT install or manage a model
# runtime — no Ollama, no vLLM, no model downloads. Start those yourself and point
# LLM_BASE_URL / EMBED_BASE_URL at them.
#
# Idempotent. Never prints secrets. Never overwrites an existing .env wholesale.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"

YES=0
DO_DAEMON=""
DO_DEV=0
DO_IDENTITY=""
SKIP_POSTGRES=0
PROVIDER_FLAG=""
BASE_URL_FLAG=""
API_KEY_FLAG=""
MODEL_FLAG=""
EMBED_PROVIDER_FLAG=""
EMBED_BASE_URL_FLAG=""
EMBED_API_KEY_FLAG=""
EMBED_MODEL_FLAG=""
EMBED_DIM_FLAG=""

VENV="$ROOT/venv"
PY="$VENV/bin/python3"
ENV_FILE="$ROOT/.env"
EXAMPLE="$ROOT/.env.example"

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Installs Python venv, PostgreSQL + pgvector, the database, and .env.
Does NOT install Ollama, vLLM, or any model. Start the model server yourself.

  --yes, -y           Non-interactive: defaults, no prompts
  --provider P        Chat provider: ollama | openai (vLLM, OpenAI, any compatible)
  --base-url URL      Chat base URL (required for --provider openai)
  --api-key KEY       Chat API key (or set LLM_API_KEY in the environment)
  --model NAME        Served model name (Ollama tag or vLLM/OpenAI model id)
  --embed-provider P  Embedding provider: ollama | openai
  --embed-base-url U  Embedding base URL (default: chat base URL)
  --embed-api-key K   Embedding API key (default: chat key)
  --embed-model NAME  Embedding model name
  --embed-dim N       Embedding dimension (default 768; must match vector(N))
  --skip-postgres     Do not install/start PostgreSQL (must already be reachable)
  --daemon            Install launchd API+Heart (default on macOS)
  --no-daemon         Skip launchd
  --dev               Also install requirements-dev.txt
  --identity          Deprecated: identity is created automatically
  -h, --help          This help

Exit codes: 0 ok, 1 fatal, 2 bad usage.
EOF
}

_parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -y|--yes) YES=1 ;;
      --provider)
        [[ -n "${2:-}" && "$2" != -* ]] || { echo "ERROR: --provider needs ollama|openai" >&2; exit 2; }
        PROVIDER_FLAG="$2"; shift ;;
      --base-url)
        [[ -n "${2:-}" && "$2" != -* ]] || { echo "ERROR: --base-url needs a URL" >&2; exit 2; }
        BASE_URL_FLAG="$2"; shift ;;
      --api-key)
        [[ -n "${2:-}" && "$2" != -* ]] || { echo "ERROR: --api-key needs a value" >&2; exit 2; }
        API_KEY_FLAG="$2"; shift ;;
      --model)
        [[ -n "${2:-}" && "$2" != -* ]] || { echo "ERROR: --model needs a name" >&2; exit 2; }
        MODEL_FLAG="$2"; shift ;;
      --embed-provider)
        [[ -n "${2:-}" && "$2" != -* ]] || { echo "ERROR: --embed-provider needs ollama|openai" >&2; exit 2; }
        EMBED_PROVIDER_FLAG="$2"; shift ;;
      --embed-base-url)
        [[ -n "${2:-}" && "$2" != -* ]] || { echo "ERROR: --embed-base-url needs a URL" >&2; exit 2; }
        EMBED_BASE_URL_FLAG="$2"; shift ;;
      --embed-api-key)
        [[ -n "${2:-}" && "$2" != -* ]] || { echo "ERROR: --embed-api-key needs a value" >&2; exit 2; }
        EMBED_API_KEY_FLAG="$2"; shift ;;
      --embed-model)
        [[ -n "${2:-}" && "$2" != -* ]] || { echo "ERROR: --embed-model needs a name" >&2; exit 2; }
        EMBED_MODEL_FLAG="$2"; shift ;;
      --embed-dim)
        [[ -n "${2:-}" && "$2" != -* ]] || { echo "ERROR: --embed-dim needs a number" >&2; exit 2; }
        EMBED_DIM_FLAG="$2"; shift ;;
      --skip-postgres) SKIP_POSTGRES=1 ;;
      --daemon) DO_DAEMON=1 ;;
      --no-daemon) DO_DAEMON=0 ;;
      --dev) DO_DEV=1 ;;
      --identity) DO_IDENTITY=1 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
  done
}

# --- log --------------------------------------------------------------------

_info() {
  local line; line="$(date '+%H:%M:%S') | INFO | $1"
  if [[ -t 2 ]]; then printf '\033[90m\033[2m%s\033[0m\n' "$line" >&2
  else printf '%s\n' "$line" >&2; fi
}
_warn() {
  local line; line="$(date '+%H:%M:%S') | WARNING | $1"
  if [[ -t 2 ]]; then printf '\033[33m%s\033[0m\n' "$line" >&2
  else printf '%s\n' "$line" >&2; fi
}
_die() { echo "$(date '+%H:%M:%S') | ERROR | $1" >&2; exit 1; }
_step() { echo ""; _info "== $1"; }

_ask() {
  local prompt="$1" def="${2:-}" reply=""
  if [[ "$YES" -eq 1 ]]; then printf '%s\n' "$def"; return; fi
  if [[ -n "$def" ]]; then
    read -r -p "$prompt [$def]: " reply || true
    printf '%s\n' "${reply:-$def}"
  else
    read -r -p "$prompt: " reply || true
    printf '%s\n' "$reply"
  fi
}

_ask_yn() {
  local prompt="$1" def="${2:-n}" reply=""
  if [[ "$YES" -eq 1 ]]; then [[ "$def" == "y" ]] && return 0 || return 1; fi
  read -r -p "$prompt [$([[ "$def" == "y" ]] && echo Y/n || echo y/N)]: " reply || true
  reply="$(printf '%s' "${reply:-$def}" | tr '[:upper:]' '[:lower:]')"
  [[ "$reply" == "y" || "$reply" == "yes" ]]
}

# --- OS / packages ----------------------------------------------------------

_os() {
  case "$(uname -s)" in
    Darwin) echo macos; return ;;
    Linux) ;;
    *) echo unknown; return ;;
  esac
  if [[ -f /etc/debian_version ]]; then echo debian
  elif [[ -f /etc/fedora-release ]]; then echo fedora
  elif [[ -f /etc/arch-release ]]; then echo arch
  else echo linux; fi
}

_brew_bin() {
  if [[ -x /opt/homebrew/bin/brew ]]; then echo /opt/homebrew/bin/brew
  elif [[ -x /usr/local/bin/brew ]]; then echo /usr/local/bin/brew
  elif command -v brew >/dev/null 2>&1; then command -v brew
  else echo ""; fi
}

_brew_path() {
  local brew; brew="$(_brew_bin)"
  [[ -n "$brew" ]] || return 1
  eval "$("$brew" shellenv)"
  local p
  for p in python@3.12 postgresql@18 postgresql@17 postgresql@16; do
    p="$("$brew" --prefix "$p" 2>/dev/null || true)"
    [[ -n "$p" && -d "$p/bin" ]] && export PATH="$p/bin:$PATH"
  done
  return 0
}

_pkg_install() {
  local os; os="$(_os)"
  case "$os" in
    macos)
      _brew_path || _die "Homebrew required on macOS: https://brew.sh"
      local f
      for f in "$@"; do
        if brew list --formula "$f" >/dev/null 2>&1; then
          _info "brew: $f already installed"
        else
          _info "brew install $f"
          brew install "$f" || _warn "brew install $f failed"
        fi
      done
      ;;
    debian)
      _need_sudo
      _info "apt-get install $*"
      $SUDO apt-get update -qq || true
      $SUDO apt-get install -y "$@" || _warn "apt-get install $* failed"
      ;;
    fedora)
      _need_sudo
      _info "dnf install $*"
      $SUDO dnf install -y "$@" || _warn "dnf install $* failed"
      ;;
    arch)
      _need_sudo
      _info "pacman -S --needed $*"
      $SUDO pacman -S --needed --noconfirm "$@" || _warn "pacman $* failed"
      ;;
    *) _warn "unknown distro — install manually: $*" ;;
  esac
}

_need_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then SUDO=""
  elif command -v sudo >/dev/null 2>&1; then SUDO="sudo"
  else _die "need root or sudo to install packages"; fi
}

# --- .env -------------------------------------------------------------------

_env_get() {
  local key="$1" file="${2:-$ENV_FILE}"
  [[ -f "$file" ]] || return 0
  python3 - "$key" "$file" <<'PY'
import sys
from pathlib import Path
key, path = sys.argv[1], sys.argv[2]
for raw in Path(path).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith(key + "="):
        print(line[len(key) + 1 :], end="")
        break
PY
}

_env_upsert() {
  local key="$1"
  python3 - "$key" "$ENV_FILE" <<'PY'
import os
import sys
from pathlib import Path

key, path = sys.argv[1], Path(sys.argv[2])
value = os.environ.get("AGENT_ENV_VALUE", "")
text = path.read_text(encoding="utf-8") if path.exists() else ""
out, found = [], False
for line in text.splitlines():
    stripped = line.lstrip()
    if stripped.startswith("#") or "=" not in line:
        out.append(line)
        continue
    name = line.split("=", 1)[0].strip()
    if name == key:
        out.append(f"{key}={value}")
        found = True
    else:
        out.append(line)
if not found:
    if out and out[-1] != "":
        out.append("")
    out.append(f"{key}={value}")
path.write_text("\n".join(out) + "\n", encoding="utf-8")
path.chmod(0o600)
PY
}

_env_set() { local key="$1" value="$2"; AGENT_ENV_VALUE="$value" _env_upsert "$key"; }

_ensure_env_file() {
  [[ -f "$EXAMPLE" ]] || _die ".env.example missing — run from the repo root"
  if [[ ! -f "$ENV_FILE" ]]; then
    umask 077
    cp "$EXAMPLE" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    _info "created .env from .env.example (mode 600)"
  else
    _info ".env exists — only requested keys change"
  fi
}

_database_url() {
  local url; url="$(_env_get DATABASE_URL)"
  [[ -n "$url" ]] || url="postgresql://localhost:5432/agent_db"
  printf '%s' "$url"
}

# --- python -----------------------------------------------------------------

_pick_python() {
  local c
  for c in python3.12 python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
        command -v "$c"; return 0
      fi
    fi
  done
  return 1
}

_ensure_python() {
  local py
  if py="$(_pick_python)"; then
    _info "python: $py ($("$py" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'))"
    printf '%s' "$py"; return 0
  fi
  case "$(_os)" in
    macos) _pkg_install python@3.12; _brew_path || true ;;
    debian) _pkg_install python3 python3-venv python3-pip ;;
    fedora) _pkg_install python3 python3-pip ;;
    arch) _pkg_install python python-pip ;;
  esac
  py="$(_pick_python)" || _die "Python ≥3.12 required"
  printf '%s' "$py"
}

_ensure_venv() {
  local py="$1"
  if [[ ! -x "$PY" ]]; then
    _info "python -m venv venv"
    "$py" -m venv "$VENV" || _die "venv creation failed (install python3-venv?)"
  fi
  _info "pip install -U pip wheel"
  "$PY" -m pip install -q -U pip wheel
  _info "pip install -r requirements.txt"
  "$PY" -m pip install -q -r "$ROOT/requirements.txt" || _die "requirements install failed"
  if [[ "$DO_DEV" -eq 1 ]]; then
    _info "pip install -r requirements-dev.txt"
    "$PY" -m pip install -q -r "$ROOT/requirements-dev.txt" || _warn "dev requirements failed"
  fi
  _info "playwright install chromium"
  "$PY" -m playwright install chromium >/dev/null 2>&1 \
    || _warn "Playwright Chromium failed — browse_web and SERP fallback will not work"
}

# --- postgres ---------------------------------------------------------------

_pg_ready() {
  local host="$1" port="$2"
  if command -v pg_isready >/dev/null 2>&1; then
    pg_isready -h "$host" -p "$port" >/dev/null 2>&1
  elif command -v nc >/dev/null 2>&1; then
    nc -z "$host" "$port" >/dev/null 2>&1
  else
    return 1
  fi
}

_pg_start() {
  case "$(_os)" in
    macos)
      local svc; svc="$(_env_get POSTGRES_SERVICE)"; [[ -n "$svc" ]] || svc="postgresql@18"
      brew services start "$svc" >/dev/null 2>&1 || true
      ;;
    debian|fedora|arch)
      _need_sudo
      $SUDO systemctl enable --now postgresql >/dev/null 2>&1 \
        || $SUDO systemctl enable --now postgresql-16 >/dev/null 2>&1 || true
      ;;
  esac
}

_pg_major_version() {
  local v
  v="$(psql --version 2>/dev/null | sed -n 's/.* \([0-9][0-9]*\)\..*/\1/p')"
  if [[ -z "$v" ]]; then
    v="$(ls -1 /usr/lib/postgresql 2>/dev/null | sort -n | tail -1)"
  fi
  printf '%s' "$v"
}

# Debian main does not ship pgvector; PGDG does. If neither is available, build it.
_install_pgvector_from_source() {
  local pgv="$1" pgcfg="/usr/lib/postgresql/${pgv}/bin/pg_config"
  [[ -x "$pgcfg" ]] || { _warn "pg_config not found for Postgres ${pgv}"; return 1; }
  _pkg_install "postgresql-server-dev-${pgv}" build-essential git
  local tmp; tmp="$(mktemp -d)"
  if git clone --depth 1 https://github.com/pgvector/pgvector.git "$tmp/pgvector" \
     && make -C "$tmp/pgvector" PG_CONFIG="$pgcfg" \
     && make -C "$tmp/pgvector" install PG_CONFIG="$pgcfg"; then
    rm -rf "$tmp"
    return 0
  fi
  rm -rf "$tmp"
  return 1
}

# Run one SQL statement as the postgres superuser (peer auth on the local socket).
_pg_as_postgres_sql() {
  local sql="$1" db="${2:-postgres}"
  if [[ "$(id -u)" -eq 0 ]]; then
    su -s /bin/sh postgres -c "psql -d '$db' -v ON_ERROR_STOP=1 -tAc \"$sql\""
  else
    sudo -u postgres psql -d "$db" -v ON_ERROR_STOP=1 -tAc "$sql"
  fi
}

# Debian/Ubuntu Postgres has no role for the OS user and requires a password over
# TCP. Create a dedicated role, own the database, and write the URL back to .env.
_ensure_pg_role_and_db() {
  local url="$1" host="$2" port="$3" db="$4"
  local user pass

  user="$(python3 -c 'import sys,urllib.parse as u; print(u.urlparse(sys.argv[1]).username or "")' "$url")"
  if [[ -n "$user" ]]; then
    _info "DATABASE_URL already names role '$user' — leaving it"
    if command -v createdb >/dev/null 2>&1; then
      createdb -h "$host" -p "$port" "$db" 2>/dev/null \
        && _info "createdb $db" || _info "database $db already exists"
    fi
    return 0
  fi

  if [[ "$(_os)" == "macos" ]]; then
    # Homebrew: the local user is a superuser, no role or password needed.
    if command -v createdb >/dev/null 2>&1; then
      createdb -h "$host" -p "$port" "$db" 2>/dev/null \
        && _info "createdb $db" || _info "database $db already exists"
    fi
    return 0
  fi

  user="agent"
  pass="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"

  if _pg_as_postgres_sql "SELECT 1 FROM pg_roles WHERE rolname='${user}'" | grep -q 1; then
    _info "role ${user} exists — rotating password"
    _pg_as_postgres_sql "ALTER ROLE ${user} WITH LOGIN PASSWORD '${pass}'" >/dev/null
  else
    _info "CREATE ROLE ${user}"
    _pg_as_postgres_sql "CREATE ROLE ${user} LOGIN PASSWORD '${pass}'" >/dev/null
  fi

  if _pg_as_postgres_sql "SELECT 1 FROM pg_database WHERE datname='${db}'" | grep -q 1; then
    _info "database ${db} already exists"
  else
    _info "CREATE DATABASE ${db} OWNER ${user}"
    _pg_as_postgres_sql "CREATE DATABASE \"${db}\" OWNER ${user}" >/dev/null
  fi

  _env_set DATABASE_URL "postgresql://${user}:${pass}@${host}:${port}/${db}"
  _info "DATABASE_URL written to .env (role ${user}, generated password)"
}

_ensure_postgres() {
  local url host port db
  url="$(_database_url)"
  host="$(python3 -c 'import sys,urllib.parse as u; p=u.urlparse(sys.argv[1]); print(p.hostname or "127.0.0.1")' "$url")"
  port="$(python3 -c 'import sys,urllib.parse as u; p=u.urlparse(sys.argv[1]); print(p.port or 5432)' "$url")"
  db="$(python3 -c 'import sys,urllib.parse as u; p=u.urlparse(sys.argv[1]); print((p.path or "/agent_db").strip("/") or "agent_db")' "$url")"

  if [[ "$SKIP_POSTGRES" -eq 0 ]]; then
    case "$(_os)" in
      macos)
        _pkg_install postgresql@18
        _brew_path || true
        if ! brew list --formula pgvector >/dev/null 2>&1; then
          _pkg_install pgvector
        else
          _info "brew: pgvector already installed"
        fi
        ;;
      debian)
        _pkg_install postgresql postgresql-contrib
        local pgv; pgv="$(_pg_major_version)"
        if [[ -n "$pgv" ]]; then
          _pkg_install "postgresql-${pgv}-pgvector"
          if ! dpkg -s "postgresql-${pgv}-pgvector" >/dev/null 2>&1; then
            _warn "pgvector package missing — building from source"
            _install_pgvector_from_source "$pgv" \
              || _warn "pgvector build failed — install PGDG or build it manually"
          fi
        else
          _pkg_install pgvector || _warn "pgvector package missing"
        fi
        ;;
      fedora)
        _pkg_install postgresql-server postgresql-contrib
        _pkg_install pgvector || _warn "pgvector package missing"
        ;;
      arch)
        _pkg_install postgresql pgvector
        ;;
    esac
    _pg_start
  fi

  local n=30
  while ((n--)); do _pg_ready "$host" "$port" && break; sleep 1; done
  _pg_ready "$host" "$port" || _die "PostgreSQL not reachable at ${host}:${port}"

  _ensure_pg_role_and_db "$url" "$host" "$port" "$db"
  url="$(_database_url)"

  _info "CREATE EXTENSION vector"
  if _pg_as_postgres_sql "CREATE EXTENSION IF NOT EXISTS vector" "$db" >/dev/null 2>&1; then
    local ver
    ver="$(_pg_as_postgres_sql "SELECT extversion FROM pg_extension WHERE extname='vector'" "$db" 2>/dev/null | tr -d '[:space:]')"
    _info "pgvector: ${ver:-installed}"
  elif psql "$url" -v ON_ERROR_STOP=1 -q -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null; then
    _info "pgvector: installed"
  else
    _warn "could not CREATE EXTENSION vector — needs a superuser role"
  fi
}

# --- providers --------------------------------------------------------------

_probe_http() {
  local url="$1" key="${2:-}"
  local args=(-s -o /dev/null -w '%{http_code}' --max-time 5)
  [[ -n "$key" ]] && args+=(-H "Authorization: Bearer ${key}")
  curl "${args[@]}" "$url" 2>/dev/null || true
}

_probe_endpoint() {
  local label="$1" provider="$2" base="$3" key="$4" model="$5"
  local url code
  if [[ "$provider" == "ollama" ]]; then
    url="${base%/}/api/tags"
  else
    url="${base%/}"
    [[ "$url" == */v1 ]] || url="$url/v1"
    url="$url/models"
  fi
  code="$(_probe_http "$url" "$key")"
  case "$code" in
    200) _info "${label} endpoint reachable: $url (model ${model})" ;;
    401) _die "${label} endpoint rejected the API key (401): $url" ;;
    000|'') _warn "${label} endpoint did not answer: $url — start the server yourself" ;;
    *) _warn "${label} endpoint returned HTTP $code: $url" ;;
  esac
}

# The agent does not install or start a model runtime. It only writes where to find one.
_configure_providers() {
  local provider embed_provider base key model embed_base embed_key embed_model embed_dim
  provider="${PROVIDER_FLAG:-$(_env_get LLM_PROVIDER)}"; [[ -n "$provider" ]] || provider="ollama"
  embed_provider="${EMBED_PROVIDER_FLAG:-$(_env_get EMBED_PROVIDER)}"; [[ -n "$embed_provider" ]] || embed_provider="ollama"

  base="${BASE_URL_FLAG:-$(_env_get LLM_BASE_URL)}"
  key="${API_KEY_FLAG:-$(_env_get LLM_API_KEY)}"
  model="${MODEL_FLAG:-$(_env_get REASONING_MODEL)}"
  embed_base="${EMBED_BASE_URL_FLAG:-$(_env_get EMBED_BASE_URL)}"
  embed_key="${EMBED_API_KEY_FLAG:-$(_env_get EMBED_API_KEY)}"
  embed_model="${EMBED_MODEL_FLAG:-$(_env_get EMBED_MODEL)}"
  embed_dim="${EMBED_DIM_FLAG:-$(_env_get EMBED_DIM)}"

  if [[ "$provider" == "ollama" ]]; then
    [[ -n "$base" ]] || base="$(_env_get OLLAMA_BASE_URL)"
    [[ -n "$base" ]] || base="http://127.0.0.1:11434"
    [[ -n "$model" ]] || model="qwen3:8b"
  else
    [[ -n "$base" ]] || _die "--provider openai requires --base-url (e.g. http://127.0.0.1:8000/v1)"
    [[ -n "$model" ]] || model="gpt-4o-mini"
  fi

  if [[ "$embed_provider" == "ollama" ]]; then
    [[ -n "$embed_base" ]] || embed_base="$base"
    [[ -n "$embed_model" ]] || embed_model="nomic-embed-text-v2-moe"
  else
    [[ -n "$embed_base" ]] || embed_base="$base"
    [[ -n "$embed_key" ]] || embed_key="$key"
    [[ -n "$embed_model" ]] || embed_model="text-embedding-3-small"
  fi
  [[ -n "$embed_dim" ]] || embed_dim="768"

  _env_set LLM_PROVIDER "$provider"
  _env_set LLM_BASE_URL "$base"
  _env_set LLM_API_KEY "$key"
  _env_set REASONING_MODEL "$model"
  _env_set EMBED_PROVIDER "$embed_provider"
  _env_set EMBED_BASE_URL "$embed_base"
  _env_set EMBED_API_KEY "$embed_key"
  _env_set EMBED_MODEL "$embed_model"
  _env_set EMBED_DIM "$embed_dim"

  _probe_endpoint "chat" "$provider" "$base" "$key" "$model"
  _probe_endpoint "embedding" "$embed_provider" "$embed_base" "$embed_key" "$embed_model"
}

# --- migrate / smoke --------------------------------------------------------

_migrate() {
  [[ -x "$VENV/bin/alembic" ]] || _die "alembic missing in venv"
  _info "alembic upgrade head"
  (cd "$ROOT" && "$VENV/bin/alembic" upgrade head) || _die "migration failed"
}

_smoke() {
  _info "import smoke"
  (cd "$ROOT" && PYTHONPATH="$ROOT" "$PY" -c "import src.graph.workflow, src.api.app" ) \
    || _die "import smoke failed — the tree is inconsistent"
  _info "env parity"
  (cd "$ROOT" && "$PY" scripts/check_env_sync.py) || _warn "env-check reported drift"
}

# --- optional extras --------------------------------------------------------

_configure_telegram() {
  local existing; existing="$(_env_get TELEGRAM_BOT_TOKEN)"
  if [[ -n "$existing" ]]; then _info "Telegram token already set — leaving it"; return 0; fi
  if [[ "$YES" -eq 1 ]]; then _info "Telegram skipped (--yes)"; return 0; fi
  _ask_yn "Enable Telegram bot?" n || { _info "Telegram skipped"; return 0; }

  local token="" chat=""
  echo "BotFather token (input hidden):"
  read -r -s token || true; echo ""
  [[ -n "$token" ]] || { _warn "empty token — Telegram left unset"; return 0; }
  chat="$(_ask "Numeric chat id (userinfobot / getUpdates)" "")"
  _env_set TELEGRAM_BOT_TOKEN "$token"
  _env_set TELEGRAM_POLLING 1
  if [[ -n "$chat" ]]; then
    _env_set TELEGRAM_ALLOWED_CHAT_IDS "$chat"
    _env_set TELEGRAM_OWNER_CHAT_ID "$chat"
    _env_set HEART_NOTIFY_CHAT_ID "$chat"
  else
    _warn "no chat id — polling will fail closed"
  fi
  _info "Telegram keys written (token not printed)"
}

_configure_solitary() {
  [[ "$YES" -eq 1 ]] && return 0
  _ask_yn "Enable solitary background passes (SOLITARY_ENABLED)?" n \
    && _env_set SOLITARY_ENABLED 1
}

_pg_identity_exists() {
  "$PY" - "$(_database_url)" <<'PY'
import sys
import psycopg

try:
    with psycopg.connect(sys.argv[1]) as c:
        print(1 if c.execute("SELECT count(*) FROM agent_identity").fetchone()[0] else 0)
except Exception:
    print(0)
PY
}

# Identity is part of a working install, not a separate chore. A random
# passphrase is generated and kept in .env (mode 600) for recovery; Keychain
# holds the working copy.
_maybe_identity() {
  if [[ "$(_pg_identity_exists)" == "1" ]]; then
    _info "identity already exists — leaving it"
    return 0
  fi
  local pass
  pass="$(_env_get IDENTITY_PASSPHRASE)"
  if [[ -z "$pass" ]]; then
    pass="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
    _env_set IDENTITY_PASSPHRASE "$pass"
    _info "IDENTITY_PASSPHRASE generated and stored in .env (recovery only)"
  fi
  if AGENT_IDENTITY_PASSPHRASE="$pass" PYTHONPATH="$ROOT" "$PY" -m src.identity.cli init; then
    _info "Ed25519 identity created"
  else
    _warn "identity init failed — later: make init-identity"
  fi
}

# The daemon is the default on both platforms; --no-daemon opts out.
# macOS uses launchd; Linux uses systemd.
_maybe_daemon() {
  if [[ "$DO_DAEMON" == "0" ]]; then
    _info "daemon skipped (--no-daemon)"
    return 0
  fi
  case "$(_os)" in
    macos)
      if make -C "$ROOT" daemon-install; then
        _info "launchd: API + Heart installed"
      else
        _warn "daemon-install failed — later: make daemon-restart"
      fi
      ;;
    debian|fedora|arch|linux)
      if ! command -v systemctl >/dev/null 2>&1; then
        _warn "no systemd on this host — start manually: make api"
        return 0
      fi
      if make -C "$ROOT" systemd-install; then
        _info "systemd: API + Heart installed"
      else
        _warn "systemd install failed — later: make systemd-install"
      fi
      ;;
    *) _info "unknown OS — no service manager installed" ;;
  esac
}

_probe_searxng() {
  local url; url="$(_env_get SEARXNG_URL)"; [[ -n "$url" ]] || url="http://127.0.0.1:8888"
  if curl -sf --max-time 2 "$url" >/dev/null 2>&1; then
    _info "SearXNG reachable at $url"
  else
    _warn "SearXNG not at $url — search_web falls back to ddgs / Playwright / Wikipedia"
  fi
}

# --- main -------------------------------------------------------------------

main() {
  echo "persistent-agent install — $ROOT"
  echo "Installs Python venv, PostgreSQL + pgvector, the database, and .env."
  echo "Does NOT install a model runtime: start Ollama or vLLM yourself."
  [[ -f "$ROOT/src/config.py" ]] || _die "not a persistent-agent tree"

  _ensure_env_file

  _step "Python"
  local py; py="$(_ensure_python)"
  _ensure_venv "$py"

  _step "PostgreSQL + pgvector"
  _ensure_postgres

  _step "Providers"
  _configure_providers

  _step "Migrations"
  _migrate

  _step "Verification"
  _smoke

  _step "Identity"
  _maybe_identity

  _step "Optional"
  _configure_telegram
  _configure_solitary
  _probe_searxng
  _maybe_daemon

  echo ""
  echo "Done."
  echo "  REPL:    ./start.sh"
  echo "  CLI:     make chat MSG='…'"
  echo "  API:     make api"
  echo "  Daemon:  make daemon-restart"
  echo "  Prompt:  prompts/agent/system_prompt.yml"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  _parse_args "$@"
  main
fi
