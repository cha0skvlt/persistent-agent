#!/usr/bin/env bash
# Allow live agent + one background job without serializing on Ollama (ARCHITECTURE.md §21).
set -euo pipefail
launchctl setenv OLLAMA_NUM_PARALLEL 2
launchctl setenv OLLAMA_MAX_LOADED_MODELS 2
echo "OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=2 (launchctl user env)"
if command -v brew >/dev/null 2>&1; then
  brew services restart ollama
  echo "ollama restarted via brew"
else
  echo "Restart Ollama manually so env vars apply to the server process."
fi
