# Security Policy

## Scope

`persistent-agent` is a single-operator agent that runs on the operator's own machine.
It is not a hosted service and has no multi-tenant isolation. The security model assumes
the operator controls the host, the database, and the model runtime.

## Reporting a vulnerability

Do not open a public issue for a security problem.

Report privately to **cha0skvlt** through GitHub Security Advisories on this repository
(`Security` → `Report a vulnerability`). Include:

- affected version or commit,
- a minimal reproduction,
- the impact you believe it has.

Expect an initial reply within seven days. This is a personal project, not a funded
product; there is no bug bounty.

## What is in scope

| Area | Why it matters |
| --- | --- |
| SSRF in `fetch_url` / `browse_web` | The agent fetches arbitrary URLs. Private and loopback targets must stay blocked (`ip.is_global` + DNS pin, post-navigation re-check). |
| Prompt injection through web content | Fetched pages reach the model. A page must not be able to trigger durable writes or tool calls the operator did not ask for. |
| Memory poisoning | `save_memory`, `diary_replace`, and deferred reflect write to Postgres. Provenance filtering must not be bypassable. |
| Telegram allowlist | An empty allowlist must deny every chat. A misconfiguration must fail closed at boot. |
| API bind | A non-loopback `API_HOST` must require a non-empty `API_KEY`. |
| Key material | Ed25519 keys live in the `agent_identity` table. They must never be logged or exported to the repository. |
| Backup handling | Database dumps are encrypted and must never be committed. |

## What is out of scope

- The model's own output. Wording comes from the weights and the prompt; the code does not
  filter content.
- Denial of service against your own host.
- Anything that requires an attacker to already have shell access to the machine.
- Third-party services (Ollama, SearXNG, Telegram, Wikipedia) and their own policies.

## Operator responsibilities

- Keep `.env` out of version control. It is gitignored; do not override that.
- Set `TELEGRAM_ALLOWED_CHAT_IDS` or `TELEGRAM_OWNER_CHAT_ID`. Boot fails without one when a
  bot token is present.
- Set `API_KEY` if `API_HOST` is anything other than loopback.
- Keep `SEARXNG_SECRET` in `.env` only.
- Encrypt database backups and store them outside the repository.
- Run `make selftest` after changing configuration.

## Supported versions

The project ships from `main`. Only the latest release receives fixes.
