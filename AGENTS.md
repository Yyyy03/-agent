# AGENTS.md

Two-process system. Do not treat this as a single Next.js app — the frontend is a
thin proxy and all real work happens in a separate Python bridge.

## Architecture

- **Frontend** — vinext (Next.js 16 + React 19 on Cloudflare Vite). Site code in `app/`,
  worker entry `worker/index.ts`, Vite/Cloudflare wiring in `vite.config.ts`. Serves on
  `:3000`. The `/api/*` routes do not compute anything; they proxy to the bridge.
- **Backend** — Python package `fire_agent` in `backend/src/fire_agent`. The web path goes
  through the streaming HTTP bridge `backend/bridge/fire_agent_bridge.py` (aiohttp), which
  lives *outside* `src/fire_agent` on purpose so the harness stays byte-compatible with the
  reference machine. Edit the harness, not the bridge, for research logic.
- **Data flow** — `app/api/research/route.ts` (`:3000`) proxies `POST /api/research` →
  bridge `POST /v1/research`, relaying an NDJSON stream. Shared proxy/CORS/auth helpers are
  in `app/api/bridge-helpers.ts` (used by `auth/*`, `history`, `research`).
- **DB** — `db/schema.ts` is intentionally empty; Drizzle/D1 is optional and unused. `drizzle.config.ts`
  is sqlite-dialect. Don't assume a database exists.

## Commands

Frontend (Node `>=22.13`):
- `npm run dev` / `npm run build` / `npm start`
- `npm run lint` (eslint, ignores `dist`/`.next`)
- `npm test` = `npm run build` + `node --test tests/rendered-html.test.mjs` (builds first, so it is slow)
- `npm run db:generate` (only if you add a real Drizzle schema)
- No typecheck script: run `npx tsc --noEmit`
- Path alias: `@/*` → repo root (tsconfig `paths`)

Backend bridge (needs a venv with `backend/requirements.txt` installed):
- `backend/venv/bin/python bridge/fire_agent_bridge.py --host 127.0.0.1 --port 4182 --env-file .env.local_mintcu`
- Python CLIs (if you `pip install -e backend/`): `fire-agent`, `fire-agent-benchmarks`, plus
  `fire-agent-sft-*` / `fire-agent-prepare-*` entry points (see `backend/pyproject.toml`).

One-shot launchers, one per OS — do not run the wrong one:
- **Windows:** `.\start_latest_finagent.ps1` (PowerShell). Auto-creates `backend\.venv-win`
  (prefers `py -3.12`, falls back to `py`/`python`), installs `backend/requirements.txt`,
  frees ports, then starts bridge `:4182` + frontend `:3000` and checks health.
- **macOS:** `./start_latest_finagent.sh` (hardcoded `/Users/tanglei/...` paths, `lsof`,
  `/opt/homebrew/bin/node`) — will not run on Windows.

Both launchers call the vinext CLI directly with `node --env-file=.env.local_mintcu` (not
`npm run dev`) so the frontend actually loads `FIRE_AGENT_BRIDGE_URL`.

## Ports (do not mix versions)

- Frontend `:3000`, bridge `:4182` (current/new link).
- An **old** version used `:4173` (frontend) and `:4180` (bridge). `:4180` is also the code
  default in `bridge-helpers.ts` / `research/route.ts` when `FIRE_AGENT_BRIDGE_URL` is unset —
  so always set `FIRE_AGENT_BRIDGE_URL=http://127.0.0.1:4182` for the current setup.
- Health check: `curl http://127.0.0.1:4182/health` → `{ok, backend:"fire-agent-harness", tools:[...], activeRuns:{...}}`.
- Cancel a stuck run: `DELETE /api/research?runId=<id>` (frontend) → bridge `DELETE /v1/research/<runId>`.

## Env files

Two near-duplicate files exist — keep them in sync:
- `.env.local` — read by the **frontend** (`node --env-file` in the start script / dev).
- `.env.local_mintcu` — read by the **bridge** (`--env-file`).
- Both must define `FIRE_AGENT_BRIDGE_URL`, `FIRE_AGENT_BRIDGE_TOKEN`, and the Tushare config
  (`FIRE_AGENT_TUSHARE_TOKEN`, `FIRE_AGENT_TUSHARE_HTTP_URL`, plus `TUSHARE_HTTP_URL`).
- Token/key changes (Tushare, model) require a **restart** of the bridge to take effect.

## Gotchas

- **Model naming:** the 27B model is shown publicly as **Mint-Ag** but its internal/vLLM id stays
  `mint-sg` / `Mint-Sg`. Frontend `ModelId` is `"mint-cu" | "mint-sg"` (see `research/route.ts`).
  Renaming the display label is fine; do not rename the internal id.
- **Research modes:** `deep` = multi-round with tools + source timeline; `fast` = single round,
  `max_steps=1`, no tools, returns native thinking then answer. Default is `deep`.
- **NDJSON protocol** (one JSON object per line): events include `run_started`, `answer_delta`,
  `done` (with `elapsedMs`, `evidenceCount`), plus plan/think/tool/evidence trajectory lines.
  Greeting-only questions (`你好`, `hi`, ...) short-circuit to a canned NDJSON response without
  hitting the bridge (`isSimpleGreeting`).
- **Auth:** local session auth under `app/api/auth/` (login/logout/me/redeem). `bridge-helpers.ts`
  `requireSessionHeader` gates session-required routes with a `401 {error:"auth_required"}`.
  The start guide references a local account, but credentials/secrets live in the env files, not code.
- **Logs:** frontend dev log `latest-finagent-dev.log`, bridge log `latest-finagent-bridge.log`
  (root); request logs in `backend/logs/*.jsonl`.
- **vite watch** ignores `backend/venv`, `backend/.venv`, `backend/logs` — keep new generated/output
  dirs out of the watched tree or HMR will thrash.