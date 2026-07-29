# Deploying the public demo

Get a public `https://…` URL for the budget-capped demo. The repo already ships
what you need: `wsgi.py` (gunicorn entrypoint), `Procfile`, `render.yaml`, and a
pre-seeded library in `demo_data/` (28 docs → vectors + KG, so Q&A and the graph
work with zero per-visitor ingestion cost).

You set **three secret keys** in the host's dashboard; everything else is preset.

> **One worker, on purpose.** The spend budget and rate limiter default to
> process-local counters, so the start command runs **one** gunicorn worker with
> threads (`--workers 1 --threads 8`). More workers would each keep a separate
> ledger and multiply the spend cap. Don't raise `--workers`.
>
> The exception is a host that clones the process for you (Vercel and any other
> serverless platform). There you can't pin the worker count, so point
> `demo_store` at a Redis ledger instead — see the Vercel section below.

## Secrets you provide (in the dashboard)
| Var | Purpose |
|---|---|
| `GEMINI_API_KEY` | general model (`gemini-2.5-flash`): summaries / Q&A / KG |
| `DASHSCOPE_API_KEY` | code model (`qwen3-coder-next`): skill generation |
| `OPENAI_API_KEY` | embeddings (`text-embedding-3-small`) — required for Q&A retrieval |

Everything else (`DEMO_MODE=1`, local backends, budget knobs) is set by the
config files / presets below. **Leave all cloud DB URLs unset** — the app
refuses to boot in demo mode if `SUPABASE_DB_URL` / `CACHED_STORE_DB_URL` /
`MEMORY_DB_URL` / `SKILL_GRAPH_DB_URL` is set.

---

## Vercel (serverless — uses `vercel.json`)

Vercel serves the Flask app as a **single function that autoscales**, which changes
two of the demo's assumptions. Both are handled in the repo; you just have to
supply the pieces below.

1. Install the CLI (`npm i -g vercel`), then from the repo root:
   ```bash
   vercel link                     # create/link the project
   vercel deploy                   # preview URL
   vercel deploy --prod            # promote when you're happy
   ```
   Or connect the repo in the dashboard and set the production branch to this one.
2. **Project → Settings → Environment Variables** — the non-secret preset:
   ```
   DEMO_MODE=1
   CACHED_STORE_BACKEND=local
   MEMORY_BACKEND=local
   KG_DATA_DIR=demo_data
   DEMO_RUNTIME_DATA_DIR=/tmp/byowiki-data
   DEMO_TRUST_PROXY=1
   DEMO_SKIP_PREFLIGHT=1
   LANGSMITH_TRACING=false
   PYTHONUNBUFFERED=1
   DEMO_VISITOR_USD=0.05
   DEMO_GLOBAL_USD=5.0
   DEMO_RPM=20
   DEMO_GLOBAL_RPM=120
   ```
   Plus the three secret keys from the table above.
3. **Add a Redis ledger.** Storage → Marketplace → **Upstash Redis** (free tier is
   ample; create it in `us-east-1` to sit beside the default `iad1` function
   region). The integration injects `KV_REST_API_URL` / `KV_REST_API_TOKEN`, which
   `demo_store` picks up on its own — no code change, no extra dependency. Confirm
   with `curl …/api/demo-status` → `"ledger": "redis"`. Override the names with
   `DEMO_REDIS_REST_URL` / `DEMO_REDIS_REST_TOKEN` if you'd rather bring your own.

### What's different from a single-process host

| | Render / Railway / Fly | Vercel |
|---|---|---|
| Spend cap | in-process, one worker | Redis counters shared by every instance (falls back to per-instance if Redis is unreachable — check the logs for `[demo store]`) |
| Rate limit | 60 s sliding window | 60 s fixed window; a burst across a boundary can briefly see 2x `DEMO_RPM` |
| Data dir | writable, survives until restart | `demo_data/` is read-only in the bundle, so `vercel_bootstrap` copies it to `/tmp` on each cold start; visitor writes die with the instance |
| Visitor identity | peer address | `X-Forwarded-For`, which is why **`DEMO_TRUST_PROXY=1` is required** — without it every visitor collapses into one budget key |
| Boot preflight | once per boot | skipped (`DEMO_SKIP_PREFLIGHT=1`); once per cold start would be three live model calls on every scale-up. Run it once by hand instead — see *Verify it's live* |
| `hnswlib` | installed | commented out of `requirements.txt`: it is source-only on PyPI and needs a C++ toolchain the builder lacks. `vectorstore` falls back to a numpy scan |

Static assets keep being served by Flask (Vercel's `public/**` convention doesn't
apply in services mode), with `SEND_FILE_MAX_AGE_DEFAULT` set in `app.py` so they
are cacheable at all — Flask's default `Cache-Control: no-cache` costs a function
invocation per asset per page load. A `headers` rule in `vercel.json` does *not*
work for this: the function's own header wins.

The build log prints `WARNING! Build output contains no "functions" or "static"
directory`. That is a false alarm in services mode — the service writes its output
elsewhere — and the deployment works. Don't chase it.

`vercel dev` currently fails on Windows while building the synthetic local package
(`No such file or directory: ...app.egg-info\dependency_links.txt`). It's a local
toolchain issue, not a config one — the cloud build is unaffected. Use
`python app.py` for local work and a preview deployment to test the Vercel path.

---

## Render (closest to click-deploy — uses `render.yaml`)
1. Push this branch to GitHub (done if you opened the PR).
2. Render dashboard → **New → Blueprint** → connect the repo → pick the branch.
   Render reads `render.yaml`, creates the `byo-wiki-demo` web service, and
   prompts for `GEMINI_API_KEY`, `DASHSCOPE_API_KEY`, `OPENAI_API_KEY`.
3. **Apply**. First build installs deps and starts gunicorn; you get
   `https://byo-wiki-demo.onrender.com`. Health check: `/api/demo-status`.

Notes: free tier sleeps after inactivity and cold-starts on the next hit (and
resets any visitor writes back to the seeded baseline — fine for a demo).

## Railway (Procfile auto-detected)
1. Railway → **New Project → Deploy from GitHub repo** → pick the repo/branch.
   Nixpacks detects Python + the `Procfile` automatically.
2. **Variables** tab → add:
   ```
   DEMO_MODE=1
   CACHED_STORE_BACKEND=local
   MEMORY_BACKEND=local
   KG_DATA_DIR=demo_data
   LANGSMITH_TRACING=false
   DEMO_VISITOR_USD=0.05
   DEMO_GLOBAL_USD=5.0
   PYTHONUNBUFFERED=1
   GEMINI_API_KEY=...
   DASHSCOPE_API_KEY=...
   OPENAI_API_KEY=...
   ```
3. **Settings → Networking → Generate Domain** for the public URL. Railway sets
   `$PORT`; the Procfile binds it.

## Fly.io (Dockerfile-less, buildpacks)
```bash
fly launch --no-deploy            # detects Python; keep the generated fly.toml
# set internal_port to 8080 (or match $PORT) and the process to the Procfile web cmd
fly secrets set GEMINI_API_KEY=... DASHSCOPE_API_KEY=... OPENAI_API_KEY=...
fly secrets set DEMO_MODE=1 CACHED_STORE_BACKEND=local MEMORY_BACKEND=local \
  KG_DATA_DIR=demo_data LANGSMITH_TRACING=false DEMO_VISITOR_USD=0.05 \
  DEMO_GLOBAL_USD=5.0 PYTHONUNBUFFERED=1
fly deploy
```
Fly has persistent volumes if you ever want visitor data to survive restarts;
not needed for a stateless demo.

---

## Verify it's live
```bash
curl https://YOUR-URL/api/demo-status
# {"demo": true, "general_model": "gemini-2.5-flash", "code_model": "qwen3-coder-next",
#  "ledger": "redis", ...}
```
`ledger` tells you which cap is in force: `redis` = shared across instances,
`memory` = per-instance (correct on a one-worker host, a misconfiguration on Vercel).

On boot the logs show the preflight (`[demo preflight] ok general/code/embeddings`).
If any line says WARN, a model id or key is wrong — fix it before sharing the URL.
Where the preflight is skipped (Vercel), run it once against the deployed keys:

```bash
DEMO_MODE=1 GEMINI_API_KEY=... DASHSCOPE_API_KEY=... OPENAI_API_KEY=... \
python -c "import demo_budget; demo_budget.preflight()"
```

## Tunables (optional env)
`DEMO_VISITOR_USD` (0.05), `DEMO_GLOBAL_USD` (5.0), `DEMO_RPM` (20),
`DEMO_GLOBAL_RPM` (120), `DEMO_INFLIGHT` (3), `DEMO_EXTRACT_MAX` (5),
`DEMO_MAX_TOKENS` (1024), `DEMO_WINDOW_SEC` (86400), `DEMO_MODEL`,
`DEMO_CODE_MODEL`, `DEMO_TRUST_PROXY` (set `1` only behind a trusted proxy),
`DEMO_SKIP_PREFLIGHT`.

## Re-seeding (optional)
The seed ships committed in `demo_data/`. To rebuild or change the library, edit
`DEMO_SEED_URLS` and run locally, then commit the result:
```bash
KG_DATA_DIR=./demo_data CACHED_STORE_BACKEND=local MEMORY_BACKEND=local \
DEMO_SEED_PROVIDER=gemini DEMO_SEED_MODEL=gemini-2.5-flash python demo_seed.py
git add -f demo_data/ && git commit -m "demo: reseed library"
```
(Prefer not to ship the seed in git? Move seeding to the host's build step
instead — e.g. Render `buildCommand: pip install -r requirements.txt && python
demo_seed.py` — but that needs the API keys at build time and re-seeds on every
deploy.)
