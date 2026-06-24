"""Production WSGI entrypoint (gunicorn wsgi:app).

Used by the Procfile / render.yaml deployment. Importing this module imports the
Flask app and, in DEMO_MODE, runs the boot preflight once (ping pinned models +
embeddings, warn on the silent-empty failure class). The app's __main__ block
does NOT run under gunicorn, so the preflight lives here instead. Tests import
``app`` directly, never ``wsgi``, so they don't trigger these live calls.

IMPORTANT for the public demo: run a SINGLE worker (gunicorn -w 1 --threads N).
The budget ledger and rate limiter in demo_budget/app are in-process; multiple
workers would each keep a separate ledger, multiplying the spend cap by the
worker count. One worker with threads handles demo concurrency fine.
"""

from __future__ import annotations

import os

import demo_budget
from app import app  # noqa: F401  (gunicorn looks up `app`)

if demo_budget.demo_enabled():
    if os.environ.get("DEMO_SKIP_PREFLIGHT", "").strip().lower() not in ("1", "true", "yes", "on"):
        try:
            demo_budget.preflight()
        except Exception as exc:  # noqa: BLE001  (never block startup on the check)
            print(f"[demo preflight] skipped: {exc}")
    print(f"[demo] public demo — general={demo_budget.general_model()[1]} "
          f"code={demo_budget.code_model()[1]}")
