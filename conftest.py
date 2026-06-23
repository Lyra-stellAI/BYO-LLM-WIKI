"""Shared pytest setup.

Force ONE throwaway ``KG_DATA_DIR`` for the whole test session, before any test
module is imported. pytest imports the root ``conftest.py`` first, so this dir
wins for every module and for the MCP subprocess.

This assignment is UNCONDITIONAL on purpose: the test setup functions call
``memory.clear()`` / ``kg.clear()`` / ``sk.clear()``, which would wipe a real
library. If the developer's shell already has ``KG_DATA_DIR`` pointing at a real
BYO-WIKI data dir, a ``setdefault`` would keep it and the suite would delete live
data — so we always override with a fresh temp dir and never honor an inherited
value. ``BYOWIKI_TEST_DATA_DIR`` is published so each ``test_*.py`` can adopt the
SAME dir (keeping every module + the MCP subprocess in agreement) without
clobbering it to a different per-file temp.

Why one shared dir: production modules capture ``DATA_DIR`` from ``KG_DATA_DIR``
at import, but the MCP loop test spawns a subprocess that reads ``KG_DATA_DIR``
from the environment at spawn time. If each module set its own dir, the values
desync and the MCP test fails only under the full suite.
"""

import os
import tempfile

# Unconditional: never inherit a (possibly real) KG_DATA_DIR from the shell.
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="byowiki_tests_")
os.environ["KG_DATA_DIR"] = _TEST_DATA_DIR
os.environ["BYOWIKI_TEST_DATA_DIR"] = _TEST_DATA_DIR

# Force LOCAL storage backends for the whole test session and blank any cloud DB
# URLs. Importing app.py runs config.load_env(), which would otherwise load the
# developer's real .env (CACHED_STORE_BACKEND/MEMORY_BACKEND=supabase + a live
# SUPABASE_DB_URL) into the test process — and with psycopg installed, tests
# would read/clear/write the production Supabase database. config.load_env uses
# override=False, so setting these here (before any import) makes it skip them.
os.environ["CACHED_STORE_BACKEND"] = "local"
os.environ["MEMORY_BACKEND"] = "local"
for _v in ("SUPABASE_DB_URL", "CACHED_STORE_DB_URL", "MEMORY_DB_URL"):
    os.environ[_v] = ""
