"""Shared pytest setup.

Pin ONE throwaway ``KG_DATA_DIR`` for the whole test session, before any test
module is imported. pytest imports the root ``conftest.py`` first, so this dir
wins; each ``test_*.py`` also sets ``KG_DATA_DIR`` via ``setdefault`` so it still
works when run directly as a script (``python test_foo.py``).

Why this is needed: the production modules capture ``DATA_DIR`` from
``KG_DATA_DIR`` at import time, but the MCP loop test spawns a subprocess that
reads ``KG_DATA_DIR`` from the environment at spawn time. When each test module
set the env var unconditionally at import, the last-imported module's value won,
desyncing the subprocess from where data was seeded and making the MCP test fail
only under the full suite (it passed in isolation). One shared dir keeps every
module — and the subprocess — in agreement.
"""

import os
import tempfile

os.environ.setdefault("KG_DATA_DIR", tempfile.mkdtemp(prefix="byowiki_tests_"))
