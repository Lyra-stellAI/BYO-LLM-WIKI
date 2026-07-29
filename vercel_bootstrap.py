"""Make the demo's data directory writable when running on Vercel.

A Vercel deployment bundle is read-only; only ``/tmp`` can be written, and it is
per-instance and thrown away when the instance is recycled. The demo still lets
visitors write (KG builds, skill builds, page caching), and ``knowledge_graph``,
``memory``, ``cached_store``, ``skill_library`` and ``skill_runs`` all resolve
``KG_DATA_DIR`` and ``mkdir()`` it AT IMPORT TIME — so the switch has to happen
before any of them is imported. ``app.py`` imports this module first for exactly
that reason; nothing else should need to touch it.

What it does, once per cold start: copy the committed seed (``demo_data/``, the
28-document pre-built library) into ``/tmp`` and repoint ``KG_DATA_DIR`` at the
copy. Visitors then read a full library and their writes land somewhere writable.
Those writes disappear when the instance goes away, which matches how the demo
already behaved on Render's free tier (restart resets to the seeded baseline).

Inert unless ``VERCEL`` is set, so local runs, Render, Railway and Fly keep using
the directory they were given. Set ``DEMO_SKIP_TMP_SEED=1`` to opt out.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}
_MARKER = ".byowiki-seeded"
_lock = threading.Lock()
_done = False


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def on_vercel() -> bool:
    """Vercel sets VERCEL=1 in both build and runtime environments."""
    return _flag("VERCEL")


def runtime_data_dir() -> str:
    return os.environ.get("DEMO_RUNTIME_DATA_DIR", "/tmp/byowiki-data").strip() \
        or "/tmp/byowiki-data"


def seed_data_dir() -> str:
    """The read-only seed shipped in the bundle — whatever KG_DATA_DIR pointed at
    before we redirected it."""
    return os.environ.get("KG_DATA_DIR", "data").strip() or "data"


def ensure_writable_data_dir() -> str | None:
    """Copy the seed into a writable dir and repoint KG_DATA_DIR at it.

    Returns the new directory, or ``None`` when nothing was done. Idempotent: a
    warm instance re-importing this module finds the marker and skips the copy.
    """
    global _done
    with _lock:
        if _done:
            return os.environ.get("KG_DATA_DIR")
        seed = Path(seed_data_dir()).resolve()
        target = Path(runtime_data_dir())
        if seed == target:  # already pointed at writable storage
            _done = True
            return str(target)
        try:
            target.mkdir(parents=True, exist_ok=True)
            marker = target / _MARKER
            if not marker.exists():
                if seed.is_dir():
                    shutil.copytree(seed, target, dirs_exist_ok=True)
                marker.write_text("", encoding="utf-8")
                print(f"[vercel] seeded writable data dir {target} from {seed}")
        except OSError as exc:
            # Never take the app down over this: without the copy the demo is
            # read-only rather than broken, and the failure needs to be visible.
            print(f"[vercel] could not prepare {target}: {exc}", file=sys.stderr)
            return None
        os.environ["KG_DATA_DIR"] = str(target)
        _done = True
        return str(target)


if on_vercel() and not _flag("DEMO_SKIP_TMP_SEED"):
    ensure_writable_data_dir()
