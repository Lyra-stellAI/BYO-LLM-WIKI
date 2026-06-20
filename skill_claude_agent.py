"""Claude Agent (Claude Code) subprocess backend for skill generation.

An alternative to the in-process LLM pipeline (``skill_agent``): generate an agent
skill by invoking the **Claude Code CLI** (``claude``) as a subprocess in headless
print mode. That makes the generator a *real tool-using agent* — it can read the
selected context from a scratch workspace, author the skill, and write both
``skill.json`` and a human-readable ``SKILL.md`` to disk — rather than a single
chat completion.

This module ONLY changes who writes the skill. The produced artifact is returned
in the same shape ``skill_agent`` uses, so it flows through the exact same
evaluation → accept/reject gate → human-review steps. The subsystem's evaluation
step is therefore backend-independent.

Robustness: the CLI flag surface varies across versions, so the allowed-tools and
permission flags are env-configurable, and the skill is recovered from
``skill.json`` on disk when present *or* parsed from the model's final message —
so generation still succeeds even if the on-disk write was skipped.

Config (env):
  CLAUDE_CODE_BIN            path to the CLI (default ``claude``)
  CLAUDE_CODE_MODEL          model to author with (default ``claude-opus-4-8``)
  CLAUDE_CODE_TIMEOUT        subprocess timeout seconds (default 240)
  CLAUDE_CODE_ALLOWED_TOOLS  comma list passed to --allowed-tools (default Read,Write,Edit,Glob)
  CLAUDE_CODE_PERMISSION_MODE  --permission-mode value (default acceptEdits)
  CLAUDE_CODE_ALLOWED_TOOLS_FLAG  the flag name (default ``--allowed-tools``)
  CLAUDE_CODE_EXTRA_ARGS     extra space-separated CLI args appended verbatim
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

CLAUDE_BIN = os.environ.get("CLAUDE_CODE_BIN", "claude")
CLAUDE_MODEL = os.environ.get("CLAUDE_CODE_MODEL", "claude-opus-4-8")
DEFAULT_TIMEOUT = int(os.environ.get("CLAUDE_CODE_TIMEOUT", "240"))
ALLOWED_TOOLS = os.environ.get("CLAUDE_CODE_ALLOWED_TOOLS", "Read,Write,Edit,Glob")
PERMISSION_MODE = os.environ.get("CLAUDE_CODE_PERMISSION_MODE", "acceptEdits")
ALLOWED_TOOLS_FLAG = os.environ.get("CLAUDE_CODE_ALLOWED_TOOLS_FLAG", "--allowed-tools")
MAX_TURNS = int(os.environ.get("CLAUDE_CODE_MAX_TURNS", "16"))


class ClaudeAgentError(RuntimeError):
    """Raised when the Claude Code subprocess is unavailable or fails."""


def cli_available() -> bool:
    """True when the ``claude`` CLI is on PATH."""
    return shutil.which(CLAUDE_BIN) is not None


def sdk_available() -> bool:
    """True when the Python Agent SDK is importable (informational)."""
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def status() -> dict:
    return {"name": "claude_code", "available": cli_available(), "bin": CLAUDE_BIN,
            "model": CLAUDE_MODEL, "sdk_installed": sdk_available(),
            "allowed_tools": ALLOWED_TOOLS, "permission_mode": PERMISSION_MODE}


# --- prompt + command building (pure; unit-tested) ---------------------------
_REQUIREMENTS = (
    "Author ONE reusable agent skill, grounded ONLY in the provided context (no "
    "invented facts). It must have: a short Title-Case name; a one-sentence "
    "description (the signal an agent uses to decide whether to invoke it); "
    "instructions with at least 2 explicit numbered steps; at least one trigger "
    "and one anti-trigger (negative control); declared tools; and a test set with "
    "at least one positive (should_trigger true) and one negative (should_trigger "
    "false) case. No placeholders (no TODO/TBD/<...>)."
)

_SKILL_JSON_SHAPE = (
    '{"name": "...", "description": "...", "instructions": "... numbered steps ...", '
    '"steps": ["..."], "triggers": ["..."], "anti_triggers": ["..."], "tools": ["..."], '
    '"success_criteria": {"outcome": "...", "process": "...", "style": "...", "efficiency": "..."}, '
    '"tests": [{"prompt": "...", "should_trigger": true, "expect": "..."}, '
    '{"prompt": "...", "should_trigger": false, "expect": "..."}], '
    '"understanding": "what the context is about", "analysis": "why this skill"}'
)


def build_prompt(bundle: dict, *, goal: str = "", revision_note: str = "",
                 allow_tools: bool = True) -> str:
    goal_line = (f"The user specifically wants: \"{goal.strip()}\".\n"
                 if goal and goal.strip() else
                 "Pick the single most useful skill the context supports.\n")
    revision_line = (f"A reviewer asked for changes — address this: \"{revision_note.strip()}\".\n"
                     if revision_note and revision_note.strip() else "")
    if allow_tools:
        body = (
            "You are an agent that authors agent skills. The selected context is in "
            "the file ./context.md in your working directory. Read it, then "
            f"{_REQUIREMENTS}\n\n{goal_line}{revision_line}"
            "Use your tools: read ./context.md, then WRITE two files in the working "
            "directory — ./skill.json (the skill as a JSON object) and ./SKILL.md "
            "(a readable version with YAML front-matter: name, description). "
            f"The skill.json object must have exactly these keys: {_SKILL_JSON_SHAPE}\n\n"
            "After writing the files, print the final skill JSON object as your last message."
        )
    else:
        body = (
            "You author agent skills. " + _REQUIREMENTS + "\n\n" + goal_line + revision_line +
            "SELECTED CONTEXT:\n" + bundle.get("context", "")[:12000] + "\n\n"
            f"Return ONLY the skill as a JSON object with these keys: {_SKILL_JSON_SHAPE}"
        )
    return body


def build_command(model: str | None, *, allow_tools: bool = True,
                  output_format: str = "json") -> list[str]:
    cmd = [CLAUDE_BIN, "-p", "--output-format", output_format, "--max-turns", str(MAX_TURNS)]
    if model:
        cmd += ["--model", model]
    if allow_tools:
        cmd += [ALLOWED_TOOLS_FLAG, ALLOWED_TOOLS, "--permission-mode", PERMISSION_MODE]
    extra = os.environ.get("CLAUDE_CODE_EXTRA_ARGS", "").strip()
    if extra:
        cmd += extra.split()
    return cmd


# --- output parsing (pure; unit-tested) --------------------------------------
def parse_envelope(stdout: str) -> dict:
    """Normalize the CLI's ``--output-format json`` envelope (or raw text)."""
    stdout = (stdout or "").strip()
    if not stdout:
        return {"result": "", "tokens": None, "cost_usd": None, "num_turns": None, "is_error": True}
    try:
        obj = json.loads(stdout)
    except Exception:  # noqa: BLE001  (text output-format, or noise) -> treat as result text
        return {"result": stdout, "tokens": None, "cost_usd": None, "num_turns": None, "is_error": False}
    if isinstance(obj, list):  # stream-json: take the terminal result event
        obj = next((e for e in reversed(obj) if isinstance(e, dict) and e.get("type") == "result"),
                   obj[-1] if obj else {})
    if not isinstance(obj, dict):
        return {"result": str(obj), "tokens": None, "cost_usd": None, "num_turns": None, "is_error": False}
    usage = obj.get("usage") or {}
    tokens = None
    if isinstance(usage, dict):
        tokens = (int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))) or None
    return {
        "result": obj.get("result") or obj.get("text") or "",
        "tokens": tokens or obj.get("total_tokens"),
        "cost_usd": obj.get("total_cost_usd"),
        "num_turns": obj.get("num_turns"),
        "is_error": bool(obj.get("is_error")),
        "session_id": obj.get("session_id"),
    }


def extract_skill(text: str) -> dict | None:
    """Pull a skill JSON object out of the model's final message text."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def read_workspace_skill(workspace: str | Path) -> dict | None:
    p = Path(workspace) / "skill.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _default_runner(cmd: list[str], prompt: str, workspace: str, timeout: int) -> dict:
    proc = subprocess.run(  # noqa: S603  (cmd is built from a fixed binary + flags)
        cmd, input=prompt, cwd=workspace, env=os.environ.copy(),
        capture_output=True, text=True, timeout=timeout)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def generate(bundle: dict, *, goal: str = "", revision_note: str = "", model: str | None = None,
             timeout: int | None = None, allow_tools: bool = True, _runner=None) -> dict:
    """Generate a skill by invoking the Claude Code CLI as a subprocess.

    Returns ``{"artifact": <skill dict>, "meta": {...}, "workspace": <dir>}``. The
    artifact is recovered from ``skill.json`` on disk when written, else parsed from
    the model's final message. Raises ``ClaudeAgentError`` on failure."""
    runner = _runner or _default_runner
    if runner is _default_runner and not cli_available():
        raise ClaudeAgentError(
            f"Claude Code CLI ('{CLAUDE_BIN}') was not found on PATH. Install it "
            "(npm i -g @anthropic-ai/claude-code) and authenticate it, or set "
            "CLAUDE_CODE_BIN — or use the in-process pipeline backend instead.")
    model = model or CLAUDE_MODEL
    workspace = tempfile.mkdtemp(prefix="skill_cc_")
    try:
        (Path(workspace) / "context.md").write_text(bundle.get("context", ""), encoding="utf-8")
        prompt = build_prompt(bundle, goal=goal, revision_note=revision_note, allow_tools=allow_tools)
        cmd = build_command(model, allow_tools=allow_tools)
        t0 = time.perf_counter()
        try:
            run = runner(cmd, prompt, workspace, timeout or DEFAULT_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise ClaudeAgentError(f"Claude Code timed out after {timeout or DEFAULT_TIMEOUT}s.") from exc
        except FileNotFoundError as exc:
            raise ClaudeAgentError(f"Could not execute '{CLAUDE_BIN}': {exc}") from exc
        duration_ms = int((time.perf_counter() - t0) * 1000)

        if run.get("returncode") not in (0, None):
            raise ClaudeAgentError(
                f"Claude Code exited {run.get('returncode')}: {(run.get('stderr') or '')[:300]}")
        env = parse_envelope(run.get("stdout", ""))
        artifact = read_workspace_skill(workspace) or extract_skill(env.get("result", ""))
        if not artifact:
            raise ClaudeAgentError(
                "Claude Code did not return a parseable skill (no skill.json and no JSON "
                f"in the final message). stderr: {(run.get('stderr') or '')[:200]}")
        skill_md_path = Path(workspace) / "SKILL.md"
        meta = {
            "backend": "claude_code", "model": model, "duration_ms": duration_ms,
            "tokens": env.get("tokens"), "cost_usd": env.get("cost_usd"),
            "num_turns": env.get("num_turns"), "session_id": env.get("session_id"),
            "understanding": artifact.pop("understanding", None),
            "analysis": artifact.pop("analysis", None),
            "skill_md": skill_md_path.read_text(encoding="utf-8") if skill_md_path.exists() else None,
            "wrote_skill_json": (Path(workspace) / "skill.json").exists(),
        }
        return {"artifact": artifact, "meta": meta, "workspace": workspace}
    finally:
        # Best-effort cleanup of the scratch workspace (the skill is persisted in
        # the library + data/skills already).
        try:
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
