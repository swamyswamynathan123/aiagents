"""
The agent's tools, as plain functions over a workspace directory. The graph's executor calls these
and enforces the rules the model can't be trusted with:

- every path must resolve inside the workspace (no `../`, no absolute paths)
- test files are read-only: "fixing" a failure by editing the test is rejected
- edits are exact text replacements that must match exactly once
- edits that introduce shell, network or filesystem-deletion calls are rejected
- tests run in a fresh `python -I` subprocess with a stripped environment (no API keys) and a timeout

Like the data analyst's sandbox, this is demo-grade isolation. Run untrusted code in a container.
"""
import difflib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

TEST_TIMEOUT_S = 60
MAX_OUTPUT = 3500
READ_LIMIT = 250  # lines per read_file call
RISKY = re.compile(r"\b(import\s+(subprocess|socket|shutil|requests|urllib|ctypes)|os\.(system|popen|remove|unlink|rmdir)"
                   r"|shutil\.|subprocess\.|eval\(|exec\(|__import__)")


class ToolError(Exception):
    """A rule violation or bad request, reported back to the model instead of crashing."""


def resolve(workspace: Path, rel: str) -> Path:
    path = (workspace / rel).resolve()
    if workspace.resolve() not in path.parents and path != workspace.resolve():
        raise ToolError(f"'{rel}' is outside the repository.")
    return path


def is_test_file(rel: str) -> bool:
    parts = Path(rel).parts
    return "tests" in parts or Path(rel).name.startswith("test_") or Path(rel).name == "conftest.py"


def list_files(workspace: Path) -> str:
    files = sorted(str(p.relative_to(workspace)).replace("\\", "/") for p in workspace.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("."))
    return "\n".join(files)


def read_file(workspace: Path, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
    target = resolve(workspace, path)
    if not target.is_file():
        raise ToolError(f"No file '{path}'.")
    lines = target.read_text(encoding="utf-8").splitlines()
    start = max(start_line, 1)
    end = min(end_line or start + READ_LIMIT - 1, start + READ_LIMIT - 1, len(lines))
    numbered = "\n".join(f"{n:4} | {lines[n - 1]}" for n in range(start, end + 1))
    more = f"\n... ({len(lines) - end} more lines)" if end < len(lines) else ""
    return (numbered or "(empty file)") + more


def search_code(workspace: Path, query: str) -> str:
    hits = []
    for p in sorted(workspace.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if query.lower() in line.lower():
                hits.append(f"{str(p.relative_to(workspace)).replace(chr(92), '/')}:{n}: {line.strip()}")
    return "\n".join(hits[:40]) or f"No matches for '{query}'."


def edit_file(workspace: Path, path: str, old_text: str, new_text: str) -> str:
    """Replace one exact occurrence of old_text. Returns the previous file content (for rollback)."""
    if is_test_file(path):
        raise ToolError("Test files are read-only. Fix the code under test, not the tests.")
    if RISKY.search(new_text):
        raise ToolError("The edit adds shell, network or file-deletion calls, which aren't allowed here.")
    target = resolve(workspace, path)
    if not target.is_file():
        raise ToolError(f"No file '{path}'.")
    text = target.read_text(encoding="utf-8")
    count = text.count(old_text) if old_text else 0
    if count == 0:
        raise ToolError("old_text was not found. Read the file again and copy the exact text, including indentation.")
    if count > 1:
        raise ToolError(f"old_text appears {count} times. Include more surrounding lines so it is unique.")
    target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8", newline="\n")
    return text


def excerpt(workspace: Path, path: str, new_text: str, context: int = 3) -> str:
    """The edited region with line numbers, so the model sees the file as it now is."""
    text = resolve(workspace, path).read_text(encoding="utf-8")
    at = text.find(new_text) if new_text else -1
    if at < 0:
        return ""
    first = text[:at].count("\n") + 1
    last = first + new_text.count("\n")
    lines = text.splitlines()
    lo, hi = max(1, first - context), min(len(lines), last + context)
    return f"{path} lines {lo}-{hi} now:\n" + "\n".join(f"{n:4} | {lines[n - 1]}" for n in range(lo, hi + 1))


def _clean_env() -> dict:
    # System locations only (no secrets). Without SYSTEMDRIVE/PROGRAMDATA, Windows components write
    # caches to a literal "%SystemDrive%" folder in the working directory.
    keep = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PROGRAMDATA", "PATH", "TEMP", "TMP")
    env = {k: v for k, v in os.environ.items() if k.upper() in keep}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_tests(workspace: Path, target: Optional[str] = None) -> Dict:
    """Run pytest. Returns counts, failing test ids and (truncated) output."""
    if target:
        resolve(workspace, target.split("::")[0])  # same path rules
    cmd = [sys.executable, "-I", "-m", "pytest", "-rf", "--tb=short", *([target] if target else [])]
    try:
        proc = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=TEST_TIMEOUT_S, env=_clean_env())
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {"passed": 0, "failed": 0, "errors": 1, "failing": [], "ok": False,
                "output": f"Tests timed out after {TEST_TIMEOUT_S}s (an infinite loop?)."}
    count = lambda word: int(m.group(1)) if (m := re.search(rf"(\d+) {word}", output)) else 0  # noqa: E731
    failing = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", output, re.M)))
    result = {"passed": count("passed"), "failed": count("failed"), "errors": count("error"), "failing": failing}
    result["ok"] = result["failed"] == 0 and result["errors"] == 0 and result["passed"] > 0
    result["output"] = output if len(output) <= MAX_OUTPUT else output[:MAX_OUTPUT] + "\n... [truncated]"
    return result


def snapshot(workspace: Path) -> Dict[str, str]:
    return {str(p.relative_to(workspace)).replace("\\", "/"): p.read_text(encoding="utf-8")
            for p in sorted(workspace.rglob("*.py")) if "__pycache__" not in p.parts}


def unified_diff(before: Dict[str, str], after: Dict[str, str]) -> str:
    chunks = []
    for path in sorted(set(before) | set(after)):
        a, b = before.get(path, ""), after.get(path, "")
        if a != b:
            chunks.append("".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                                       f"a/{path}", f"b/{path}")))
    return "\n".join(chunks)


def changed_lines(diff: str) -> int:
    return sum(1 for line in diff.splitlines()
               if line[:1] in "+-" and not line.startswith(("+++", "---")))


def failing_count(result: Dict) -> int:
    return result["failed"] + result["errors"] + (0 if result["passed"] or result["failed"] else 1)


def list_test_names(result: Dict) -> List[str]:
    return result["failing"]
