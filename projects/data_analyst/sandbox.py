"""
Sandboxed execution of model-written pandas code.

Stricter than module 12's blocklist, but still a demo-grade sandbox (see guide.md 7.12):
  - allow-list of imports (anything not listed is rejected), blocked calls, and no dunder
    attribute access, checked on the AST before running
  - each run is a fresh `python -I` subprocess: isolated mode, a working directory holding only
    a copy of the CSV, and a stripped environment, so API keys from .env never reach model code
  - timeout, truncated output, and charts saved only via the save_chart() helper
Runs behave like notebook cells: each new run first replays the earlier successful runs (output and
charts suppressed), so variables carry over the way the model expects, while every run is still a
fresh isolated process that starts from the original CSV.
"""
import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "statistics", "datetime", "calendar", "collections", "itertools",
    "functools", "operator", "re", "json", "decimal", "fractions", "string", "textwrap", "matplotlib",
}
BLOCKED_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "help",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
}
BLOCKED_ATTRS = {  # pandas/numpy entry points to files, pickles, databases, clipboard or the shell
    "read_pickle", "to_pickle", "read_sql", "read_sql_query", "read_sql_table", "to_sql", "read_clipboard",
    "to_clipboard", "read_html", "read_xml", "read_json", "read_parquet", "to_parquet", "read_excel",
    "to_excel", "to_csv", "read_csv", "read_table", "read_fwf", "read_hdf", "to_hdf", "read_feather",
    "to_feather", "load", "save", "savez", "savetxt", "loadtxt", "fromfile", "tofile", "memmap",
    "system", "popen", "savefig",
}
TIMEOUT_S = 20
MAX_OUTPUT_CHARS = 4000
MPL_CACHE = Path(tempfile.gettempdir()) / "analyst-mplconfig"


def check_code(code: str) -> Optional[str]:
    """Return a reason if the code is not allowed, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"Import of '{name}' is not allowed (allowed: {', '.join(sorted(ALLOWED_IMPORTS))})"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            return f"Call to '{node.func.id}()' is not allowed"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return f"Access to '{node.attr}' is not allowed"
            if node.attr in BLOCKED_ATTRS:
                return f"'{node.attr}' is not allowed: the data is already loaded as `df`; use save_chart() for plots"
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            return f"Access to '{node.id}' is not allowed"
    return None


def _preamble(run_id: str, history: List[str]) -> str:
    return f'''
import contextlib as _ctx, io as _io, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
pd.set_option("display.width", 120); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 60)
df = pd.read_csv("data.csv")
_n_charts, _replaying = 0, False
def save_chart(title=""):
    """Save the current matplotlib figure so it is shown to the user."""
    global _n_charts
    if _replaying:  # charts from earlier cells were already saved
        plt.close("all")
        return
    _n_charts += 1
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig("chart_{run_id}_" + str(_n_charts) + ".png", dpi=110)
    plt.close("all")
_replaying = True
for _cell in {history!r}:  # earlier successful cells, already checked: replay quietly
    try:
        with _ctx.redirect_stdout(_io.StringIO()):
            exec(compile(_cell, "<previous cell>", "exec"), globals())
    except Exception:
        pass
_replaying = False
plt.close("all")
'''


def display_last_expression(code: str) -> str:
    """Like a notebook: if the cell ends with a bare expression (e.g. `totals`), print it."""
    tree = ast.parse(code)
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr) and not (isinstance(last.value, ast.Call)
                                           and getattr(last.value.func, "id", "") in ("print", "save_chart")):
        tree.body[-1] = ast.Expr(ast.Call(ast.Name("print", ast.Load()), [last.value], []))
        return ast.unparse(ast.fix_missing_locations(tree))
    return code


def _clean_env(workdir: Path) -> dict:
    import os

    # System locations only (no secrets): numpy needs SYSTEMROOT, and without SYSTEMDRIVE/PROGRAMDATA
    # Windows components write caches to a literal "%SystemDrive%" folder in the working directory.
    keep = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PROGRAMDATA", "PATH", "TEMP", "TMP")
    env = {k: v for k, v in os.environ.items() if k.upper() in keep}
    # Shared matplotlib cache: a fresh one per workdir rebuilds the font cache (seconds) on first plot
    MPL_CACHE.mkdir(exist_ok=True)
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(MPL_CACHE), PYTHONIOENCODING="utf-8")
    return env


def _execute(code: str, workdir: Path, run_id: str, timeout: int = TIMEOUT_S, history: Optional[List[str]] = None) -> dict:
    """Run code after the preamble. Internal: callers must check_code() first."""
    script = workdir / f"run_{run_id}.py"
    # Like a notebook, figures left open at the end of the cell are shown (saved) automatically
    epilogue = "\nfor _num in plt.get_fignums():\n    plt.figure(_num)\n    save_chart()\n"
    script.write_text(_preamble(run_id, history or []) + "\n" + display_last_expression(code) + epilogue,
                      encoding="utf-8")
    before = set(workdir.glob("chart_*.png"))
    try:
        proc = subprocess.run([sys.executable, "-I", script.name], cwd=workdir, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, env=_clean_env(workdir))
        output, ok = proc.stdout, proc.returncode == 0
        if proc.stderr.strip():
            # keep the useful tail of a traceback, minus the preamble's line numbers
            output += "\n[error]\n" + "\n".join(proc.stderr.strip().splitlines()[-6:])
    except subprocess.TimeoutExpired:
        output, ok = f"[error] Execution timed out after {timeout}s", False
    finally:
        script.unlink(missing_ok=True)
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + f"\n... [output truncated at {MAX_OUTPUT_CHARS} chars; print less]"
    if "NameError" in output:
        output += ("\n[hint] Variables persist only from runs that succeeded. Define the missing name in this "
                   "run, or re-run the code that creates it.")
    charts = sorted(str(p) for p in set(workdir.glob("chart_*.png")) - before)
    return {"ok": ok, "output": output.strip() or "(no output: use print())", "charts": charts}


def run(code: str, workdir: Path, run_id: str, history: Optional[List[str]] = None) -> dict:
    """Check and run one cell. `history` = earlier successful cells to replay first."""
    reason = check_code(code)
    if reason:
        return {"ok": False, "output": f"[blocked] {reason}", "charts": [], "blocked": True}
    return {**_execute(code, workdir, run_id, history=history), "blocked": False}


def make_workdir(csv_path: Path) -> Path:
    """A fresh directory containing only data.csv."""
    workdir = Path(tempfile.mkdtemp(prefix="analyst-"))
    (workdir / "data.csv").write_bytes(Path(csv_path).read_bytes())
    return workdir


def chart_files(workdir: Path) -> List[str]:
    return sorted(str(p) for p in workdir.glob("chart_*.png"))
