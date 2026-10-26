"""
12 - Code Execution Agent (LangChain)
Pattern: LLM writes Python via a tool call; a sandboxed tool runs it in a
         subprocess; a Python loop feeds output back until the LLM answers.
Use cases: Data analysis assistants, math/computation helpers, code
           generation with verification, notebook-style agents.
LangGraph equivalent: ../12_code_execution_agent.py (agent ⇄ executor graph)
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env

SECURITY: The AST filter below is a demo safeguard, not a real sandbox.
Use containers or a hosted code interpreter in production.
"""
import ast
import os
import subprocess
import sys
import tempfile
from typing import List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "socket", "requests", "urllib",
    "http", "ftplib", "smtplib", "shutil", "pathlib", "io",
    "builtins", "importlib", "ctypes", "multiprocessing", "threading",
}
BLOCKED_CALLS = {
    "eval", "exec", "compile", "__import__", "open",
    "globals", "locals", "getattr", "setattr", "delattr", "vars",
}


def _is_safe_code(code: str):
    """Return (True, '') if safe, or (False, reason) if blocked."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BLOCKED_MODULES:
                    return False, f"Blocked import: '{top}'"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in BLOCKED_MODULES:
                return False, f"Blocked import: '{top}'"
        elif isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in BLOCKED_CALLS:
                return False, f"Blocked call: '{func_name}()'"
    return True, ""


@tool
def execute_python(code: str) -> str:
    """
    Execute Python code and return stdout + stderr output.
    Only pure computation is allowed — no file I/O, network, or OS access.
    """
    safe, reason = _is_safe_code(code)
    if not safe:
        return f"BLOCKED: {reason}. Write pure computational code only."
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=tempfile.gettempdir(),
        )
        output = proc.stdout
        if proc.stderr:
            output += f"\n[stderr]: {proc.stderr}"
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Execution timed out (10s limit)"
    except Exception as exc:
        return f"ERROR: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


TOOLS = [execute_python]
TOOL_MAP = {t.name: t for t in TOOLS}

_llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "You are an expert Python developer. When asked to compute something:\n"
    "1. Write clean, correct Python code using the execute_python tool.\n"
    "2. After seeing the output, interpret and explain the results clearly.\n"
    "3. Fix any errors and retry if needed.\n"
    "Use only pure Python — no imports of os, sys, or network libraries."
)

# ---------------------------------------------------------------------------
# Write → execute → interpret loop
# ---------------------------------------------------------------------------

MAX_EXECUTIONS = 5  # safety cap


def run(task: str) -> dict:
    messages: List[BaseMessage] = [HumanMessage(content=task)]
    last_code, last_output, execution_count = "", "", 0

    while True:
        response = _llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
        messages.append(response)
        if not response.tool_calls or execution_count >= MAX_EXECUTIONS:
            break
        for tc in response.tool_calls:
            output = TOOL_MAP[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"], name=tc["name"]))
            last_code, last_output = tc["args"].get("code", ""), str(output)
        execution_count += 1

    return {"messages": messages, "last_code": last_code, "last_output": last_output, "execution_count": execution_count}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_task(task: str) -> None:
    print(f"Task: {task}")
    result = run(task)
    print(f"Executions: {result['execution_count']}")
    if result["last_code"]:
        print(f"Last code:\n{result['last_code']}")
    if result["last_output"]:
        print(f"Output: {result['last_output']}")
    print(f"Agent: {result['messages'][-1].content[:300]}\n")


def main():
    print("=== 12 Code Execution Agent (LangChain) ===\n")

    tasks = [
        "Generate and print the first 10 Fibonacci numbers.",
        "Calculate the sum of squares of all even numbers from 1 to 20.",
        "Print a simple multiplication table for numbers 1 through 5.",
    ]

    for task in tasks:
        run_task(task)


if __name__ == "__main__":
    main()
