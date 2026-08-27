"""
12 - Code Execution Agent
Pattern: LLM generates Python code; a sandboxed tool executes it via
         subprocess; output feeds back to the LLM for interpretation.
Use cases: Data analysis assistants, automated scripting, computational
           problem solving, REPL-style AI coding environments.

Safety: The execute_python tool blocks dangerous patterns before execution.
        Only arithmetic, string ops, and standard library math/statistics
        are allowed in this demo.

LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import ast
import os
import subprocess
import sys
import tempfile
from typing import Annotated, List
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

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

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CodeExecState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    last_code: str
    last_output: str
    execution_count: int


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)


def code_agent_node(state: CodeExecState) -> dict:
    resp = _llm.invoke([
        SystemMessage(content=(
            "You are an expert Python developer. When asked to compute something:\n"
            "1. Write clean, correct Python code using the execute_python tool.\n"
            "2. After seeing the output, interpret and explain the results clearly.\n"
            "3. Fix any errors and retry if needed.\n"
            "Use only pure Python — no imports of os, sys, or network libraries."
        )),
        *state["messages"],
    ])
    return {"messages": [resp]}


def tool_exec_node(state: CodeExecState) -> dict:
    last = state["messages"][-1]
    results: List[ToolMessage] = []
    last_code = state["last_code"]
    last_output = state["last_output"]

    for tc in last.tool_calls:
        code = tc["args"].get("code", "")
        output = TOOL_MAP[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
        last_code = code
        last_output = str(output)

    return {
        "messages": results,
        "last_code": last_code,
        "last_output": last_output,
        "execution_count": state["execution_count"] + 1,
    }


def code_router(state: CodeExecState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        if state["execution_count"] < 5:  # safety cap
            return "executor"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(CodeExecState)
_builder.add_node("agent", code_agent_node)
_builder.add_node("executor", tool_exec_node)
_builder.set_entry_point("agent")
_builder.add_conditional_edges("agent", code_router, {"executor": "executor", END: END})
_builder.add_edge("executor", "agent")
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_task(task: str) -> None:
    print(f"Task: {task}")
    result = app.invoke({
        "messages": [HumanMessage(content=task)],
        "last_code": "",
        "last_output": "",
        "execution_count": 0,
    })
    print(f"Executions: {result['execution_count']}")
    if result["last_code"]:
        print(f"Last code:\n{result['last_code']}")
    if result["last_output"]:
        print(f"Output: {result['last_output']}")
    print(f"Agent: {result['messages'][-1].content[:300]}\n")


def main():
    print("=== 12 Code Execution Agent ===\n")

    tasks = [
        "Generate and print the first 10 Fibonacci numbers.",
        "Calculate the sum of squares of all even numbers from 1 to 20.",
        "Print a simple multiplication table for numbers 1 through 5.",
    ]

    for task in tasks:
        run_task(task)


if __name__ == "__main__":
    main()
