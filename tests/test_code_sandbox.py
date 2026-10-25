"""Agent 12's code filter and executor (both frameworks share the same implementation)."""
import pytest

from conftest import load


@pytest.fixture
def sandbox(framework):
    return load(framework, "12")


@pytest.mark.parametrize("code", [
    "import os",
    "import subprocess",
    "from os import path",
    "import urllib.request",
    "open('secrets.txt')",
    "eval('1+1')",
    "exec('print(1)')",
    "__import__('os')",
    "getattr(object, '__subclasses__')",
    "x = (1).__class__; globals()",
])
def test_dangerous_code_blocked(sandbox, code):
    safe, reason = sandbox._is_safe_code(code)
    assert not safe and reason
    assert sandbox.execute_python.invoke({"code": code}).startswith("BLOCKED")


@pytest.mark.parametrize("code", [
    "print(sum(i * i for i in range(1, 11)))",
    "import math\nprint(math.sqrt(144))",
    "from collections import Counter\nprint(Counter('aab'))",
])
def test_safe_code_allowed(sandbox, code):
    assert sandbox._is_safe_code(code) == (True, "")


def test_syntax_error_reported(sandbox):
    safe, reason = sandbox._is_safe_code("def broken(:")
    assert not safe and reason.startswith("Syntax error")


def test_executes_and_captures_output(sandbox):
    assert sandbox.execute_python.invoke({"code": "print(sum(i * i for i in range(1, 11)))"}) == "385"


def test_captures_stderr(sandbox):
    out = sandbox.execute_python.invoke({"code": "print(1 / 0)"})
    assert "[stderr]" in out and "ZeroDivisionError" in out


def test_no_output(sandbox):
    assert sandbox.execute_python.invoke({"code": "x = 1"}) == "(no output)"
