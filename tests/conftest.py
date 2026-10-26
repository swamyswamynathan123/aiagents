"""
Test setup: offline by default. Dummy API keys, tracing off, fake LLM/embeddings/
search installed before any agent module loads, and the moderation API stubbed.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "evals"), str(Path(__file__).resolve().parent)]

# Override (not setdefault) so a developer's real .env never makes tests hit the network.
# Keys are set to empty rather than removed: the agents' load_dotenv() won't overwrite an
# existing variable, but would re-add a missing one from .env.
os.environ.update({
    "OPENAI_API_KEY": "sk-test", "TAVILY_API_KEY": "tvly-test",
    "LANGCHAIN_API_KEY": "", "LANGSMITH_API_KEY": "",
    "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false",
})

import fakes  # noqa: E402

fakes.install()

import guardrails  # noqa: E402
import registry  # noqa: E402

FRAMEWORKS = ["LangGraph", "LangChain"]
AGENT_NUMS = [f"{n:02d}" for n in range(1, 17)]


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch):
    fakes.script.reset()
    fakes.FakeTavily.queries.clear()
    monkeypatch.setattr(guardrails, "moderate", lambda text: None)  # no network


@pytest.fixture(params=FRAMEWORKS)
def framework(request):
    return request.param


def load(framework: str, num: str):
    """Fresh module instance per call, so in-memory state never leaks between tests."""
    return registry.load_module(registry.module_path(framework, num))


def runner(framework: str, num: str):
    m = load(framework, num)
    return m, registry.make_runner(framework, num, m)
