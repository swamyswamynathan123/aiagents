"""
Agent control flow, run offline against the scripted fake LLM.
Every test runs against both the LangGraph and LangChain versions (`framework` fixture).
"""
import sys

import pytest
from langchain_core.messages import SystemMessage, ToolMessage

import registry
from cases import SUITES
from conftest import AGENT_NUMS, load, runner
from fakes import FakeTavily, script


def _tool_names(messages):
    return [m.name for m in messages if isinstance(m, ToolMessage)]


def _last_system_prompt():
    return "\n".join(m.content for m in script.prompts[-1] if isinstance(m, SystemMessage))


# ---------------------------------------------------------------------------
# Loading and wiring
# ---------------------------------------------------------------------------

def test_both_frameworks_have_the_same_16_modules():
    names = registry.module_files("LangGraph")
    assert len(names) == 16
    assert names == registry.module_files("LangChain")


@pytest.mark.parametrize("num", AGENT_NUMS)
def test_module_loads_and_has_demo(framework, num):
    assert callable(load(framework, num).main)


@pytest.mark.parametrize("num", AGENT_NUMS)
def test_agent_model_comes_from_env(framework, num, monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "some-other-model")
    m = load(framework, num)
    assert m.MODEL == "some-other-model"
    models = {getattr(v, "model", None) or getattr(getattr(v, "bound", None), "model", None)
              for v in vars(m).values() if "Chat" in type(v).__name__ or hasattr(v, "bound")}
    assert models - {None} <= {"some-other-model"}  # every model object built from MODEL


def test_no_hardcoded_model_names():
    from conftest import ROOT

    files = [*ROOT.glob("[0-9][0-9]_*.py"), *ROOT.glob("langchain/*.py"), *ROOT.glob("projects/*/*.py"),
             ROOT / "guardrails.py", ROOT / "evals" / "run_evals.py"]
    offenders = [f.name for f in files if 'ChatOpenAI(model="' in f.read_text(encoding="utf-8")]
    assert offenders == []


@pytest.mark.parametrize("num", AGENT_NUMS)
def test_langchain_version_does_not_import_langgraph(monkeypatch, num):
    for name in [k for k in sys.modules if k == "langgraph" or k.startswith("langgraph.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "langgraph", None)  # any import now raises ImportError
    load("LangChain", num)


@pytest.mark.parametrize("num", AGENT_NUMS)
def test_eval_cases_run_offline(framework, num):
    """Registry adapters + eval normalizers line up with each agent's real output."""
    m = load(framework, num)
    ctx = {"framework": framework, "module": m, "run": registry.make_runner(framework, num, m)}
    for example in SUITES[num]["examples"]:
        out = SUITES[num]["call"](ctx, example["inputs"])
        assert set(out) == {"answer", "tools", "fields"}
        expected = {**example["reference"].get("fields", {}), **example["reference"].get("min_fields", {}),
                    **example["reference"].get("max_fields", {})}
        assert set(expected) <= set(out["fields"]), f"eval checks fields the agent doesn't produce: {expected}"


# ---------------------------------------------------------------------------
# Tool loops (01, 02, 10, 12)
# ---------------------------------------------------------------------------

def test_01_runs_tools_then_answers(framework):
    _, run = runner(framework, "01")
    result = run(query="Weather in Paris and 42 * 7?")
    assert _tool_names(result["messages"]) == ["get_weather", "calculate"]
    assert result["messages"][-1].content == "FAKE answer after 2 tool calls"
    assert FakeTavily.queries[0] == {"query": "Paris weather today temperature", "time_range": "day"}


def test_01_calculator_rejects_non_arithmetic(framework):
    m = load(framework, "01")
    assert m.calculate.invoke({"expression": "6 * 7"}) == "42"
    assert m.calculate.invoke({"expression": "__import__('os')"}).startswith("Error")


@pytest.mark.parametrize("num", ["01", "02"])
def test_langchain_tool_loops_are_capped(num):
    script.tool_calls = 10_000  # a model that never stops calling tools
    _, run = runner("LangChain", num)
    result = run(query="loop forever")
    assert len(_tool_names(result["messages"])) == load("LangChain", num).MAX_STEPS


def test_02_uses_finance_and_news_topics(framework):
    script.tool_order = ["get_stock_price", "get_news_headlines"]
    _, run = runner(framework, "02")
    run(query="AAPL price and AI news")
    assert [q.get("topic") for q in FakeTavily.queries] == ["finance", "news"]


def test_10_stops_at_max_iterations_without_completion(framework):
    script.tool_calls, script.tool_order = 10_000, ["research_topic"]
    _, run = runner(framework, "10")
    result = run(goal="never finishes", max_iterations=4)
    assert result["iteration_count"] == 4
    assert result["completed"] is False


def test_10_records_notes_and_stops_on_completion(framework):
    script.tool_calls, script.tool_order = 3, ["research_topic", "save_note", "mark_goal_complete"]
    _, run = runner(framework, "10")
    result = run(goal="research then finish", max_iterations=6)
    assert result["completed"] is True
    assert result["progress_notes"] == ["qubits scale"]
    assert result["iteration_count"] < 6


def test_12_records_code_and_output(framework):
    script.tool_calls = 1
    _, run = runner(framework, "12")
    result = run(task="sum 0..4")
    assert result["last_code"] == "print(sum(range(5)))"
    assert result["last_output"] == "10"
    assert result["execution_count"] == 1


def test_12_execution_is_capped(framework):
    script.tool_calls = 10_000
    _, run = runner(framework, "12")
    assert run(task="loop forever")["execution_count"] <= 5


# ---------------------------------------------------------------------------
# Routing and multi-agent (03, 08, 09)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["math", "writing", "general"])
def test_03_dispatches_to_chosen_specialist(framework, route):
    script.structured["RouteDecision"] = {"route": route, "reason": "test"}
    _, run = runner(framework, "03")
    result = run(query="anything")
    assert result["route"] == route
    expected = {"math": "math tutor", "writing": "creative writing", "general": "general assistant"}[route]
    assert expected in _last_system_prompt()


def test_08_supervisor_runs_researcher_then_writer(framework):
    _, run = runner(framework, "08")
    result = run(objective="research and write")
    assert result["research_result"] and result["written_result"]
    assert result["iteration"] == 2


def test_09_hands_off_to_specialist(framework):
    script.reply = lambda system, human: "HANDOFF:billing" if "triage agent" in system else "Refund issued."
    _, run = runner(framework, "09")
    result = run(query="charged twice")
    assert (result["active_agent"], result["handoff_count"]) == ("billing", 1)
    assert result["messages"][-1].content == "[Billing Agent] Refund issued."


def test_09_handoff_ping_pong_terminates(framework):
    """Billing and technical keep bouncing the customer; the loop must still end."""
    script.reply = lambda system, human: "HANDOFF:technical" if "billing specialist" in system else "HANDOFF:billing"
    _, run = runner(framework, "09")
    result = run(query="ambiguous issue")
    assert result["handoff_count"] <= 3
    assert script.calls <= 5


# ---------------------------------------------------------------------------
# Iterative loops (04, 05, 15)
# ---------------------------------------------------------------------------

def test_04_plans_executes_and_finishes(framework):
    _, run = runner(framework, "04")
    result = run(objective="three tips")
    assert result["final_answer"] == "FAKE final answer"
    assert len(result["completed_steps"]) == 2  # one per planned step


def test_04_step_budget_forces_an_answer(framework):
    script.structured["ReplanDecision"] = {"is_complete": False, "revised_steps": ["more research"]}
    _, run = runner(framework, "04")
    result = run(objective="never satisfied")
    assert result["final_answer"]  # falls back to the step results
    assert len(result["completed_steps"]) <= 11


def test_04_empty_replan_still_answers(framework):
    script.structured["ReplanDecision"] = {"is_complete": False, "revised_steps": []}
    _, run = runner(framework, "04")
    result = run(objective="nothing left to do")
    assert result["final_answer"]


@pytest.mark.parametrize("approved, expected_iterations", [(True, 1), (False, 3)])
def test_05_reflection_loop(framework, approved, expected_iterations):
    script.structured["CritiqueResult"] = {"approved": approved, "score": 9 if approved else 5, "feedback": "fb"}
    _, run = runner(framework, "05")
    result = run(task="pitch", max_iterations=3)
    assert (result["iteration"], result["approved"]) == (expected_iterations, approved)


@pytest.mark.parametrize("passed, expected_iterations", [(True, 1), (False, 3)])
def test_15_optimizer_loop(framework, passed, expected_iterations):
    script.structured["EvalResult"].update(passed=passed, score=9 if passed else 5)
    _, run = runner(framework, "15")
    result = run(task="headphones", target_score=8, max_iterations=3)
    assert result["iteration"] == expected_iterations
    assert len(result["score_history"]) == expected_iterations


# ---------------------------------------------------------------------------
# Memory and chat (06, 16)
# ---------------------------------------------------------------------------

def test_06_long_term_facts_reach_a_new_thread(framework):
    m = load(framework, "06")
    m.chat("u1", "thread-a", "Hi, my name is Priya and I love hiking.")
    m.chat("u1", "thread-b", "Do you remember me?")
    assert "User's name is Priya" in _last_system_prompt()
    m.chat("u2", "thread-c", "Hello")
    assert "Priya" not in _last_system_prompt()  # facts are per user


def test_06_and_16_keep_thread_history(framework):
    m6, m16 = load(framework, "06"), load(framework, "16")
    m6.chat("u1", "t1", "first message")
    m6.chat("u1", "t1", "second message")
    assert "first message" in [msg.content for msg in script.prompts[-1]]

    m16.chat("remember the word banana", thread_id="t2")
    m16.chat("what was the word?", thread_id="t2")
    assert "remember the word banana" in [msg.content for msg in script.prompts[-1]]
    m16.chat("new thread", thread_id="t3")
    assert "remember the word banana" not in [msg.content for msg in script.prompts[-1]]


# ---------------------------------------------------------------------------
# Human-in-the-loop (07)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("approved, prefix", [(True, "[EXECUTED]"), (False, "[CANCELLED]")])
def test_07_waits_for_human_decision(framework, approved, prefix):
    m = load(framework, "07")
    propose, resolve = registry.hitl_api(framework, m)
    token, proposed = propose("send the email")
    calls_after_propose = script.calls
    assert proposed.startswith("FAKE reply")
    assert resolve(proposed, token, approved).startswith(prefix)
    assert script.calls == calls_after_propose + (1 if approved else 0)  # rejection never executes


# ---------------------------------------------------------------------------
# Structured output, parallel, RAG (11, 13, 14)
# ---------------------------------------------------------------------------

def test_11_returns_validated_dict(framework):
    _, run = runner(framework, "11")
    analysis = run(raw_text="email text")["analysis"]
    assert analysis["contact"]["email"] == "sjohnson@acme.com"
    assert analysis["action_items"] == []  # schema defaults applied


def test_13_fans_out_one_analysis_per_item(framework):
    _, run = runner(framework, "13")
    result = run(items=["Python", "Rust", "Go"], category="safety")
    assert sorted(a.split("]")[0] + "]" for a in result["analyses"]) == ["[Go]", "[Python]", "[Rust]"]
    assert result["summary"]


@pytest.mark.parametrize("k", [1, 3])
def test_14_retrieves_k_docs_and_grounds_prompt(framework, k):
    _, run = runner(framework, "14")
    result = run(query="What does Send() do?", num_results=k)
    assert len(result["retrieved_docs"]) == k
    assert all(doc in _last_system_prompt() for doc in result["retrieved_docs"])
