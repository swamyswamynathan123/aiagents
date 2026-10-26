"""Eval runner: local-mode evaluate() wiring and the reference checks."""
from run_evals import make_checks, run_agent_evals, run_guardrail_evals

from fakes import script


def test_local_agent_evals_run_and_score(framework):
    script.structured["RouteDecision"] = {"route": "math", "reason": "r"}
    script.reply = lambda system, human: "x = 5"
    rows = run_agent_evals(framework, "03", upload=False, use_judge=True)
    assert len(rows) == 3
    by_query = {r["inputs"]["query"]: r for r in rows}
    math = by_query["Solve 3x + 7 = 22 for x."]
    assert math["passed"], math["checks"]
    # the fake always routes to math, so the writing/general cases must fail their route check
    assert not by_query["What is the capital of Australia?"]["passed"]


def test_local_guardrail_evals_pass(monkeypatch):
    import guardrails

    # Script the two network-backed verdicts the red-team suite relies on
    monkeypatch.setattr(guardrails, "moderate", lambda text: "violence" if "hurt my coworker" in text else None)
    script.structured["InjectionVerdict"] = {"is_injection": True, "reason": "asks for configuration"}
    rows = run_guardrail_evals(upload=False)
    assert len(rows) == 8
    assert all(r["passed"] for r in rows), [(r["inputs"], r["checks"]) for r in rows if not r["passed"]]


def _scores(reference, outputs, use_judge=False):
    return {k: s for k, s, *_ in [(r["key"], r["score"]) for r in make_checks(use_judge)({}, outputs, reference)["results"]]}


def test_checks_text_tools_fields_and_bounds():
    outputs = {"answer": "The total is 7,006,652.", "tools": ["calculate"], "fields": {"route": "Math", "steps": 4}}
    reference = {
        "must_contain": ["7006652"], "must_not_contain": ["error"], "must_use_tools": ["calculate"],
        "any_tool_of": ["search_web", "calculate"], "fields": {"route": "math"},
        "min_fields": {"steps": 2}, "max_fields": {"steps": 3},
    }
    assert _scores(reference, outputs) == {
        "contains:7006652": 1, "excludes:error": 1, "tool:calculate": 1, "tool:any": 1,
        "field:route": 1, "min:steps": 1, "max:steps": 0,
    }


def test_checks_report_target_errors():
    assert _scores({"must_contain": ["x"]}, {"error": "ValueError: boom"}) == {"error": 0}


def test_judge_skipped_when_disabled():
    assert "judge" not in _scores({"criteria": "anything"}, {"answer": "a", "tools": [], "fields": {}})
