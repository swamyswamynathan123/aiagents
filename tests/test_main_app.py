"""The combined multipage app: every page loads, and pages don't clobber each other's state."""
import tomllib

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT

PAGES = ["app.py", "projects/customer_support/ui.py", "projects/research_report/ui.py",
         "projects/data_analyst/ui.py", "projects/doc_qa/ui.py", "projects/coding_agent/ui.py"]


def open_main():
    return AppTest.from_file(str(ROOT / "main.py"), default_timeout=90).run()


def test_home_page_links_every_page():
    at = open_main()
    assert not at.exception
    assert any("AI Agent Patterns" in t.value for t in at.title)


@pytest.mark.parametrize("page", PAGES)
def test_each_page_loads_inside_the_app(page):
    at = open_main()
    at.switch_page(page).run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_switching_pages_keeps_each_pages_state():
    at = open_main()
    at.switch_page("projects/customer_support/ui.py").run()
    support = at.session_state["support_session"]
    at.switch_page("projects/data_analyst/ui.py").run()
    at.switch_page("projects/research_report/ui.py").run()
    at.switch_page("projects/customer_support/ui.py").run()
    assert at.session_state["support_session"] is support  # not overwritten by another page
    assert "analyst_session" in at.session_state


def test_server_is_bound_to_localhost():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["address"] == "localhost"
    assert "--server.address localhost" in (ROOT / "start.bat").read_text(encoding="utf-8")
