"""
One Streamlit app for everything in this repo: the 16 agent patterns and the four capstones.
Run: streamlit run main.py   (or double-click start.bat on Windows)

Each page is still a standalone script (e.g. `streamlit run projects/data_analyst/ui.py` works on
its own); this file only adds navigation. .streamlit/config.toml binds the server to localhost.
"""
import streamlit as st

st.set_page_config(page_title="AI Agents", page_icon="🤖", layout="wide")


def home():
    st.title("🤖 AI Agent Patterns")
    st.write("Sixteen agent design patterns in LangGraph and LangChain, plus three applications that combine "
             "them. The design write-up is in `guide.md`.")
    cards = [
        ("app.py", "🧩 Agent patterns", "All 16 patterns, LangGraph vs LangChain, with guardrail toggles, "
         "built-in demos, evals and source."),
        ("projects/customer_support/ui.py", "🎧 Customer support", "Router, handoffs, help-center RAG, memory "
         "and supervisor approval for refunds."),
        ("projects/research_report/ui.py", "📑 Research reports", "Plan review, parallel web research, a "
         "reviewer loop, and citations checked by code."),
        ("projects/data_analyst/ui.py", "📊 Data analyst", "Pandas in a sandbox, numbers verified against code "
         "output, charts and \"show your work\"."),
        ("projects/doc_qa/ui.py", "📚 Document Q&A", "Ingest PDFs and Markdown, hybrid search with re-ranking, "
         "cited answers, and a retrieval benchmark."),
    ]
    for col, (path, title, text) in zip(st.columns(len(cards)), cards):
        with col.container(border=True):
            st.subheader(title)
            st.caption(text)
            st.page_link(path, label="Open", icon="➡️")
    st.caption("Requires OPENAI_API_KEY and TAVILY_API_KEY in .env. The server only listens on localhost.")


pages = {
    "": [st.Page(home, title="Home", icon="🏠", default=True)],
    "Patterns": [st.Page("app.py", title="Agent patterns", icon="🧩", url_path="patterns")],
    "Capstones": [
        st.Page("projects/customer_support/ui.py", title="Customer support", icon="🎧", url_path="support"),
        st.Page("projects/research_report/ui.py", title="Research reports", icon="📑", url_path="research"),
        st.Page("projects/data_analyst/ui.py", title="Data analyst", icon="📊", url_path="analyst"),
        st.Page("projects/doc_qa/ui.py", title="Document Q&A", icon="📚", url_path="docs"),
    ],
}
st.navigation(pages).run()
