"""
Streamlit UI for document Q&A: upload documents (or use the samples), pick a retrieval mode,
chat with cited answers, and inspect what each retriever found.
Run from the repo root:  streamlit run projects/doc_qa/ui.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `projects.` and guardrails

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from projects.doc_qa.graph import SAMPLE_DOCS, DocQASession  # noqa: E402
from projects.doc_qa.ingest import SUPPORTED, ingest  # noqa: E402
from projects.doc_qa.retrieval import MODES, Retriever  # noqa: E402
from projects.research_report.graph import format_cost  # noqa: E402

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")

EXAMPLES = [
    "How much paid time off do I get per year?",
    "What does error E204 mean on the X200?",
    "What address opens the router's admin console?",
    "Do I need a receipt for a $40 taxi ride?",
    "What's our stock option vesting schedule?",
]
MODE_HELP = {
    "bm25": "Keyword search only: exact words, codes and numbers.",
    "vector": "Embedding search only: meaning and paraphrases.",
    "hybrid": "Keyword + vector, merged with reciprocal rank fusion.",
    "hybrid+rerank": "Hybrid, then an LLM grades each passage and drops irrelevant ones (can refuse).",
}


@st.cache_resource(show_spinner="Indexing documents...")
def build_retriever(key: tuple, paths: tuple) -> Retriever:
    return Retriever(ingest(paths))


# ---------------------------------------------------------------------------
# Documents and settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 Document Q&A")
    uploads = st.file_uploader("Upload documents (optional)", type=[s.lstrip(".") for s in SUPPORTED],
                               accept_multiple_files=True)
    if uploads:
        folder = Path(tempfile.gettempdir()) / "docqa-uploads" / str(abs(hash(tuple(u.name for u in uploads))))
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for u in uploads:
            (folder / u.name).write_bytes(u.getvalue())
            paths.append(str(folder / u.name))
        key = tuple((u.name, u.size) for u in uploads)
    else:
        paths, key = [str(p) for p in sorted(SAMPLE_DOCS.iterdir())], ("samples",)
    retriever = build_retriever(key, tuple(paths))

    mode = st.selectbox("Retrieval mode", MODES, index=MODES.index("hybrid+rerank"), help="\n\n".join(
        f"**{m}**: {h}" for m, h in MODE_HELP.items()))
    st.caption(MODE_HELP[mode])
    k = st.slider("Passages per answer", 1, 8, 4)

    corpus = retriever.corpus
    st.caption(f"**{len(corpus.documents)}** documents · **{len(corpus.chunks)}** chunks")
    for name in corpus.documents:
        st.caption(f"📄 {name}")
    for q in corpus.quarantined:
        st.caption(f"🛡️ quarantined a passage in {q['doc']}: {q['reason']}")
    for s in corpus.skipped:
        st.caption(f"⚠️ skipped {s}")

    if st.button("🔄 New conversation", use_container_width=True) or \
            st.session_state.get("docqa_key") != (key, mode, k):
        st.session_state.docqa_session = DocQASession(retriever=retriever, mode=mode, k=k)
        st.session_state.docqa_key, st.session_state.docqa_turns = (key, mode, k), []
    st.subheader("Try")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True, key=f"ex_{q}"):
            st.session_state.docqa_pending = q

session: DocQASession = st.session_state.docqa_session

# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

question = st.chat_input("Ask about the documents...") or st.session_state.pop("docqa_pending", None)
if question:
    with st.spinner("Searching and answering..."):
        try:
            st.session_state.docqa_turns.append((question, session.ask(question)))
        except Exception as exc:  # network / API errors
            st.error(f"{type(exc).__name__}: {exc}")

for q, r in st.session_state.docqa_turns:
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        st.markdown(r["answer"])
        if r["citations"]:
            st.caption("Sources: " + " · ".join(f"[{c['n']}] {c['label']}" for c in r["citations"]))
        badge = {"answered": "✓ citations checked", "answered_with_concerns": "⚠️ some claims unverified",
                 "not_found": "∅ not in the documents", "blocked": "⛔ blocked"}.get(r["status"], r["status"])
        st.caption(f"{badge} · mode `{session.mode}`"
                   + (f" · searched as: _{r['standalone']}_" if r["standalone"] and r["standalone"] != q else ""))
        for note in r["guard_notes"]:
            st.caption(f"🛡️ {note}")
        if r["hits"]:
            with st.expander(f"Retrieved passages ({len(r['hits'])})"):
                for n, h in enumerate(r["hits"], 1):
                    ranks = " · ".join(f"{name} #{rank}" for name, rank in h["ranks"].items())
                    grade = f" · relevance {h['relevance']}/3" if h["relevance"] is not None else ""
                    st.markdown(f"**[{n}] {h['label']}**  \n<small>{ranks}{grade}</small>", unsafe_allow_html=True)
                    st.text(h["text"][:600])

if st.session_state.docqa_turns:
    tokens = sum(u["input_tokens"] + u["output_tokens"] for u in session.usage.values())
    st.caption(f"Conversation tokens: {tokens:,} ({format_cost(session.usage)})")
else:
    st.info("Ask a question about the documents in the sidebar, or try an example. Upload your own Markdown, "
            "text or PDF files to search those instead.")
