"""
Streamlit UI for the support assistant: customer chat on the left, supervisor console on the right.
Run from the repo root:  streamlit run projects/customer_support/ui.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `projects.` and guardrails

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from projects.customer_support import data, policies  # noqa: E402
from projects.customer_support.graph import SupportSession, build_app  # noqa: E402

st.set_page_config(page_title="Support Assistant", page_icon="🎧", layout="wide")


@st.cache_resource
def shared_graph():
    """One graph (checkpointer + long-term store) per server, so case notes persist across conversations."""
    return build_app()


def new_session(customer_id: str) -> None:
    st.session_state.support_session = SupportSession(customer_id, graph=shared_graph())
    st.session_state.support_chat = []
    st.session_state.support_last_turn = None


def record(turn: dict) -> None:
    st.session_state.support_last_turn = turn
    if turn["reply"]:
        st.session_state.support_chat.append(("assistant", turn["reply"]))


# ---------------------------------------------------------------------------
# Sidebar: who is the customer
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎧 Acme Gadgets Support")
    customer_id = st.selectbox("Signed-in customer", list(data.CUSTOMERS),
                               format_func=lambda c: f"{data.CUSTOMERS[c]['name']} ({c})")
    if "support_session" not in st.session_state or st.session_state.support_session.customer_id != customer_id:
        new_session(customer_id)
    if st.button("🔄 New conversation"):
        new_session(customer_id)
    if st.button("♻️ Reset demo data"):
        data.reset()
        st.cache_resource.clear()
        new_session(customer_id)
        st.rerun()

    st.subheader("Orders")
    for o in data.list_orders(customer_id):
        flag = " · refunded" if o["refunded"] else ""
        st.caption(f"**{o['order_id']}** {o['item']} · ${o['amount']:.2f} · {o['status']}{flag}")
    st.caption(f"Policy: refunds within {policies.REFUND_WINDOW_DAYS} days of delivery; "
               f"over ${policies.AUTO_APPROVE_LIMIT:.0f} needs a supervisor.")
    st.caption("Try: *Where is my USB-C hub?* · *Refund my earbuds A-1001* · *Refund the monitor A-1002* · "
               "*What's the status of order B-2001?* · *I want to speak to a manager*")

session: SupportSession = st.session_state.support_session
chat_col, console_col = st.columns([3, 2])

# ---------------------------------------------------------------------------
# Supervisor console (rendered first so approvals update the chat in the same run)
# ---------------------------------------------------------------------------

with console_col:
    st.subheader("🧑‍💼 Supervisor console")
    pending = session.pending_approval
    if pending:
        st.warning(f"**Refund approval needed**\n\n{data.CUSTOMERS[pending['customer_id']]['name']} · "
                   f"{pending['order_id']} · {pending['item']} · **${pending['amount']:.2f}**\n\n"
                   f"Customer's reason: _{pending['reason']}_")
        note = st.text_input("Note to customer (optional)", key="supervisor_note")
        c1, c2 = st.columns(2)
        decision = True if c1.button("✅ Approve", use_container_width=True) else \
            False if c2.button("❌ Reject", use_container_width=True) else None
        if decision is not None:
            with st.spinner("Resuming the graph..."):
                record(session.decide(decision, note))
            st.rerun()
    else:
        st.caption("No approvals waiting.")

    turn = st.session_state.support_last_turn
    if turn:
        st.markdown(f"**Last turn** · status `{turn['status']}` · intent `{turn['intent']}` · agent `{turn['agent']}`")
        for note in turn["guard_notes"]:
            st.caption(f"🛡️ {note}")
        for name, result in turn["tool_results"]:
            with st.expander(f"🔧 {name}"):
                st.text(result)

    with st.expander("Long-term case notes (this customer)"):
        notes = shared_graph().store.search(("case_notes", customer_id), limit=20)
        for item in sorted(notes, key=lambda i: i.created_at):
            st.caption(f"- {item.value['note']}")
        if not notes:
            st.caption("None yet.")
    with st.expander(f"Refund ledger ({len(data.REFUNDS)}) · Tickets ({len(data.TICKETS)})"):
        st.json({"refunds": data.REFUNDS, "tickets": data.TICKETS})

# ---------------------------------------------------------------------------
# Customer chat
# ---------------------------------------------------------------------------

with chat_col:
    st.subheader(f"💬 Chat — {data.CUSTOMERS[customer_id]['name']}")
    for role, text in st.session_state.support_chat:
        st.chat_message(role).write(text)
    if prompt := st.chat_input("Message support...", disabled=bool(session.pending_approval)):
        st.session_state.support_chat.append(("user", prompt))
        st.chat_message("user").write(prompt)
        with st.spinner("Assistant is working..."):
            try:
                record(session.send(prompt))
            except Exception as exc:  # network / API errors
                st.error(f"{type(exc).__name__}: {exc}")
        st.rerun()
    if session.pending_approval:
        st.info("⏸ Waiting for a supervisor to review the refund (see console).")
