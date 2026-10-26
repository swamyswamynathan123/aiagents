"""
11 - Structured Output Agent
Pattern: Single-node extraction pipeline using llm.with_structured_output()
         and Pydantic v2 models. No loops — the model produces validated,
         type-safe output in one call.
Use cases: Data extraction from unstructured text, form field population,
           entity recognition, document parsing, ETL from natural language.
LangSmith tracing: automatic when LANGCHAIN_TRACING_V2=true is set in .env
"""
import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")  # set AGENT_MODEL in .env to try another model

_llm = ChatOpenAI(model=MODEL, temperature=0)

# ---------------------------------------------------------------------------
# Pydantic v2 schemas
# ---------------------------------------------------------------------------

class ContactInfo(BaseModel):
    name: str = Field(description="Full name of the person")
    email: Optional[str] = Field(default=None, description="Email address if present")
    phone: Optional[str] = Field(default=None, description="Phone number if present")
    company: Optional[str] = Field(default=None, description="Company or organization name")
    role: Optional[str] = Field(default=None, description="Job title or role")


class MeetingRequest(BaseModel):
    date: Optional[str] = Field(default=None, description="Proposed meeting date (e.g. 'Tuesday', '2026-09-15')")
    time: Optional[str] = Field(default=None, description="Proposed meeting time (e.g. '2pm', '14:00')")
    duration_minutes: Optional[int] = Field(default=None, description="Meeting duration in minutes")
    participants: List[str] = Field(default_factory=list, description="Names of all participants to invite")
    topic: Optional[str] = Field(default=None, description="Main topic or agenda of the meeting")
    location: Optional[str] = Field(default=None, description="Meeting location or video call link")


class EmailAnalysis(BaseModel):
    contact: ContactInfo
    meeting: Optional[MeetingRequest] = Field(default=None, description="Meeting request details if present")
    sentiment: str = Field(description="Overall tone: positive, negative, or neutral")
    priority: str = Field(description="Urgency level: high, medium, or low")
    action_items: List[str] = Field(default_factory=list, description="List of explicit action items or requests")
    summary: str = Field(description="One-sentence summary of the email's main purpose")


_extractor = _llm.with_structured_output(EmailAnalysis)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ExtractionState(TypedDict):
    raw_text: str
    analysis: Optional[dict]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def extract_node(state: ExtractionState) -> dict:
    result: EmailAnalysis = _extractor.invoke([
        SystemMessage(content=(
            "You are an expert at extracting structured information from business emails. "
            "Extract all relevant details accurately. For missing fields, use null."
        )),
        HumanMessage(content=f"Extract information from this email:\n\n{state['raw_text']}"),
    ])
    return {"analysis": result.model_dump()}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_builder = StateGraph(ExtractionState)
_builder.add_node("extract", extract_node)
_builder.set_entry_point("extract")
_builder.add_edge("extract", END)
app = _builder.compile()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

SAMPLE_EMAILS = [
    """
    Hi,

    I'm Sarah Johnson, Head of Partnerships at Acme Corp (sjohnson@acme.com, 555-867-5309).

    I'd love to schedule a 45-minute call with you and your team (Tom Chen and Lisa Park)
    to discuss a potential Q4 integration partnership. Are you available Tuesday at 2pm or
    Thursday at 10am? We can use Zoom.

    This is fairly time-sensitive as we're finalizing our vendor list by end of month.

    Best,
    Sarah
    """,
    """
    Team,

    Just a quick heads-up: the deployment last night introduced a regression in the
    payment flow. Customers are reporting 403 errors when checking out.

    Priority: CRITICAL. Please investigate immediately and roll back if needed.
    I need an incident report by EOD.

    - Alex (alex.rivera@startup.io, Engineering Lead)
    """,
]


def main():
    print("=== 11 Structured Output Agent ===\n")

    for i, email in enumerate(SAMPLE_EMAILS, 1):
        print(f"--- Email {i} ---")
        print(f"Raw: {email.strip()[:100]}...\n")

        result = app.invoke({"raw_text": email.strip(), "analysis": None})
        analysis = result["analysis"]

        print(f"Extracted:")
        print(json.dumps(analysis, indent=2, default=str))
        print()


if __name__ == "__main__":
    main()
