"""
Offline stand-ins for ChatOpenAI, OpenAIEmbeddings and TavilySearch.

FakeChat is a real BaseChatModel, so prompts, bind_tools(), with_structured_output(),
ToolNode and LangGraph all run their actual code paths — only the model's replies
are scripted. Tests steer behavior through the module-level `script` object:

    script.structured["RouteDecision"] = {"route": "writing", "reason": "poem"}
    script.reply = lambda system, human: "HANDOFF:billing"
    script.tool_calls = 1          # how many tool rounds a tool-bound model makes
"""
import uuid
from typing import Any, Callable, List, Optional

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

DEFAULT_STRUCTURED = {
    "RouteDecision": {"route": "math", "reason": "numbers"},
    "Plan": {"steps": ["gather facts", "summarize"]},
    "ReplanDecision": {"is_complete": True, "final_answer": "FAKE final answer"},
    "CritiqueResult": {"approved": True, "score": 9, "feedback": "good"},
    "EmailAnalysis": {"contact": {"name": "Sarah Johnson", "email": "sjohnson@acme.com"}, "sentiment": "positive",
                      "priority": "high", "meeting": {"duration_minutes": 45}, "summary": "partnership call"},
    "EvalResult": {"score": 9, "passed": True, "strengths": "s", "weaknesses": "w", "feedback": "f"},
    "InjectionVerdict": {"is_injection": False, "reason": "benign"},
    "Verdict": {"score": 5, "reasoning": "meets the criteria"},  # evals LLM judge
}

TOOL_ARGS = {
    "get_weather": {"city": "Paris"}, "calculate": {"expression": "42 * 7"},
    "search_web": {"query": "langgraph"}, "get_stock_price": {"ticker": "AAPL"},
    "get_news_headlines": {"topic": "ai"}, "research_topic": {"topic": "quantum"},
    "save_note": {"note": "qubits scale"}, "generate_report": {"title": "T", "sections": "S"},
    "mark_goal_complete": {"summary": "done"}, "execute_python": {"code": "print(sum(range(5)))"},
}


class Script:
    def reset(self):
        self.structured = {k: dict(v) for k, v in DEFAULT_STRUCTURED.items()}
        self.reply: Optional[Callable[[str, str], str]] = None  # (system, human) -> text
        self.tool_calls = 2          # tool rounds before a tool-bound model answers
        self.tool_order: Optional[List[str]] = None  # which tools to call, in order
        self.calls = 0               # total model invocations
        self.prompts: List[List] = []  # every message list the model received


script = Script()
script.reset()


def _supervisor_next(human: str) -> str:
    if "Research complete" not in human:
        return "researcher"
    return "writer" if "Writing complete" not in human else "FINISH"


class FakeChat(BaseChatModel):
    model: str = "fake"
    temperature: float = 0
    tool_names: List[str] = []
    tool_choice: Optional[Any] = None

    @property
    def _llm_type(self):
        return "fake-chat"

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"tool_names": names, "tool_choice": tool_choice})

    @staticmethod
    def _tool_call(name, args):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        script.calls += 1
        script.prompts.append(list(messages))
        system = "\n".join(m.content for m in messages if isinstance(m, SystemMessage))
        human = "\n".join(m.content for m in messages if isinstance(m, HumanMessage))

        if self.tool_choice and len(self.tool_names) == 1:  # with_structured_output()
            name = self.tool_names[0]
            args = {"next": _supervisor_next(human), "reason": "r"} if name == "RoutingDecision" else script.structured[name]
            msg = self._tool_call(name, args)
        elif self.tool_names:  # tool-calling agent
            used = sum(isinstance(m, ToolMessage) for m in messages)
            order = script.tool_order or self.tool_names
            if used < script.tool_calls:
                name = order[used % len(order)]
                msg = self._tool_call(name, TOOL_ARGS[name])
            else:
                text = script.reply(system, human) if script.reply else f"FAKE answer after {used} tool calls"
                msg = AIMessage(content=text)
        elif script.reply:
            msg = AIMessage(content=script.reply(system, human))
        else:
            msg = AIMessage(content=f"FAKE reply #{script.calls}")
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FakeTavily:
    queries: List[dict] = []
    respond: Optional[Callable[[dict], Any]] = None  # tests can override the whole response

    def __init__(self, **kwargs):
        pass

    def invoke(self, payload):
        FakeTavily.queries.append(payload)
        if FakeTavily.respond:
            return FakeTavily.respond(payload)
        slug = "-".join(payload["query"].lower().split())[:40]
        return {"answer": f"fake answer for {payload['query']}",
                "results": [{"title": f"Result for {payload['query']}", "url": f"https://example.com/{slug}",
                             "content": f"content about {payload['query']}"}]}


def install():
    """Patch the provider classes before any agent module imports them."""
    import langchain_openai
    import langchain_tavily

    langchain_openai.ChatOpenAI = FakeChat
    langchain_openai.OpenAIEmbeddings = lambda **kwargs: DeterministicFakeEmbedding(size=16)
    langchain_tavily.TavilySearch = FakeTavily
