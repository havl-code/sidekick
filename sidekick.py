"""The Sidekick: a create_agent worker wrapped in a homemade evaluator loop.

The worker is a single create_agent. Around it we run our own loop that checks the worker's
answer against the user's success criteria, and either accepts it, sends it back for another
attempt, or returns to the user with a question. Middleware gives the worker a plan it shares
with the UI, guardrails for PII and runaway costs, and a pause for human approval before
sensitive actions.

The conversation handed back to the UI is a list of entries, each with a role:
  user        what the user asked
  assistant   the worker's final reply, with the steps it took
  evaluator   the verdict, with a status of met, needs_input or not_met
  approval    actions waiting for (or resolved by) the user's decision
  notice      a short system note, such as a stopped task
"""

import json
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sidekick_tools import get_all_tools

load_dotenv(override=True)

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.join(HERE, "sandbox")
MAX_ATTEMPTS = 2  # each retry resends the whole conversation, so keep this low
DEFAULT_CRITERIA = "The answer should be clear, correct and complete"
TRACE_BUDGET = 9000  # characters of tool evidence the evaluator sees, shared across all calls


class EvaluatorOutput(BaseModel):
    feedback: str = Field(
        description="Brief, specific feedback: what is missing or wrong, and what to do about it"
    )
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True only if the assistant needs information or a decision that only the user can give"
    )


WORKER_PROMPT = """You are Sidekick, a capable general personal assistant who gets real tasks done for the user.
The user mostly asks for help with, in order: everyday personal tasks and planning; study and research;
travel and shopping; and job hunting and career.

## Your tools
- web_search: your first stop for facts, news, prices and finding the right pages. Each result has a title,
  link and often a date, so you can cite it.
- fetch_page: the default way to read a page in full. It is fast and cheap. Pass find="some phrase" to pull
  out just the part you need (for example find="per month" on a pricing page).
- wikipedia: background and history on well-known topics.
- The browser (browser_* tools): only for pages that fetch_page cannot read, that need JavaScript, clicking,
  forms or logging in. Browser actions do not send the page back: use browser_find to locate text or an
  element, and browser_snapshot only when you need the whole page. Dismiss cookie banners yourself.
- A sandbox filesystem: the only place you can read and write files.
- send_push_notification: only when the user asks you to send them something.
- request_human_help: when you reach something only a human can do, like logging in, a captcha or
  two-factor authentication. Tell the user exactly what to do in your browser window. If they say they
  cannot, find another way.

## How to work
- Be efficient. Run independent searches and page reads together in one step rather than one at a time.
  Do not read the same page twice; for a different part of a page you have read, use fetch_page with find.
  Stop researching once you have enough to do the task well.
- For tasks with several steps, write a short plan with write_todos, and update it alongside your next
  tool calls rather than as a step on its own.
- For research, reports and comparisons: plan what you need up front, gather it in two or three batches
  of parallel calls, then write. Use official pages for facts like prices and features, and independent
  sources (benchmarks, reviews) for judgements. One official page per product and two or three
  independent sources is usually enough.
- For flights, use Google Flights in the browser: go straight to https://www.google.com/travel/flights?q=...
  with a query like "flights from Auckland to Wellington leaving 14 July returning 21 July".
- Give prices in NZD unless the user asks otherwise. When converting, look up the current rate and state
  the rate and its date.
- If the user declines an action you asked approval for, do not try it again; carry on without it.

## Always deliver
- Produce the full deliverable in this reply. Never stop to ask permission to continue, and never end by
  offering to do the work later or to do more. The user wants the result now.
- If some facts cannot be verified or sources disagree, still deliver the complete result: mark those
  items clearly (for example "unverified" or "sources differ") and give your best judgement.
- Ask the user a question only when you truly cannot go on without something only they know.

## Your final answer
- Lead with the answer or verdict, then the supporting detail. Use headings, tables and lists when they help.
- Cite sources as markdown links next to the facts they support, each with its date: the published date
  if the page gives one, otherwise the date you retrieved it, e.g. [Cursor pricing](url) (accessed 29 Sep 2026).
- Put the deliverable in your reply. Save a file only when the user asks for one, and then name its path.
- Write in New Zealand English spelling (colour, organise, centre, licence as a noun) and never use em dashes."""


EVALUATOR_PROMPT = """You check whether an AI assistant has met the user's success criteria for a task.

## Recent conversation, for context
{conversation}

## The user's request
{message}

## The success criteria
{success_criteria}

## What the assistant did (tool calls with a short snippet of each result)
{trace}

## The assistant's reply
{reply}

Decide whether the success criteria are met.
- Judge against the criteria as the user wrote them, and pass the reply when it meets them in substance.
  Do not add requirements of your own. Do not fail it for choices the criteria leave open (such as which
  official page or which exchange rate source to use), for wording, or for improvements nobody asked for.
  Each retry costs the user time and money, so fail only for problems that matter to them.
- The deliverable matters most. If the assistant refused, stalled, asked permission to continue, or offered
  to do the work later instead of doing it, the criteria are NOT met and user input is NOT needed: tell it
  to do the work now with what it can find.
- A complete deliverable that clearly flags the items it could not verify meets criteria about flagging
  or verification for those items. A missing deliverable cannot.
- If a criterion cannot be fully met because the information does not exist or could not be found (for
  example no benchmark covers all the products), and the reply says so clearly, count it as met.
- Use the tool results as evidence. The snippets are short, so a fact missing from a snippet is not proof
  it is wrong; fail the reply when it contradicts the evidence, cites sources it never looked at, or claims
  to have done something no tool call did.
- Set user input needed only when the assistant needs information or a decision that only the user can give.
Keep feedback brief and specific: name what to fix. Write in New Zealand English with no em dashes."""


# Plain-English labels for the live activity feed, keyed by tool name
ACTIVITY_LABELS = {
    "web_search": ("Searching the web for", "query"),
    "fetch_page": ("Reading", "url"),
    "browser_find": ("Finding on the page", "text"),
    "wikipedia": ("Reading Wikipedia", "query"),
    "send_push_notification": ("Sending a notification", None),
    "request_human_help": ("Asking for your help", None),
    "write_todos": ("Updating the plan", None),
    "browser_navigate": ("Opening", "url"),
    "browser_navigate_back": ("Going back a page", None),
    "browser_snapshot": ("Reading the page", None),
    "browser_click": ("Clicking", "element"),
    "browser_type": ("Typing into", "element"),
    "browser_fill_form": ("Filling in a form", None),
    "browser_select_option": ("Choosing an option", "element"),
    "browser_press_key": ("Pressing", "key"),
    "browser_wait_for": ("Waiting for the page", None),
    "browser_take_screenshot": ("Taking a screenshot", None),
    "browser_tabs": ("Switching tabs", None),
    "write_file": ("Saving", "path"),
    "edit_file": ("Editing", "path"),
    "read_text_file": ("Reading", "path"),
    "read_file": ("Reading", "path"),
    "read_multiple_files": ("Reading files", None),
    "list_directory": ("Looking in", "path"),
    "create_directory": ("Creating folder", "path"),
    "move_file": ("Moving", "source"),
    "search_files": ("Searching files for", "pattern"),
}


def describe_tool_call(name: str, args: dict) -> str:
    label, key = ACTIVITY_LABELS.get(name, (name.replace("browser_", "browser ").replace("_", " ").capitalize(), None))
    detail = args.get(key) if key else None
    if detail and isinstance(detail, str):
        if key == "path":
            detail = os.path.relpath(detail, SANDBOX) if os.path.isabs(detail) else detail
        looking_for = f' for "{clip(args["find"], 30)}"' if name == "fetch_page" and args.get("find") else ""
        return f"{label} {clip(detail, 80)}{looking_for}"
    return label


def user_entry(message: str, success_criteria: str = "") -> dict:
    """How a user's request is kept in the conversation, with the success criteria they gave (if any)."""
    return {"role": "user", "content": message, "criteria": success_criteria}


def clip(text, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def message_text(message) -> str:
    """The plain text of a message, whether its content is a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


class TolerateToolErrors(AgentMiddleware):
    """Hand tool failures back to the model as a message so it can recover, rather than
    crashing the run. Tools that touch the outside world, like a browser, fail now and then."""

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as error:
            return ToolMessage(
                content=f"That tool call failed: {error}. Try another approach.",
                tool_call_id=request.tool_call["id"],
            )


class Sidekick:
    def __init__(self):
        self.thread_id = str(uuid.uuid4())
        self.memory = InMemorySaver()
        self.tools = None
        self.sessions = None
        self.worker = None
        self.evaluator = None
        self.task = ""
        self.success_criteria = ""
        self.context = []
        self.attempts = 0
        self.paused = False
        self.pending_actions = 0
        self.todos = []
        self.activity = []
        self._turn_start = 0
        self._seen = 0
        self._fresh_thread = False

    async def setup(self):
        os.makedirs(SANDBOX, exist_ok=True)
        self.tools, self.sessions = await get_all_tools(SANDBOX)
        self.worker = create_agent(
            model="openai:gpt-5.4-mini",
            tools=self.tools,
            system_prompt=f"{WORKER_PROMPT}\n\nToday is {datetime.now():%A %d %B %Y}.",
            middleware=[
                TolerateToolErrors(),
                TodoListMiddleware(),
                # Once the conversation passes ~50k tokens, older tool results (mostly browser
                # pages) are swapped for a short placeholder in what is sent to the model. The
                # full results stay in the saved thread, so the evaluator still sees them.
                ContextEditingMiddleware(
                    edits=[ClearToolUsesEdit(trigger=50_000, keep=6, exclude_tools=("write_todos",))]
                ),
                PIIMiddleware("email"),
                PIIMiddleware("credit_card", apply_to_tool_results=True),
                ModelCallLimitMiddleware(run_limit=30),
                HumanInTheLoopMiddleware(
                    interrupt_on={"send_push_notification": True, "request_human_help": True}
                ),
            ],
            checkpointer=self.memory,
        )
        # A little reasoning costs a few hundred tokens per check, and saves whole retries caused by
        # a snap judgement that fails a good answer
        self.evaluator = ChatOpenAI(model="gpt-5.4-mini", reasoning_effort="low").with_structured_output(
            EvaluatorOutput
        )

    @property
    def config(self):
        return {"configurable": {"thread_id": self.thread_id}}

    # ---------- Evaluation ----------

    def _conversation_summary(self) -> str:
        turns = [e for e in self.context if e["role"] in ("user", "assistant")][-6:]
        if not turns:
            return "(this is the first request)"
        return "\n".join(f"{e['role'].capitalize()}: {clip(e['content'], 500)}" for e in turns)

    def _trace(self, messages: list) -> str:
        """Every tool call made this turn, with its arguments and a snippet of what came back.
        All calls are listed so the evaluator can see every source; the snippets share a fixed
        budget, so a long run gets shorter snippets rather than a bigger bill."""
        results = {m.tool_call_id: message_text(m) for m in messages if isinstance(m, ToolMessage)}
        calls = [
            call
            for message in messages
            for call in getattr(message, "tool_calls", None) or []
            if call["name"] != "write_todos"
        ]
        if not calls:
            return "(no tools were called)"
        snippet = max(150, TRACE_BUDGET // len(calls))
        return "\n".join(
            f"- {call['name']}({clip(json.dumps(call['args'], ensure_ascii=False), 150)})\n"
            f"  -> {clip(results.get(call['id'], '(no result)'), snippet)}"
            for call in calls
        )

    async def evaluate(self, reply: str, messages: list) -> EvaluatorOutput:
        prompt = EVALUATOR_PROMPT.format(
            conversation=self._conversation_summary(),
            message=self.task,
            success_criteria=self.success_criteria,
            trace=self._trace(messages),
            reply=reply,
        )
        return await self.evaluator.ainvoke(prompt)

    # ---------- Running a turn ----------

    async def run_turn(self, message: str, success_criteria: str, history: list) -> list:
        """One turn of conversation: the worker attempts the task and the evaluator checks it,
        retrying with feedback up to MAX_ATTEMPTS. If the worker pauses for approval, this
        returns straight away with paused set, and resume() continues the same turn."""
        self.task = message
        self.success_criteria = success_criteria or DEFAULT_CRITERIA
        self.context = history
        self.attempts = 0
        self.todos = []
        self.activity = []
        state = await self.worker.aget_state(self.config)
        self._turn_start = self._seen = len(state.values.get("messages", []))

        content = f"{message}\n\nThe success criteria for this task are: {self.success_criteria}"
        if self._fresh_thread and history:
            # After a stop we start a clean thread, so bring the worker up to speed
            content = f"Earlier in our conversation:\n{self._conversation_summary()}\n\nNew request: {content}"
        self._fresh_thread = False

        payload = {"messages": [{"role": "user", "content": content}]}
        return await self._advance(payload, history + [user_entry(message, success_criteria)])

    async def resume(self, history: list, approve: bool, note: str = "") -> list:
        """Approve or decline the actions the worker paused on, and continue the turn."""
        if approve:
            decision = {"type": "approve"}
        else:
            reason = f" Their reason: {note}" if note else ""
            decision = {
                "type": "reject",
                "message": f"The user declined this action, so it was not carried out.{reason} "
                "Do not try it again; carry on without it or explain what you could not do.",
            }
        history = [dict(e) for e in history]
        for entry in reversed(history):
            if entry["role"] == "approval" and entry.get("status") == "pending":
                entry["status"] = "approved" if approve else "declined"
                break
        payload = Command(resume={"decisions": [decision] * self.pending_actions})
        return await self._advance(payload, history)

    def _track_activity(self, messages: list):
        """Turn new tool calls into plain-English steps for the UI, and tick them off as results arrive."""
        for message in messages[self._seen:]:
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    self.activity.append(
                        {"id": call["id"], "label": describe_tool_call(call["name"], call["args"]), "state": "running"}
                    )
            elif isinstance(message, ToolMessage):
                for step in self.activity:
                    if step["id"] == message.tool_call_id:
                        step["state"] = "failed" if getattr(message, "status", "") == "error" else "done"
        self._seen = len(messages)

    async def _advance(self, payload, history: list) -> list:
        while True:
            result = None
            async for result in self.worker.astream(payload, config=self.config, stream_mode="values"):
                self.todos = result.get("todos", self.todos)
                self._track_activity(result.get("messages", []))

            if "__interrupt__" in result:
                actions = result["__interrupt__"][0].value["action_requests"]
                self.paused = True
                self.pending_actions = len(actions)
                return history + [
                    {
                        "role": "approval",
                        "status": "pending",
                        "actions": [{"name": a["name"], "args": a["args"]} for a in actions],
                    }
                ]

            self.paused = False
            messages = result["messages"][self._turn_start:]
            reply = message_text(result["messages"][-1])
            self.attempts += 1
            self.activity.append({"id": f"eval-{self.attempts}", "label": "Checking the answer", "state": "running"})
            verdict = await self.evaluate(reply, messages)
            self.activity[-1]["state"] = "done"

            if verdict.success_criteria_met or verdict.user_input_needed or self.attempts >= MAX_ATTEMPTS:
                status = "met" if verdict.success_criteria_met else "needs_input" if verdict.user_input_needed else "not_met"
                steps = [
                    {"label": s["label"], "state": s["state"]}
                    for s in self.activity
                    if not s["id"].startswith(("eval-", "retry-"))
                ]
                return history + [
                    {"role": "assistant", "content": reply, "steps": steps},
                    {"role": "evaluator", "content": verdict.feedback, "status": status, "attempts": self.attempts},
                ]
            self.activity.append(
                {"id": f"retry-{self.attempts}", "label": f"Trying again (attempt {self.attempts + 1})", "state": "done"}
            )
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "An internal reviewer checked your last response against the success "
                        f"criteria and found problems: {verdict.feedback}\n"
                        "Fix them now, reusing what you have already found rather than starting again. "
                        "The user never saw your last response or this review, so write your reply as a fresh, "
                        "complete answer: do not mention drafts, corrections or feedback, and do not ask "
                        "permission or offer to continue later.",
                    }
                ]
            }

    def after_stop(self, history: list, request: dict | None = None) -> list:
        """The user stopped a run part way. The agent's thread may now end in a tool call with no
        result, which the model would reject, so start a clean thread; the next turn gets a recap.
        Pass the user's entry when a new turn was stopped, so it still shows in the conversation."""
        self.thread_id = str(uuid.uuid4())
        self._fresh_thread = True
        self.paused = False
        self.pending_actions = 0
        for step in self.activity:
            if step["state"] == "running":
                step["state"] = "failed"
        added = [request] if request else []
        return history + added + [{"role": "notice", "content": "You stopped this task."}]

    def cleanup(self):
        """Shut down the MCP servers; the browser window closes."""
        if self.sessions:
            self.sessions.stop()
