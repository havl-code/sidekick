"""Tests for the Sidekick's own logic: the activity labels, the evaluator's trace and the turn bookkeeping.
None of these call a model or start the MCP servers."""

import os

from langchain_core.messages import AIMessage, ToolMessage

from sidekick import (
    SANDBOX,
    Sidekick,
    clip,
    describe_tool_call,
    message_text,
    user_entry,
)


def test_clip_collapses_whitespace_and_cuts_long_text():
    assert clip("  a \n b\tc  ", 20) == "a b c"
    assert clip("abcdefghij", 5) == "abcd…"
    assert clip("abcde", 5) == "abcde"


def test_message_text_handles_strings_and_content_blocks():
    assert message_text(AIMessage(content="hello")) == "hello"
    blocks = AIMessage(content=[{"type": "text", "text": "one "}, {"type": "image"}, {"type": "text", "text": "two"}])
    assert message_text(blocks) == "one two"


def test_describe_tool_call_uses_labels_and_details():
    assert describe_tool_call("web_search", {"query": "kiwi fruit"}) == "Searching the web for kiwi fruit"
    assert describe_tool_call("write_todos", {"todos": []}) == "Updating the plan"
    assert (
        describe_tool_call("fetch_page", {"url": "https://example.com", "find": "per month"})
        == 'Reading https://example.com for "per month"'
    )


def test_describe_tool_call_shows_sandbox_paths_relative():
    path = os.path.join(SANDBOX, "notes", "plan.md")
    assert describe_tool_call("write_file", {"path": path}) == f"Saving {os.path.join('notes', 'plan.md')}"


def test_describe_tool_call_falls_back_for_unknown_tools():
    assert describe_tool_call("browser_drag", {}) == "Browser drag"


def test_user_entry_keeps_criteria():
    assert user_entry("Plan my week", "Cover every day") == {
        "role": "user",
        "content": "Plan my week",
        "criteria": "Cover every day",
    }


def test_trace_lists_tool_calls_with_results_and_skips_todos():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "nz rates"}, "id": "1"},
                {"name": "write_todos", "args": {"todos": []}, "id": "2"},
                {"name": "fetch_page", "args": {"url": "https://example.com"}, "id": "3"},
            ],
        ),
        ToolMessage(content="OCR is 2.5%", tool_call_id="1"),
    ]
    trace = Sidekick()._trace(messages)
    assert "web_search" in trace and "OCR is 2.5%" in trace
    assert "fetch_page" in trace and "(no result)" in trace
    assert "write_todos" not in trace


def test_trace_with_no_tools():
    assert Sidekick()._trace([AIMessage(content="Just an answer")]) == "(no tools were called)"


def test_conversation_summary_keeps_recent_user_and_assistant_turns():
    sidekick = Sidekick()
    assert sidekick._conversation_summary() == "(this is the first request)"
    sidekick.context = [
        user_entry("First question"),
        {"role": "assistant", "content": "First answer"},
        {"role": "evaluator", "content": "Looks good", "status": "met"},
    ]
    assert sidekick._conversation_summary() == "User: First question\nAssistant: First answer"


def test_settle_todos_completes_everything_on_success():
    sidekick = Sidekick()
    sidekick.todos = [{"content": "a", "status": "completed"}, {"content": "b", "status": "in_progress"}]
    sidekick._settle_todos(True)
    assert [t["status"] for t in sidekick.todos] == ["completed", "completed"]


def test_settle_todos_returns_unfinished_steps_to_pending_on_failure():
    sidekick = Sidekick()
    sidekick.todos = [{"content": "a", "status": "completed"}, {"content": "b", "status": "in_progress"}]
    sidekick._settle_todos(False)
    assert [t["status"] for t in sidekick.todos] == ["completed", "pending"]


def test_after_stop_starts_a_fresh_thread_and_fails_running_steps():
    sidekick = Sidekick()
    old_thread = sidekick.thread_id
    sidekick.paused = True
    sidekick.activity = [{"id": "1", "label": "Searching", "state": "running"}]
    request = user_entry("Find flights")

    history = sidekick.after_stop([], request)

    assert sidekick.thread_id != old_thread
    assert sidekick._fresh_thread and not sidekick.paused
    assert sidekick.activity[0]["state"] == "failed"
    assert history == [request, {"role": "notice", "content": "You stopped this task."}]
