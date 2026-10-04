"""Tests for our own tools. Network calls are replaced with fakes, so no API keys are needed."""

import requests
from bs4 import BeautifulSoup

import sidekick_tools
from sidekick_tools import MCP_TOOLS, PAGE_CHAR_LIMIT, _around, _published_date, fetch_page, mcp_connections, web_search


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_around_returns_matching_paragraphs_with_neighbours():
    text = "intro\nbefore\nIt costs $20 per month\nafter\nunrelated\nend"
    assert _around(text, "PER MONTH", 1000) == "before\nIt costs $20 per month\nafter"
    assert _around(text, "missing", 1000) == ""


def test_published_date_prefers_meta_then_time_then_header():
    meta = BeautifulSoup('<meta property="article:published_time" content="2026-09-01T10:00:00Z">', "html.parser")
    assert _published_date(meta, {}) == "2026-09-01"
    time_tag = BeautifulSoup('<time datetime="2026-08-15">15 Aug</time>', "html.parser")
    assert _published_date(time_tag, {}) == "2026-08-15"
    assert _published_date(BeautifulSoup("<p>hi</p>", "html.parser"), {"Last-Modified": "Mon"}) == "Mon"


def test_fetch_page_cuts_long_pages(monkeypatch):
    monkeypatch.setattr(sidekick_tools, "_load_page", lambda url: ("Title", "2026-09-01", "x" * (PAGE_CHAR_LIMIT + 10)))
    result = fetch_page.invoke({"url": "https://example.com"})
    assert "Title: Title" in result and "Published: 2026-09-01" in result
    assert f"[Cut at {PAGE_CHAR_LIMIT} of {PAGE_CHAR_LIMIT + 10} characters" in result


def test_fetch_page_find_returns_only_matching_parts(monkeypatch):
    monkeypatch.setattr(sidekick_tools, "_load_page", lambda url: ("T", "", "header\nPro: $20 per month\nfooter\nother"))
    result = fetch_page.invoke({"url": "https://example.com", "find": "per month"})
    assert "Published: unknown" in result
    assert "Pro: $20 per month" in result and "other" not in result


def test_fetch_page_passes_errors_through(monkeypatch):
    monkeypatch.setattr(sidekick_tools, "_load_page", lambda url: f"{url} returned HTTP 404.")
    assert fetch_page.invoke({"url": "https://example.com"}) == "https://example.com returned HTTP 404."


def test_load_page_reports_unreachable_sites(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(sidekick_tools.requests, "get", fail)
    assert sidekick_tools._load_page("https://unreachable.test").startswith("Could not reach")


def test_web_search_formats_results(monkeypatch):
    data = {
        "answerBox": {"answer": "42", "link": "https://answer.test"},
        "organic": [{"title": "A result", "link": "https://a.test", "snippet": "Snippet", "date": "1 Sep 2026"}],
    }
    monkeypatch.setattr(sidekick_tools.requests, "post", lambda *args, **kwargs: FakeResponse(data))
    result = web_search.invoke({"query": "life"})
    assert result.splitlines()[0] == "Answer box: 42 (https://answer.test)"
    assert "1. A result (1 Sep 2026)" in result and "https://a.test" in result


def test_web_search_with_no_results(monkeypatch):
    monkeypatch.setattr(sidekick_tools.requests, "post", lambda *args, **kwargs: FakeResponse({}))
    assert web_search.invoke({"query": "zzz"}) == "No results. Try different words."


def test_mcp_connections_point_the_filesystem_at_the_sandbox():
    connections = mcp_connections("/tmp/sandbox")
    assert set(connections) == {"playwright", "filesystem"}
    assert connections["filesystem"]["args"][-1] == "/tmp/sandbox"
    assert "--snapshot-mode" in connections["playwright"]["args"]


def test_kept_mcp_tools_have_activity_labels():
    from sidekick import ACTIVITY_LABELS

    unlabelled = {name for name in MCP_TOOLS if name not in ACTIVITY_LABELS}
    assert unlabelled <= {"browser_handle_dialog"}
