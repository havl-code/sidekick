"""Tools for the Sidekick: a mix of MCP servers, ready-made LangChain tools and our own.

Every tool's description is sent with every model call, and every tool result stays in the
conversation, so tokens matter here: our tools return compact text, the MCP servers are
trimmed to the tools the Sidekick actually uses, and the browser only sends a page back
when the agent asks for one.
"""

import asyncio
import os
import re
import tempfile
import time
from contextlib import AsyncExitStack
from datetime import date

import requests
import wikipedia
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv(override=True)

USER_AGENT = "Mozilla/5.0 (compatible; sidekick/0.1; +https://github.com/havl-code)"
PAGE_CHAR_LIMIT = 6000
SEARCH_RESULTS = 8
PAGE_CACHE_SECONDS = 30 * 60
BLOCK_TAGS = [
    "p", "div", "li", "tr", "br", "section", "article", "table", "ul", "ol",
    "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "blockquote", "pre",
]

# Wikimedia rejects the wikipedia library's default user agent, so identify ourselves properly
wikipedia.set_user_agent(USER_AGENT)

wikipedia_lookup = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
)

# The MCP tools the Sidekick needs. The servers offer 39 between them; sending only these
# saves roughly 3,500 tokens on every model call and gives the model fewer ways to wander.
MCP_TOOLS = {
    "browser_navigate", "browser_navigate_back", "browser_snapshot", "browser_find",
    "browser_click", "browser_type", "browser_fill_form", "browser_select_option",
    "browser_press_key", "browser_wait_for", "browser_tabs", "browser_handle_dialog",
    "read_text_file", "write_file", "edit_file", "list_directory", "create_directory",
}


@tool
def web_search(query: str) -> str:
    """Search the web with Google. Returns the top results, each with its title, link,
    date (when Google knows it) and a snippet. Cite the links as sources, and read the
    most useful ones in full with fetch_page."""
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": os.getenv("SERPER_API_KEY", ""), "Content-Type": "application/json"},
        json={"q": query, "num": SEARCH_RESULTS, "gl": "nz"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    lines = []
    answer = data.get("answerBox") or {}
    if answer.get("answer") or answer.get("snippet"):
        lines.append(f"Answer box: {answer.get('answer') or answer.get('snippet')} ({answer.get('link', 'no link')})")
    for i, result in enumerate(data.get("organic", [])[:SEARCH_RESULTS], 1):
        dated = f" ({result['date']})" if result.get("date") else ""
        lines.append(f"{i}. {result.get('title', '')}{dated}\n   {result.get('link', '')}\n   {result.get('snippet', '')}")
    return "\n".join(lines) or "No results. Try different words."


def _published_date(soup: BeautifulSoup, headers) -> str:
    for attrs in (
        {"property": "article:published_time"},
        {"property": "article:modified_time"},
        {"name": "date"},
        {"itemprop": "datePublished"},
        {"itemprop": "dateModified"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return tag["content"][:10]
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        return time_tag["datetime"][:10]
    return headers.get("Last-Modified", "")


def _around(text: str, phrase: str, limit: int) -> str:
    """The paragraphs of text that mention phrase, each with its neighbours, up to limit characters."""
    paragraphs = text.split("\n")
    wanted = set()
    for i, paragraph in enumerate(paragraphs):
        if phrase.lower() in paragraph.lower():
            wanted.update({i - 1, i, i + 1})
    picked = [paragraphs[i] for i in sorted(wanted) if 0 <= i < len(paragraphs)]
    return "\n".join(picked)[:limit]


_page_cache: dict[str, tuple[float, str, str, str]] = {}


def _load_page(url: str) -> tuple[str, str, str] | str:
    """Download a page and return (title, published date, text), or an error message.
    Pages are cached for a while, so reading another part of a page with find is instant."""
    cached = _page_cache.get(url)
    if cached and time.time() - cached[0] < PAGE_CACHE_SECONDS:
        return cached[1:]
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    except requests.RequestException as error:
        return f"Could not reach {url}: {error}"
    if response.status_code >= 400:
        return f"{url} returned HTTP {response.status_code}. Try the browser, or another source."
    if "html" not in response.headers.get("Content-Type", "html"):
        return f"{url} is not a web page ({response.headers.get('Content-Type')}). Try the browser."

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    published = _published_date(soup, response.headers)
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside", "iframe"]):
        tag.decompose()
    body = soup.find("main") or soup.find("article") or soup.body or soup
    # One line per block (paragraph, list item, table row), with inline pieces like "$20" and "/ mo." kept together
    for block in body.find_all(BLOCK_TAGS):
        block.append("\n")
    text = re.sub(r"[^\S\n]+", " ", body.get_text(" "))
    text = re.sub(r" ?\n[\s]*", "\n", text).strip()

    if len(text) < 200:
        return f"{url} returned almost no text; it probably needs JavaScript. Use the browser instead."
    _page_cache[url] = (time.time(), title, published, text)
    return title, published, text


@tool
def fetch_page(url: str, find: str = "") -> str:
    """Read a web page as plain text without opening the browser. Much faster and cheaper
    than the browser, so use it for articles, documentation, pricing pages and reviews.
    Long pages are cut short; pass find (a word or phrase, like "per month") to get just
    the parts of the page that mention it. Use the browser instead for pages that need
    JavaScript, logging in, or clicking."""
    page = _load_page(url)
    if isinstance(page, str):
        return page
    title, published, text = page
    header = f"Title: {title}\nURL: {url}\nPublished: {published or 'unknown'}\nRetrieved: {date.today():%d %B %Y}\n\n"
    if find:
        found = _around(text, find, PAGE_CHAR_LIMIT)
        return header + (found or f"The page does not mention \"{find}\". Its opening:\n{text[:1500]}")
    if len(text) > PAGE_CHAR_LIMIT:
        return header + text[:PAGE_CHAR_LIMIT] + f"\n\n[Cut at {PAGE_CHAR_LIMIT} of {len(text)} characters. Use find to get a specific part.]"
    return header + text


@tool
def send_push_notification(text: str) -> str:
    """Send a short push notification to the user's phone."""
    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={"token": os.getenv("PUSHOVER_TOKEN"), "user": os.getenv("PUSHOVER_USER"), "message": text},
        timeout=15,
    )
    response.raise_for_status()
    return "Notification sent"


@tool
def request_human_help(instructions: str) -> str:
    """Ask the user to do something in the browser window that you cannot do yourself,
    such as logging in to a site, passing a captcha, or approving two-factor authentication.
    Explain exactly what you need them to do. The run pauses until they have done it."""
    return "The user says it is done. Continue with the task."


def mcp_connections(sandbox: str) -> dict:
    """The MCP servers the Sidekick uses: a headed browser and a sandbox filesystem.

    The browser runs with --snapshot-mode none, so clicks and page loads do not send the whole
    page back each time (the agent reads a page with browser_snapshot or browser_find when it
    needs to), and with images omitted, since screenshots are expensive in tokens. Its log
    files go to the temp folder rather than the project."""
    return {
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y", "@playwright/mcp@latest", "--isolated",
                "--snapshot-mode", "none",
                "--image-responses", "omit",
                "--console-level", "error",
                "--output-dir", os.path.join(tempfile.gettempdir(), "sidekick-playwright"),
            ],
        },
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", sandbox],
        },
    }


class McpSessions:
    """Holds persistent MCP sessions open so the browser keeps its state between tool calls.

    The stdio transport must be opened and closed from the same asyncio task, so one
    background task owns the sessions: it opens them, waits, and unwinds them when stop()
    is called. Stopping shuts down the servers, and you will see the browser close.
    """

    def __init__(self, connections: dict):
        self.connections = connections
        self.tools = []
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task = None

    async def _run(self):
        client = MultiServerMCPClient(self.connections)
        async with AsyncExitStack() as stack:
            for name in self.connections:
                session = await stack.enter_async_context(client.session(name))
                self.tools += await load_mcp_tools(session, server_name=name)
            self._ready.set()
            await self._stop.wait()

    async def start(self) -> list:
        self._task = asyncio.create_task(self._run())
        ready = asyncio.create_task(self._ready.wait())
        await asyncio.wait([ready, self._task], return_when=asyncio.FIRST_COMPLETED)
        ready.cancel()
        if self._task.done():
            self._task.result()  # the servers failed to start; raise the real error
        return self.tools

    def stop(self):
        self._stop.set()


async def get_all_tools(sandbox: str):
    """Return the full tool list (our tools plus the MCP tools we keep) and the session holder."""
    sessions = McpSessions(mcp_connections(sandbox))
    mcp_tools = [t for t in await sessions.start() if t.name in MCP_TOOLS]
    our_tools = [web_search, fetch_page, wikipedia_lookup, send_push_notification, request_human_help]
    return our_tools + mcp_tools, sessions
