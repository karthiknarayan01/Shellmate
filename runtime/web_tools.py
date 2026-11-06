"""Web tools: search the web and read a page. Free, with no API key or account.

    web_search(query, topic?, time_range?, max_results?)
        topic       "general" (default) or "news" for news articles, with dates and sources
        time_range  "day" | "week" | "month" | "year": only results from that period
    read_web_page(url)
        the page's main content as markdown, for when a search snippet isn't enough

Both use the `ddgs` library (formerly duckduckgo-search). It is a metasearch client: it sends
the query to public search engines (DuckDuckGo, Bing, Brave, ...) the way a browser would and
parses their result pages, so it needs no key and has no quota. It is unofficial: the engines
can rate-limit it, and a site change can break it until the library is updated. Rate limits
come back to the model as a result saying to wait.

ddgs is synchronous, so the tool functions are plain functions and run in worker threads.
Both tools only read, so they need no approval and are idempotent (safe to re-run after a crash).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from runtime.tools import Tool

DEFAULT_RESULTS = 5
MAX_RESULTS = 10
MAX_SNIPPET_CHARS = 1_200
MAX_PAGE_CHARS = 20_000
TIMEOUT_SECONDS = 10
TIME_LIMITS = {"day": "d", "week": "w", "month": "m", "year": "y"}


def web_tools(client: Any = None) -> list[Tool]:
    """The tools, sharing one search client."""
    client = client or DDGS(timeout=TIMEOUT_SECONDS)
    return [
        Tool(
            name="web_search",
            description=(
                "Search the web. Use it for current events, news, recent releases, prices and "
                "anything that may have changed recently. Returns titles, URLs, publication dates "
                "and sources for news, and a snippet of each result."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news"],
                        "description": "'news' for news articles; default 'general'",
                    },
                    "time_range": {
                        "type": "string",
                        "enum": list(TIME_LIMITS),
                        "description": "Only results from this recent period",
                    },
                    "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            fn=lambda query, topic="general", time_range=None, max_results=DEFAULT_RESULTS: (
                web_search(client, query, topic, time_range, max_results)
            ),
            idempotent=True,
        ),
        Tool(
            name="read_web_page",
            description="Read a web page's main content as markdown. Use it after web_search "
            "when a result's snippet isn't enough.",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "http(s) URL"}},
                "required": ["url"],
                "additionalProperties": False,
            },
            fn=lambda url: read_web_page(client, url),
            idempotent=True,
        ),
    ]


def web_search(client: Any, query: str, topic: str, time_range: str | None, max_results: int) -> str:
    """Run one search and format the results as numbered entries the model can cite."""
    options: dict[str, Any] = {"max_results": max_results}
    if time_range:
        options["timelimit"] = TIME_LIMITS[time_range]
    search = client.news if topic == "news" else client.text
    try:
        results = _call(search, query, **options)
    except _NoResults:
        return "No results."
    return format_results(results)


def format_results(results: list[dict[str, Any]]) -> str:
    """[n] title / url / date and source (news) / snippet, clipped.

    Text results carry `href` and `body`; news results carry `url`, `body`, `date`, `source`.
    """
    if not results:
        return "No results."
    entries = []
    for n, result in enumerate(results, 1):
        lines = [f"[{n}] {result.get('title', '').strip()}", result.get("url") or result.get("href", "")]
        published = " · ".join(part for part in (result.get("date"), result.get("source")) if part)
        if published:
            lines.append(f"published: {published}")
        lines.append(_clip(result.get("body", "").strip(), MAX_SNIPPET_CHARS))
        entries.append("\n".join(lines))
    return "\n\n".join(entries)


def read_web_page(client: Any, url: str) -> str:
    """The page's main content as markdown, clipped."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"not an http(s) URL: {url!r}")
    try:
        page = _call(client.extract, url, fmt="text_markdown")
    except _NoResults:
        raise RuntimeError(f"could not read {url}: no content") from None
    content = page.get("content", "")
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return _clip(content.strip() or "(the page has no text)", MAX_PAGE_CHARS)


class _NoResults(Exception):
    pass


def _call(method, *args, **options):
    """Call ddgs, turning its errors into messages that say what happened.

    ddgs raises (rather than returning []) when no engine found anything.
    """
    try:
        return method(*args, **options)
    except RatelimitException:
        raise RuntimeError(
            "the search engines are rate-limiting requests; wait a minute before searching again"
        ) from None
    except TimeoutException:
        raise RuntimeError(f"the search timed out after {TIMEOUT_SECONDS}s; try again") from None
    except DDGSException as e:
        if "no results" in str(e).lower():
            raise _NoResults() from None
        raise RuntimeError(f"search failed: {e}") from None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f" ... [{len(text) - limit} more characters]"
