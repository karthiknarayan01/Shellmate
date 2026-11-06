"""Web tools backed by Tavily, a search API built for agents: search the web, read a page.

    web_search(query, topic?, time_range?, max_results?)
        topic       "general" (default) or "news" for news articles
        time_range  "day" | "week" | "month" | "year": only results from that period
    read_web_page(url)
        the page's main content as markdown, for when a search snippet isn't enough

Tavily returns each result's relevant text already extracted, so the model gets content it can
use rather than raw HTML. Without TAVILY_API_KEY the client runs in Tavily's keyless mode, which
supports these two calls with a small free quota; with a key (free tier: 1,000 credits a month)
the limits are higher. Queries and URLs are sent to Tavily.

Both tools only read, so they need no approval and are idempotent (safe to re-run after a crash).
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from tavily import AsyncTavilyClient
from tavily.errors import TavilyKeylessLimitError, UsageLimitExceededError

from runtime.tools import Tool

API_KEY_ENV = "TAVILY_API_KEY"
DEFAULT_RESULTS = 5
MAX_RESULTS = 10
MAX_SNIPPET_CHARS = 1_200
MAX_PAGE_CHARS = 20_000


def web_tools(client: Any = None) -> list[Tool]:
    """The tools, sharing one Tavily client (keyed by TAVILY_API_KEY when it is set)."""
    client = client or AsyncTavilyClient(api_key=os.environ.get(API_KEY_ENV) or None)

    async def search(query, topic="general", time_range=None, max_results=DEFAULT_RESULTS):
        return await web_search(client, query, topic, time_range, max_results)

    async def read(url):
        return await read_web_page(client, url)

    return [
        Tool(
            name="web_search",
            description=(
                "Search the web. Use it for current events, news, recent releases, prices and "
                "anything that may have changed recently. Returns titles, URLs, publication dates "
                "for news, and the relevant text of each result."
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
                        "enum": ["day", "week", "month", "year"],
                        "description": "Only results from this recent period",
                    },
                    "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            fn=search,
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
            fn=read,
            idempotent=True,
        ),
    ]


async def web_search(
    client: Any, query: str, topic: str, time_range: str | None, max_results: int
) -> str:
    """Run one search and format the results as numbered entries the model can cite."""
    options: dict[str, Any] = {
        "query": query,
        "topic": topic,
        "max_results": max_results,
        "search_depth": "basic",  # 1 credit; "advanced" costs 2
    }
    if time_range:
        options["time_range"] = time_range
    response = await _call(client.search, **options)
    return format_results(response.get("results", []))


def format_results(results: list[dict[str, Any]]) -> str:
    """[n] title / url / published date (news) / the result's text, clipped."""
    if not results:
        return "No results."
    entries = []
    for n, result in enumerate(results, 1):
        lines = [f"[{n}] {result.get('title', '').strip()}", result.get("url", "")]
        if result.get("published_date"):
            lines.append(f"published: {result['published_date']}")
        lines.append(_clip(result.get("content", "").strip(), MAX_SNIPPET_CHARS))
        entries.append("\n".join(lines))
    return "\n\n".join(entries)


async def read_web_page(client: Any, url: str) -> str:
    """The page's content as markdown, clipped. A page Tavily couldn't read is an error."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"not an http(s) URL: {url!r}")
    response = await _call(client.extract, urls=[url], format="markdown")
    for failed in response.get("failed_results", []):
        reason = failed.get("error", "unknown error")
        raise RuntimeError(f"could not read {failed.get('url', url)}: {reason}")
    pages = response.get("results", [])
    if not pages:
        raise RuntimeError(f"could not read {url}: no content returned")
    return _clip(pages[0].get("raw_content", "").strip() or "(the page has no text)", MAX_PAGE_CHARS)


async def _call(method, **options) -> dict[str, Any]:
    """Call Tavily, turning quota errors into a message that says what to do about them."""
    try:
        return await method(**options)
    except TavilyKeylessLimitError:
        raise RuntimeError(
            f"the free keyless search quota is used up; set {API_KEY_ENV} "
            "(free key at https://app.tavily.com) for more"
        ) from None
    except UsageLimitExceededError:
        raise RuntimeError("the Tavily plan's monthly credits are used up") from None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f" ... [{len(text) - limit} more characters]"
