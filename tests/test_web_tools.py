"""web_search and read_web_page, against a fake Tavily client. No network, except the smoke test,
which runs only when AGENT_LIVE_WEB_TESTS is set."""

import os

import pytest
from tavily.errors import TavilyKeylessLimitError, UsageLimitExceededError

from runtime.messages import ToolCall
from runtime.tools import ToolRegistry
from runtime.web_tools import MAX_PAGE_CHARS, MAX_SNIPPET_CHARS, web_tools


class FakeTavily:
    """Records each call's options and returns the prepared response (or raises it)."""

    def __init__(self, search=None, extract=None):
        self.search_response = search if search is not None else {"results": []}
        self.extract_response = extract if extract is not None else {"results": [], "failed_results": []}
        self.calls = []

    async def search(self, **options):
        self.calls.append(("search", options))
        if isinstance(self.search_response, BaseException):
            raise self.search_response
        return self.search_response

    async def extract(self, **options):
        self.calls.append(("extract", options))
        if isinstance(self.extract_response, BaseException):
            raise self.extract_response
        return self.extract_response


async def test_search_results_are_numbered_with_url_date_and_text():
    client = FakeTavily(search={"results": [
        {"title": "Chip export rules eased", "url": "https://news.example/a", "content": "The rules...",
         "published_date": "Thu, 05 Nov 2026 14:00:00 GMT", "score": 0.9},
        {"title": "Second story", "url": "https://news.example/b", "content": "More text"},
    ]})
    out = await _run(client, "web_search", query="chip export rules", topic="news", time_range="week")

    assert out.splitlines()[:4] == [
        "[1] Chip export rules eased",
        "https://news.example/a",
        "published: Thu, 05 Nov 2026 14:00:00 GMT",
        "The rules...",
    ]
    assert "[2] Second story\nhttps://news.example/b\nMore text" in out


async def test_search_sends_topic_recency_and_a_cheap_depth():
    client = FakeTavily()
    await _run(client, "web_search", query="q", topic="news", time_range="day", max_results=3)
    assert client.calls == [("search", {
        "query": "q", "topic": "news", "time_range": "day", "max_results": 3, "search_depth": "basic",
    })]


async def test_search_defaults():
    client = FakeTavily()
    await _run(client, "web_search", query="q")
    options = client.calls[0][1]
    assert options["topic"] == "general" and options["max_results"] == 5
    assert "time_range" not in options


async def test_no_results():
    assert await _run(FakeTavily(), "web_search", query="nothing") == "No results."


async def test_long_snippets_are_clipped():
    client = FakeTavily(search={"results": [{"title": "t", "url": "u", "content": "x" * 5_000}]})
    out = await _run(client, "web_search", query="q")
    assert "x" * MAX_SNIPPET_CHARS in out and "x" * (MAX_SNIPPET_CHARS + 1) not in out


@pytest.mark.parametrize(
    "arguments, error",
    [({"query": "q", "topic": "sports"}, "invalid arguments"),
     ({"query": "q", "max_results": 50}, "invalid arguments"),
     ({}, "invalid arguments")],
    ids=["unknown-topic", "too-many-results", "no-query"],
)
async def test_bad_search_arguments_are_rejected_before_calling_tavily(arguments, error):
    client = FakeTavily()
    result = await ToolRegistry(web_tools(client)).run(ToolCall("c", "web_search", arguments))
    assert result.is_error and error in result.output
    assert client.calls == []


async def test_read_web_page_returns_markdown():
    client = FakeTavily(extract={"results": [{"url": "https://a.example", "raw_content": "# Title\n\nBody"}],
                                 "failed_results": []})
    assert await _run(client, "read_web_page", url="https://a.example") == "# Title\n\nBody"
    assert client.calls == [("extract", {"urls": ["https://a.example"], "format": "markdown"})]


async def test_long_pages_are_clipped():
    client = FakeTavily(extract={"results": [{"raw_content": "y" * 50_000}], "failed_results": []})
    out = await _run(client, "read_web_page", url="https://a.example")
    assert out.startswith("y" * MAX_PAGE_CHARS) and "more characters" in out


async def test_a_page_tavily_could_not_read_is_an_error():
    client = FakeTavily(extract={"results": [],
                                 "failed_results": [{"url": "https://a.example", "error": "timed out"}]})
    result = await _registry(client).run(ToolCall("c", "read_web_page", {"url": "https://a.example"}))
    assert result.is_error and "could not read https://a.example: timed out" in result.output


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://a.example/x", "not a url"])
async def test_only_http_urls_are_read(url):
    client = FakeTavily()
    result = await _registry(client).run(ToolCall("c", "read_web_page", {"url": url}))
    assert result.is_error and "not an http(s) URL" in result.output
    assert client.calls == []


@pytest.mark.parametrize(
    "error, message",
    [(TavilyKeylessLimitError("limit"), "set TAVILY_API_KEY"),
     (UsageLimitExceededError("limit"), "monthly credits are used up")],
    ids=["keyless", "plan"],
)
async def test_quota_errors_say_what_to_do(error, message):
    result = await _registry(FakeTavily(search=error)).run(ToolCall("c", "web_search", {"query": "q"}))
    assert result.is_error and message in result.output


def test_web_tools_only_read():
    registry = _registry(FakeTavily())
    for name in ("web_search", "read_web_page"):
        assert registry.is_idempotent(name)
        assert not registry.requires_approval(name)


@pytest.mark.skipif(not os.environ.get("AGENT_LIVE_WEB_TESTS"), reason="set AGENT_LIVE_WEB_TESTS=1")
async def test_smoke_live_news_search():
    """Real Tavily request (keyless unless TAVILY_API_KEY is set)."""
    out = await ToolRegistry(web_tools()).run(
        ToolCall("c", "web_search", {"query": "technology news", "topic": "news", "time_range": "week"})
    )
    assert not out.is_error and out.output.startswith("[1] ")

# ── helpers ──────────────────────────────────────────────────────────────────────────────


def _registry(client):
    return ToolRegistry(web_tools(client))


async def _run(client, name, **arguments):
    result = await _registry(client).run(ToolCall("c", name, arguments))
    assert not result.is_error, result.output
    return result.output
