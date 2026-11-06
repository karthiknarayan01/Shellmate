"""web_search and read_web_page, against a fake search client (no network). The smoke test makes
real requests and runs only when AGENT_LIVE_WEB_TESTS is set."""

import os

import pytest
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from runtime.messages import ToolCall
from runtime.tools import ToolRegistry
from runtime.web_tools import MAX_PAGE_CHARS, MAX_SNIPPET_CHARS, web_tools

NEWS = [
    {"date": "2026-11-05T14:00:00+00:00", "title": "Chip export rules eased", "body": "The rules...",
     "url": "https://news.example/a", "source": "Example News"},
    {"title": "Second story", "body": "More text", "url": "https://news.example/b"},
]
TEXT = [{"title": "Python 3.14 released", "href": "https://python.example/314", "body": "What's new"}]


class FakeSearch:
    """Stands in for ddgs.DDGS: records calls and returns (or raises) prepared responses."""

    def __init__(self, text=TEXT, news=NEWS, page=None):
        self.responses = {"text": text, "news": news, "extract": page or {"content": "# Title\n\nBody"}}
        self.calls = []

    def text(self, query, **options):
        return self._respond("text", query, options)

    def news(self, query, **options):
        return self._respond("news", query, options)

    def extract(self, url, **options):
        return self._respond("extract", url, options)

    def _respond(self, kind, arg, options):
        self.calls.append((kind, arg, options))
        response = self.responses[kind]
        if isinstance(response, BaseException):
            raise response
        return response


async def test_news_results_are_numbered_with_url_date_source_and_snippet():
    out = await _run(FakeSearch(), "web_search", query="chip export rules", topic="news")
    assert out.splitlines()[:4] == [
        "[1] Chip export rules eased",
        "https://news.example/a",
        "published: 2026-11-05T14:00:00+00:00 · Example News",
        "The rules...",
    ]
    assert "[2] Second story\nhttps://news.example/b\nMore text" in out


async def test_general_results_use_the_text_search():
    client = FakeSearch()
    out = await _run(client, "web_search", query="python release")
    assert out == "[1] Python 3.14 released\nhttps://python.example/314\nWhat's new"
    assert client.calls == [("text", "python release", {"max_results": 5})]


async def test_topic_recency_and_count_are_passed_on():
    client = FakeSearch()
    await _run(client, "web_search", query="q", topic="news", time_range="week", max_results=3)
    assert client.calls == [("news", "q", {"max_results": 3, "timelimit": "w"})]


async def test_no_results():
    client = FakeSearch(text=DDGSException("No results found."))
    assert await _run(client, "web_search", query="nothing") == "No results."


async def test_long_snippets_are_clipped():
    client = FakeSearch(text=[{"title": "t", "href": "u", "body": "x" * 5_000}])
    out = await _run(client, "web_search", query="q")
    assert "x" * MAX_SNIPPET_CHARS in out and "x" * (MAX_SNIPPET_CHARS + 1) not in out


@pytest.mark.parametrize(
    "arguments",
    [{"query": "q", "topic": "sports"}, {"query": "q", "max_results": 50}, {}],
    ids=["unknown-topic", "too-many-results", "no-query"],
)
async def test_bad_search_arguments_are_rejected_before_searching(arguments):
    client = FakeSearch()
    result = await _registry(client).run(ToolCall("c", "web_search", arguments))
    assert result.is_error and "invalid arguments" in result.output
    assert client.calls == []


@pytest.mark.parametrize(
    "error, message",
    [(RatelimitException("202 Ratelimit"), "wait a minute"),
     (TimeoutException("timed out"), "timed out"),
     (DDGSException("engine broke"), "search failed: engine broke")],
    ids=["rate-limit", "timeout", "other"],
)
async def test_search_errors_say_what_happened(error, message):
    result = await _registry(FakeSearch(text=error)).run(ToolCall("c", "web_search", {"query": "q"}))
    assert result.is_error and message in result.output


async def test_read_web_page_returns_markdown():
    client = FakeSearch()
    assert await _run(client, "read_web_page", url="https://a.example") == "# Title\n\nBody"
    assert client.calls == [("extract", "https://a.example", {"fmt": "text_markdown"})]


async def test_long_pages_are_clipped():
    client = FakeSearch(page={"content": "y" * 50_000})
    out = await _run(client, "read_web_page", url="https://a.example")
    assert out.startswith("y" * MAX_PAGE_CHARS) and "more characters" in out


async def test_a_page_that_cant_be_fetched_is_an_error():
    client = FakeSearch(page=DDGSException("Failed to fetch https://a.example: HTTP 404"))
    result = await _registry(client).run(ToolCall("c", "read_web_page", {"url": "https://a.example"}))
    assert result.is_error and "HTTP 404" in result.output


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://a.example/x", "not a url"])
async def test_only_http_urls_are_read(url):
    client = FakeSearch()
    result = await _registry(client).run(ToolCall("c", "read_web_page", {"url": url}))
    assert result.is_error and "not an http(s) URL" in result.output
    assert client.calls == []


def test_web_tools_only_read():
    registry = _registry(FakeSearch())
    for name in ("web_search", "read_web_page"):
        assert registry.is_idempotent(name)
        assert not registry.requires_approval(name)


@pytest.mark.skipif(not os.environ.get("AGENT_LIVE_WEB_TESTS"), reason="set AGENT_LIVE_WEB_TESTS=1")
async def test_smoke_live_news_search():
    result = await ToolRegistry(web_tools()).run(
        ToolCall("c", "web_search", {"query": "technology", "topic": "news", "time_range": "week"})
    )
    assert not result.is_error and result.output.startswith("[1] ")


# ── helpers ──────────────────────────────────────────────────────────────────────────────


def _registry(client):
    return ToolRegistry(web_tools(client))


async def _run(client, name, **arguments):
    result = await _registry(client).run(ToolCall("c", name, arguments))
    assert not result.is_error, result.output
    return result.output
