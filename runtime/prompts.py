"""The system prompt the CLI gives the model.

It states today's date, since a model can't know it otherwise and needs it to judge what
"latest" or "this week" means and how old a search result is. It is built fresh on every start,
so a resumed session gets the current date.
"""

from __future__ import annotations

from datetime import date

BASE = (
    "You are a helpful assistant working in the user's terminal. Today's date is {today}. "
    "Use the tools to look things up instead of guessing."
)

WEB = (
    "For current events, news, recent releases, prices, or anything that may have changed since "
    "your training, search the web before answering: use web_search with topic \"news\" for news, "
    "and a time_range when recency matters. Use read_web_page when a result's snippet isn't "
    "enough. Prefer the most recent sources, mention dates, and list the URLs you relied on."
)

END = "When you have the answer, reply without calling a tool."


def system_prompt(today: date, web: bool) -> str:
    """The prompt for a session; `web` adds the guidance for the web tools."""
    parts = [BASE.format(today=today.strftime("%A, %B %d, %Y"))]
    if web:
        parts.append(WEB)
    parts.append(END)
    return " ".join(parts)
