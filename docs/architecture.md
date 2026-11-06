# How agent-runtime works

This document explains the runtime from the outside in: what runs where, how one turn becomes a sequence of events in a database, and how that log makes crash recovery, approvals and compaction work. Code paths are given as `file.py: function` so you can follow along.

1. [What runs where](#1-what-runs-where)
2. [The components](#2-the-components)
3. [The event log](#3-the-event-log)
4. [Rebuilding state from events](#4-rebuilding-state-from-events)
5. [One turn, step by step](#5-one-turn-step-by-step)
6. [Approvals](#6-approvals)
7. [Crash recovery](#7-crash-recovery)
8. [Compaction](#8-compaction)
9. [Model clients](#9-model-clients)
10. [Tools](#10-tools)
11. [The command-line chat](#11-the-command-line-chat)
12. [Tests](#12-tests)
13. [Limits](#13-limits)

---

## 1. What runs where

Everything happens in **one OS process with one thread**. There is no server, no background worker and no async event loop: the agent loop is ordinary blocking Python. While the model is generating a reply, the thread is blocked on an HTTPS request; while a tool runs, the thread is inside the tool's function.

```
Fig. 1  One running agent-runtime

 ┌──────────────────────── process: python (agent-runtime) ─────────────────────────┐
 │  one thread                                                                      │
 │                                                                                  │
 │   CLI prompt loop ──► Session.run_turn() ──► model client ──┐                    │
 │        ▲                    │                               │ HTTPS (blocking)   │
 │        │                    ├──► ToolRegistry.execute() ──┐ │                    │
 │        │                    │                             │ │                    │
 │   in memory:                ▼                             │ │                    │
 │     SessionState     EventLog.append()                    │ │                    │
 │     (messages, tool         │ INSERT + COMMIT + fsync     │ │                    │
 │      call status)           │                             │ │                    │
 └─────────────────────────────┼─────────────────────────────┼─┼────────────────────┘
                               ▼                             ▼ ▼
                       agent.db (SQLite, WAL)       workspace     api.anthropic.com
                       sessions, events,            directory     or api.deepseek.com
                       schema_migrations            (files)
```

There are two kinds of state:

- **The event log** in `agent.db`. This is the only state that is stored. Every change is written here first.
- **`SessionState`** in memory. It is derived from the log by replaying it, and kept in step with the log as new events are written. If the process dies, it is simply rebuilt from the log.

The rest of this document follows from that split.

## 2. The components

| file | job |
|---|---|
| `runtime/migrations/*.sql`, `runtime/migrate.py` | schema, applied in numbered order |
| `runtime/events.py` | `EventLog`: append and load events |
| `runtime/messages.py` | `Message`, `ToolCall`: the conversation, independent of any provider |
| `runtime/state.py` | `project()` / `apply()`: fold events into `SessionState` |
| `runtime/session.py` | `Session`: the agent loop, approvals, resume, cancel |
| `runtime/compaction.py` | when and how to summarize old messages |
| `runtime/model.py` | the `ModelClient` interface and its reply type |
| `runtime/anthropic_model.py`, `runtime/deepseek_model.py` | the two provider adapters |
| `runtime/providers.py` | `make_model()`: pick a provider by name |
| `runtime/tools.py` | `Tool`, `ToolRegistry`: validate and run tool calls |
| `runtime/workspace_tools.py` | `list_files`, `read_file`, `write_file` |
| `runtime/approval.py` | `Decision`, the `Approver` interface, `AutoApprove`, `DenyAll` |
| `runtime/cli.py` | the terminal chat and its approval prompt |

Dependencies point one way: `cli → session → (state, compaction, tools, approval, model) → (events, messages)`. Only the two adapter files know a provider's wire format.

## 3. The event log

### 3.1 Schema

```
Fig. 2  Tables (runtime/migrations/0001_create_sessions_and_events.sql)

sessions                         events
┌──────────────┬────────────┐    ┌─────────┬──────────────┬──────────────────┬──────────────┬─────┐
│ session_id PK│ created_at │    │ seq PK  │ session_id FK│ type             │ payload JSON │ ts  │
├──────────────┼────────────┤    ├─────────┼──────────────┼──────────────────┼──────────────┼─────┤
│ 3f9a…        │ 2026-11-06 │    │ 101     │ 3f9a…        │ user_message     │ {"text":…}   │ …   │
└──────────────┴────────────┘    │ 102     │ 3f9a…        │ assistant_message│ {…}          │ …   │
                                 │ 103     │ 3f9a…        │ tool_started     │ {"call_id":…}│ …   │
                                 └─────────┴──────────────┴──────────────────┴──────────────┴─────┘
                                 index events_by_session (session_id, seq)
```

`seq` is SQLite's rowid: assigned on insert, always increasing, never reused because rows are never deleted. Ordering a session's events by `seq` gives the order they were written in.

### 3.2 Appending

`events.py: EventLog.append` is a single `INSERT` on a connection in autocommit mode. When it returns, the event is committed. The connection uses:

| setting | effect |
|---|---|
| `journal_mode=WAL` | writes go to a write-ahead log; readers don't block the writer |
| `synchronous=FULL` | the WAL is fsynced on every commit, so a committed event survives power loss, not only a process crash |
| `foreign_keys=ON` | an event for a session that doesn't exist is rejected by the insert itself |

Nothing is ever updated or deleted. A tool call's life is several events (`tool_started`, then `tool_finished`), not a status column that changes, so a crash between two events is visible afterwards as "started, never finished".

### 3.3 Migrations

`migrate.py: migrate` applies every `NNNN_name.sql` file whose number isn't in `schema_migrations` yet, in order. Each file runs in one transaction together with its `schema_migrations` row, so a failing migration leaves nothing behind. `EventLog` refuses to open a database that is behind and names the command to fix it. The CLI migrates on every start.

## 4. Rebuilding state from events

`state.py: project(events)` replays a session's events into a `SessionState`. `state.py: apply(state, event)` folds in one event; a running session calls it after every append, so its state always equals a full replay without reading the log back.

### 4.1 Event types

| type | payload | effect on state |
|---|---|---|
| `user_message` | `text` | user message; opens a turn |
| `assistant_message` | `text, reasoning, tool_calls, usage` | assistant message; each call registered as `requested`; a reply with no calls closes the turn |
| `approval_requested` | `call_id` | `requested → awaiting_approval` |
| `approval_decided` | `call_id, decision, feedback, source` | `→ approved`, or `→ finished` with a denial result |
| `tool_started` | `call_id` | `requested` or `approved → started` |
| `tool_restarted` | `call_id` | `started → started` (re-run after a crash) |
| `tool_finished` | `call_id, output, is_error, interrupted?` | `started → finished`; tool result message |
| `tool_skipped` | `call_id, reason` | not-yet-started `→ finished` with an error result |
| `turn_stopped` | `reason` | closes the turn without an answer (denied, cancelled, step limit, truncated reply) |
| `compaction` | `replaced_count, summary` | first `replaced_count` messages replaced by one summary message |

### 4.2 Tool call statuses

```
Fig. 3  How a tool call moves through its statuses (one event per arrow)

  requested ──tool_started──────────────────────────────────────────────► started
  requested ──approval_requested──► awaiting_approval ──approve──► approved ──tool_started──► started
                                    awaiting_approval ──deny─────► finished      (never started)
  requested | awaiting_approval | approved ──tool_skipped────────► finished      (never started)
  started   ──tool_restarted──► started                                          (re-run after a crash)
  started   ──tool_finished───► finished
```

### 4.3 Replay refuses corrupt logs

Each handler checks that its event is legal in the current state before changing anything: a call can't finish before it starts, a user message can't arrive while a turn is open, a compaction can't separate a tool result from its call. A log that breaks a rule raises `InvalidEventSequence` instead of being "repaired", because a guess at what happened is worse than a clear error.

## 5. One turn, step by step

`session.py: Session.run_turn` → `_continue_turn` → `_step` (repeated) → `_resolve_open_calls` → `_resolve` for each call.

```
Fig. 4  A turn: "read config.py and fix the typo"

 user ─► run_turn
          │ record user_message                                         turn_open = True
          ▼
   ┌─► _step
   │      │ _compact_if_needed()           (§8)
   │      │ model.complete(system, messages, tools) ──HTTPS──► provider
   │      │ reply truncated?  → record turn_stopped, raise ReplyTruncated
   │      │ record assistant_message {tool_calls: [read_file(config.py)]}
   │      │ for each call:
   │      │     check() → read_file is safe → record tool_started
   │      │                                   execute()
   │      │                                   record tool_finished {output}
   │      ▼
   │   any calls?  yes ─┐
   └────────────────────┘                     second step: write_file(config.py, …)
                                              needs approval → record approval_requested
                                              approver.ask() → you answer "y"
                                              record approval_decided, tool_started,
                                              tool_finished
          third step: reply has no tool calls → record assistant_message
                                              turn_open = False → return TurnResult
```

Every arrow labelled "record" goes through `Session._record`: append the event to the log, apply it to `SessionState`, then call the observer (the CLI prints progress from it). Writing to the log first is what makes the rest safe: if the process dies between the append and the update in memory, the restarted process rebuilds memory from the log anyway.

The events this turn leaves in the log:

```
user_message        {"text": "read config.py and fix the typo"}
assistant_message   {"tool_calls": [{"id": "t1", "name": "read_file", "arguments": {"path": "config.py"}}], ...}
tool_started        {"call_id": "t1"}
tool_finished       {"call_id": "t1", "output": "    1  DEBGU = True", "is_error": false}
assistant_message   {"tool_calls": [{"id": "t2", "name": "write_file", "arguments": {...}}], ...}
approval_requested  {"call_id": "t2"}
approval_decided    {"call_id": "t2", "decision": "approve", "feedback": "", "source": "approver"}
tool_started        {"call_id": "t2"}
tool_finished       {"call_id": "t2", "output": "wrote 13 characters to config.py", "is_error": false}
assistant_message   {"text": "Fixed: DEBGU → DEBUG.", "tool_calls": []}
```

A turn also ends when:

| condition | recorded | caller sees |
|---|---|---|
| the user denies a call without feedback | `tool_skipped` for the rest of the reply, `turn_stopped(denied)` | `TurnResult.denied_call` |
| `max_steps` (default 20) steps without a final answer | `turn_stopped(max_steps)` | `MaxStepsReached` |
| a reply is cut off at the output token cap | `turn_stopped(truncated)` | `ReplyTruncated` |
| `Session.cancel()` (Ctrl+C in the CLI) | see §7.3 | |

A model API failure (after the client's retries) raises `ModelError` and records nothing, so the turn stays open and `resume()` retries the step.

## 6. Approvals

A tool declares `requires_approval=True` if running it changes something (`write_file` does). For such a call, `_resolve` records `approval_requested`, gets a `Decision`, records `approval_decided`, and only then records `tool_started`. Replay rejects a `tool_started` for a call that is awaiting approval, so a log in which a denied or unanswered call ran cannot exist.

| decision | effect |
|---|---|
| `approve` | run this call |
| `approve_for_session` | run it, and add the tool to `state.approved_tools`: later calls of that tool are approved without asking (`source: session`) |
| `deny` with feedback | don't run it; the model gets "The user denied this tool call. Their feedback: …" and the turn continues |
| `deny` without feedback | don't run it; skip the rest of the reply; the turn stops |

The decision comes from an `Approver` (`ask(call) -> Decision`). The CLI's `TerminalApprover` prompts you; `AutoApprove` (`--yes`) approves everything; a session created without an approver uses `DenyAll`, so it can never run a risky tool by accident.

Session-wide approvals live in the log, not in the approver, so they survive a restart.

## 7. Crash recovery

### 7.1 Is there a turn to resume?

`state.turn_open` is set by `user_message` and cleared by a final reply or `turn_stopped`. Deliberate endings always record `turn_stopped`, so after replay `turn_open` is True only if the process died mid-turn. `Session.needs_resume` reports it, and `run_turn` refuses with `TurnInProgress` until it's handled.

### 7.2 Resuming

`Session.resume` gives every open call a result, starting from the status it reached, then continues the turn with the next model step. It uses the same `_resolve` as a normal step.

| status after replay | what happened before the crash | resume |
|---|---|---|
| `requested` | nothing | handle as new: check, approval, run |
| `awaiting_approval` | asked, never answered | ask again |
| `approved` | approved, never started | run |
| `started`, tool `idempotent` | may or may not have run | record `tool_restarted`, run again |
| `started`, tool not idempotent | may or may not have run | **don't run**; record `tool_finished` with `interrupted: true` and the result "Interrupted: … may or may not have completed. Check its effect before running it again." |
| (no open calls) | crashed during a model call, or between steps | make the next model step |

The rule for `started` is the core of the design. Between `tool_started` and the crash, the tool may have done its work (sent the message, charged the card) or not, and the log can't tell. Repeating an idempotent tool is harmless: reading a file, or writing the same full content again, leaves the same result. Repeating anything else could do the work twice, so the runtime refuses and lets the model check first.

Example, with a hypothetical non-idempotent `send_email` tool: the process is killed while it runs.

```
log before the crash                      after Session.open(...).resume()
────────────────────                      ────────────────────────────────
user_message       "email Sam the notes"
assistant_message  send_email(...)
approval_requested
approval_decided   approve
tool_started       c1          ◄── killed here
                                          tool_finished  c1 interrupted=true
                                          assistant_message  "The email may already
                                            have gone out. Want me to check your sent
                                            folder before sending again?"
```

### 7.3 Cancelling

`Session.cancel` ends an open turn without running anything more. Calls that never started get `tool_skipped("cancelled by the user")`; a started call gets the interrupted result (it may have had its effect); then `turn_stopped(cancelled)`. The CLI calls it when you press Ctrl+C during a turn. A crash and a Ctrl+C differ only in intent: after a crash you want the turn finished, after Ctrl+C you want it dropped.

## 8. Compaction

Before every model call, `Session._compact_if_needed` decides whether the conversation must be shortened.

### 8.1 When

```
estimate = measured_tokens + Σ chars(m)/4 for messages after the measurement
```

- `measured_tokens` is the provider's own count from the latest reply (input + output tokens). It covers every message up to that reply exactly.
- Messages added since (tool results, a new user message) are estimated at four characters per token.
- After a compaction there is no measurement until the next reply, so everything is estimated, including the system prompt and the tool definitions.

Compaction triggers when the estimate reaches `trigger_ratio × context_window`.

| setting | default | from |
|---|---|---|
| `context_window` | 128,000 | `AGENT_CONTEXT_WINDOW` or `CompactionPolicy(context_window=…)` |
| `trigger_ratio` | 0.8 | `CompactionPolicy` |
| `keep_ratio` | 0.25 | `CompactionPolicy` |

With the defaults, compaction starts at about 102,400 tokens and keeps about 32,000 tokens verbatim.

### 8.2 Where to cut

`compaction.py: choose_cut` walks back from the newest message, adding up tokens, and keeps the longest recent tail that fits in `keep_ratio × context_window`. Two rules constrain the cut:

- the kept part never starts with a tool result, which would separate it from the assistant message that made the call (model APIs reject that);
- at least one message is kept and at least one summarized.

```
Fig. 5  Cutting a history of 9 messages

 index   0      1         2      3         4      5         6      7         8
 role    user   asst      tool   asst      tool   asst      tool   asst      tool
                call A    res A  call B    res B  call C    res C  call D    res D
                                                            ◄──── fits in keep budget ────►
                                                            but 6 is a tool result,
                                                            so the cut moves to 7 ─┐
 summarized: 0–6 ───────────────────────────────────────────────────────────────┘ kept: 7–8
```

### 8.3 What replaces the old part

The messages before the cut are rendered as a plain-text transcript (`USER: …`, `ASSISTANT CALLED read_file({...})`, `TOOL RESULT: …`, with long tool results clipped to 2,000 characters) and sent to the same model as a separate request, with a summarizing system prompt and no tools. The summary replaces them as one user message:

```
Summary of the earlier conversation:
<the model's summary>

The user's current request, verbatim:
<the user message that started this turn>
```

The second part matters for long turns. A 100-step turn may compact many times, and each summary is a summary of the previous one; copying the request word for word means the goal can't drift or disappear, whatever the summaries leave out.

### 8.4 How it's recorded

The result is one `compaction` event: `{replaced_count, summary}`. Replay applies it the same way, so `Session.open` and `resume()` continue from the compacted history. The original messages stay in the log before the event; only the state sent to the model changes. Nothing is recorded until the summary exists, so a crash during the summary request, or a summary cut off at the output cap (`CompactionFailed`), leaves the log unchanged and the next step compacts again.

## 9. Model clients

A `ModelClient` has one method: `complete(system, messages, tools) -> ModelReply`, where a reply holds `text`, `reasoning`, `tool_calls`, `usage` and `truncated`. The adapters translate in both directions.

| | Anthropic (`anthropic_model.py`) | DeepSeek (`deepseek_model.py`) |
|---|---|---|
| API | Messages API (`anthropic` SDK) | Chat Completions format (`openai` SDK, `https://api.deepseek.com`) |
| tool results | all results for one reply in one user message, as `tool_result` blocks with `is_error` | one `role: tool` message each; errors prefixed `ERROR:` (no error flag in this format) |
| reasoning | not used | `reasoning_content` is stored and **always sent back**: DeepSeek rejects a request with tools if earlier assistant messages lack it |
| cut off at the cap | `stop_reason == "max_tokens"` | `finish_reason == "length"` |
| tool arguments | already a JSON object | a JSON string; one that isn't a valid object becomes `ToolCall.arguments_error`, answered with an error result |
| default model | `claude-sonnet-5-5` | `deepseek-v4-pro` |

Both clients rely on their SDK for retries (up to 3 retries with exponential backoff on 429, 5xx and connection errors) and raise `ModelError` once those are used up. The output cap is `AGENT_MAX_OUTPUT_TOKENS` (default 16,000). Only generated tokens are billed, so the cap only bounds the worst case. Anthropic caps above 21,333 are rejected at startup, because the SDK refuses larger requests that don't stream.

## 10. Tools

A `Tool` is a name, a description, a JSON schema for its arguments, a Python function, and two flags: `requires_approval` and `idempotent`. `ToolRegistry` runs a call in two parts, so approval can happen in between:

- `check(call)`: unknown tool, arguments that weren't valid JSON, or arguments failing the schema → an error result. A failed check is recorded as `tool_skipped`; nothing is asked or run.
- `execute(call)`: run the function. An exception inside it becomes an error result.

Neither raises. The model sees the error and can correct itself, and the turn goes on.

The workspace tools resolve every path against the workspace root, following symlinks, and refuse anything that lands outside it. Output is capped at 20,000 characters. All three are idempotent; only `write_file` requires approval.

To give an agent other tools, build a `ToolRegistry` from your own `Tool`s and pass it to `Session.create` or `Session.open`.

## 11. The command-line chat

`cli.py: main` migrates the database, opens or creates the session (`--resume`), finishes an interrupted turn if there is one, then loops: read a line, `run_turn`, print the answer. `TerminalApprover` shows the tool's arguments and parses `y` / `a` / `n` / free text. Turn-level errors (`MaxStepsReached`, `ReplyTruncated`, `CompactionFailed`, `ModelError`) are printed and the chat continues; after a `ModelError`, `/resume` retries the turn.

## 12. Tests

`python -m pytest -q` (install with `pip install -e ".[dev]"`). Every test uses a scripted fake model (`tests/fake_model.py`), so the suite needs no API key and no network, except two smoke tests that run only when a real key is set. CI runs it on Python 3.11 and 3.13 for every pull request.

The tests that check the central guarantees:

| guarantee | test |
|---|---|
| an appended event survives SIGKILL; concurrent writers never share a seq | `test_events_durability.py` |
| killing the process inside a tool, during approval, or during a model call, then resuming, leaves the side effect done exactly once | `test_crash_resume.py` (real SIGKILL of a child process) |
| a denied tool never runs | `test_session_approvals.py` |
| a 100-step turn compacts repeatedly and still answers the original question | `test_session_compaction.py` |
| the live state always equals a replay of the log | in `test_session*.py` |

## 13. Limits

- **One process per session.** Nothing stops two processes from opening the same session at once; they would interleave events. Run one at a time.
- **No streaming.** Replies arrive whole, which also caps Anthropic requests at 21,333 output tokens.
- **Tools run one after another**, in the order the model listed them.
- **Token estimates are rough** (four characters per token) for messages the provider hasn't counted yet. The trigger leaves 20% headroom for that.
- **The context window default is conservative.** Set `AGENT_CONTEXT_WINDOW` to your model's real window so it doesn't compact earlier than needed.
