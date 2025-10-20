# AMM Status & Citation Standards v1.0

Purpose
- Define consistent, non-blocking status and citation emission across modules.
- Clarify the difference in semantics:
  - Status: streaming, sequential, “what is happening now.”
  - Citations: final, grouped summaries, “what happened.”
- Codify async fire-and-forget behavior and grouping patterns as implemented in:
  - [MRS/AMS_MRS_v4_Module.py](MRS/AMS_MRS_v4_Module.py)
- Related references:
  - [Reference/OW_Events.md](Reference/OW_Events.md)
  - [Reference/OW_Development.md](Reference/OW_Development.md)
  - Emoji taxonomy: [Reference/_Emoji_Taxonomy_1.2.md](Reference/_Emoji_Taxonomy_1.2.md)

1) Event channels and semantics
- Status events
  - One-way streaming updates rendered above the message content.
  - Used to show progress, activity, and transitions; short, present-tense descriptions.
  - Emitted frequently during long-running steps; must be non-blocking and async.
  - Canonical helper: [Filter._emit_status()](MRS/AMS_MRS_v4_Module.py:1705)
- Citation events
  - One-way final “what happened” summaries and references, grouped and nestable.
  - Emitted when a phase completes, or at session boundaries; must be non-blocking and async.
  - Canonical helper: [Filter._send_citation()](MRS/AMS_MRS_v4_Module.py:1719)

2) Non-blocking async requirements
- Always emit via an async wrapper that safely awaits the emitter and swallows UI exceptions:
  - [Filter._emit_event()](MRS/AMS_MRS_v4_Module.py:1695)
- Never block critical paths to wait on UI rendering; prefer background tasks for work, and emit status/citations within those tasks:
  - Background writer pattern: [Filter._memory_writer_task()](MRS/AMS_MRS_v4_Module.py:2243)
  - Spawn and detach with asyncio.create_task; manage lifecycle in a set and discard on completion
- Always send a final status to mark completion and then clear the status row:
  - Final message, brief delay, then clear with done=True: [await self._emit_status(...)](MRS/AMS_MRS_v4_Module.py:2301)

3) Valve contract (UI controls)
- Valves.show_status: bool = True
  - Controls whether status and citations are shown to the user
  - Checked in helpers before emitting
- Valves.show_assistant_mem_content: bool
  - When false, hide inline “command blocks” from the assistant output after processing and use citations instead
  - See outlet cleanup logic near: [Filter.outlet()](MRS/AMS_MRS_v4_Module.py:2355)

4) Canonical helpers (module-level API)
- Low-level emitter (do not bypass):
  - [Filter._emit_event()](MRS/AMS_MRS_v4_Module.py:1695)
    - Grabs __event_emitter__ from kwargs and awaits it, catching and logging warnings on failure
- Status helper:
  - [Filter._emit_status()](MRS/AMS_MRS_v4_Module.py:1705)
    - Payload shape:
      {
        "type": "status",
        "data": {
          "description": string,
          "done": bool
        }
      }
- Citation helper:
  - [Filter._send_citation()](MRS/AMS_MRS_v4_Module.py:1719)
    - Payload shape:
      {
        "type": "citation",
        "data": {
          "document": [string],     // one or more grouped blocks
          "metadata": [ { "source": string, "html": bool } ],
          "source": { "name": string }  // and optional URL if applicable
        }
      }
    - For grouped content, pre-compose a single document with newline-delimited citation lines

5) Status standards (what to send, when)
- Start of a phase
  - “💭 Trying to remember...”: [inlet start](MRS/AMS_MRS_v4_Module.py:2114)
  - “🧹 Deduplicating near-duplicate memories...”: [dedupe start](MRS/AMS_MRS_v4_Module.py:1843)
  - “⚙️ Backfilling N vector(s)…”: [backfill start](MRS/AMS_MRS_v4_Module.py:1801)
  - “💾 Saving memories…”: [write start](MRS/AMS_MRS_v4_Module.py:1304)
- During/after phase
  - Intermediate: “🧠 Thinking about memories...” (keep user informed) [inlet finally](MRS/AMS_MRS_v4_Module.py:2232)
  - Complete: “🏁 …complete” (or) “🚫 No … required” [finalize and clear](MRS/AMS_MRS_v4_Module.py:2296)
- Errors
  - Use concise, user-safe phrasing, emoji: ⚠️, then done=True
  - Examples:
    - “⚠️ Vector backfill error.” [backfill error](MRS/AMS_MRS_v4_Module.py:1833)
    - “⚠️ Deduplication error.” [dedupe error](MRS/AMS_MRS_v4_Module.py:2001)
    - “⚠️ Pruning task error.” [prune error](MRS/AMS_MRS_v4_Module.py:2026)
- Clearing rule
  - Always terminate a run with a deterministic final status message, pause briefly (e.g., ~4s), then clear line by sending an empty description with done=True: [final status clearing](MRS/AMS_MRS_v4_Module.py:2301)
- Hidden statuses
  - If needed, emit “hidden: True” per [Reference/OW_Events.md](Reference/OW_Events.md) guidance; otherwise omit to show in UI

6) Citation standards (what to send, how to group)
- Purpose
  - Record the outcomes as a compact, inspectable block users can click/expand
  - Use emoji bullets per taxonomy to encode the operation type
- Grouping/nesting pattern (single document, many lines)
  - Build one final list of lines for the phase (created + deleted + updated + purged, etc.), then send a single citation event
  - See grouping and emission: [writer finalization and send](MRS/AMS_MRS_v4_Module.py:2291)
- Titles and sources
  - Use succinct titles reflecting the phase:
    - “Memories Read” for context-surfaced items [title selection](MRS/AMS_MRS_v4_Module.py:1733)
    - “Memories Processed” for storage operations [title selection](MRS/AMS_MRS_v4_Module.py:1736)
  - Source path convention for module-generated citations:
    - module://{module_key}/{domain}/{snake_title}
    - Example: "module://mrs/memories/memories_read" [source path](MRS/AMS_MRS_v4_Module.py:1739)
  - Set metadata.html = False when sending plain text lines [metadata html flag](MRS/AMS_MRS_v4_Module.py:1748)
- Line-format grammar (pictorial bullets)
  - Canonical line builder: [Filter._format_citation_line()](MRS/AMS_MRS_v4_Module.py:1190)
  - Memory “read” items: use 🔍 and include bracketed meta and optional relevance
    - “🔍 [short_id | timestamp | flags?] content [relevance: x.xx]”
    - Construction reference: [Filter._format_memories_for_context()](MRS/AMS_MRS_v4_Module.py:1207)
    - Timestamp helper with timezone offset: [Filter._format_memory_timestamp()](MRS/AMS_MRS_v4_Module.py:626)
  - Created: 💾  Updated: ♻️  Deleted: 🗑️  Purged: 🔥
    - Examples of composing lines during operations:
      - Created: [log + line](MRS/AMS_MRS_v4_Module.py:1338)
      - Updated: [log + line](MRS/AMS_MRS_v4_Module.py:1351)
      - Deleted: [log + line](MRS/AMS_MRS_v4_Module.py:1365)
      - Purged: [prune line](MRS/AMS_MRS_v4_Module.py:518)
- Read vs Processed
  - Read: context surface (non-mutating), use title “Memories Read,” bullets=🔍
    - Emission in inlet flow: [send read citation](MRS/AMS_MRS_v4_Module.py:2208)
  - Processed: storage mutations (NEW/UPDATE/DELETE/PRUNE), use title “Memories Processed,” bullets as per operation
    - Emission at end of writer task: [send processed citation](MRS/AMS_MRS_v4_Module.py:2291)

7) Payload shapes (wire-level)
- Status event
  {
    "type": "status",
    "data": {
      "description": "🧹 Deduplicating near-duplicate memories...",
      "done": false
    }
  }
- Citation event (grouped)
  {
    "type": "citation",
    "data": {
      "document": [
        "🔍 [a1b2c3d4 | Sep18-2025_16:21] Meeting notes summary [relevance: 0.82]",
        "💾 Project plan v2 [importance: 0.70]",
        "🗑️ Outdated link ref",
        "♻️ Consolidated spec",
        "🔥 Session dump [importance: 0.15]"
      ],
      "metadata": [
        { "source": "module://mrs/memories/memories_processed", "html": false }
      ],
      "source": { "name": "Memories Processed" }
    }
  }

8) Placement guidelines
- Emit status at:
  - Phase start (immediately), before any network/IO-intensive step
  - During long work (periodic updates are ok; avoid spam)
  - At phase end (🏁 or 🚫), then clear
- Emit citations at:
  - After read/retrieval selection (inlet), with 🔍 lines [inlet send](MRS/AMS_MRS_v4_Module.py:2208)
  - After write/mutate phases (outlet/writer task), with 💾 / 🗑️ / ♻️ / 🔥 lines [writer finalize](MRS/AMS_MRS_v4_Module.py:2291)
- Never emit blocking UI calls within hot request paths; use background tasks for heavy post-processing [task spawn](MRS/AMS_MRS_v4_Module.py:2230)

9) Error handling and resilience
- Emission failures must not break core logic; catch and warn in the low-level emitter:
  - [Filter._emit_event()](MRS/AMS_MRS_v4_Module.py:1695)
- Prefer minimal, user-safe error descriptions in status; leave details to logger
- Gate emissions with Valves.show_status to allow administrators to toggle UI chatter:
  - [valve check in _emit_status()](MRS/AMS_MRS_v4_Module.py:1710)
  - [valve check at top of _send_citation()](MRS/AMS_MRS_v4_Module.py:1729)

10) Emoji taxonomy (visual language)
- Use icons from [Reference/_Emoji_Taxonomy_1.2.md](Reference/_Emoji_Taxonomy_1.2.md)
  - Status lifecycle: 💭 🧠 💾 🗑️ 🔥 🤷‍♂️ 🏁 ✅ ⚠️ 🧹 💬 🤔
  - Citation bullets: 🔍 💾 🗑️ ♻️ 🔥
- Be consistent: same operation type = same emoji across modules

11) Minimal implementation recipe (per module)
- Provide three helpers wired like MRS:
  - _emit_event(payload, **kwargs) → awaits __event_emitter__, catches exceptions [reference](MRS/AMS_MRS_v4_Module.py:1695)
  - _emit_status(description, done=False, **kwargs) → type="status" [reference](MRS/AMS_MRS_v4_Module.py:1705)
  - _send_citation(lines, user_id, citation_type="processed", **kwargs) → type="citation" [reference](MRS/AMS_MRS_v4_Module.py:1719)
- For heavy work, spawn a background task (asyncio.create_task), store the handle, and discard on completion [pattern](MRS/AMS_MRS_v4_Module.py:2221)
- At task end, emit final citation (if any), final status, brief pause, then clear [finalization](MRS/AMS_MRS_v4_Module.py:2291)

12) Reviewer checklist
- Async discipline
  - Emissions are awaited but do not block critical paths
  - Heavy phases run in background tasks; handles managed/cleared
- Status
  - Start/transition/end statuses present; final status is cleared
  - Uses taxonomy icons; concise text; no payloads/PII
- Citations
  - Grouped lines into a single document per phase
  - Title, source path, metadata.html are set appropriately
  - Line grammar via _format_citation_line; timestamps via helper
- Valves
  - show_status respected
  - show_assistant_mem_content applied where inline commands exist
- Logging
  - Errors/warnings logged on emitter failures (no hard crashes)

13) Worked examples (from MRS)
- Emit “read” citations after retrieval:
  - Build 🔍 lines: [Filter._format_memories_for_context()](MRS/AMS_MRS_v4_Module.py:1207)
  - Send citation with title “Memories Read”: [Filter._send_citation()](MRS/AMS_MRS_v4_Module.py:1719)
  - Invocation site: [inlet](MRS/AMS_MRS_v4_Module.py:2208)
- Emit “processed” citations after storage ops:
  - Build 💾/🗑️/♻️/🔥 lines during operations:
    - NEW: [created line](MRS/AMS_MRS_v4_Module.py:1334)
    - UPDATE: [updated line](MRS/AMS_MRS_v4_Module.py:1347)
    - DELETE: [deleted line](MRS/AMS_MRS_v4_Module.py:1361)
    - PURGE: [purged line](MRS/AMS_MRS_v4_Module.py:518)
  - Group and send once: [writer finalize](MRS/AMS_MRS_v4_Module.py:2291)
- Status clearing sequence at end of writer:
  - [final + clear](MRS/AMS_MRS_v4_Module.py:2301)

Decision notes
- Grouped, non-blocking emissions reduce UI traffic and preserve UX while guaranteeing visibility of key outcomes.
- A shared helper trio (_emit_event/_emit_status/_send_citation) keeps modules uniform and simplifies cold-start coding sessions.
- Titles, source paths, and emoji-coded bullets deliver consistent, scannable summaries across tools and modules.
14) Addendum: Status "done" flag behavior
- The "done" field is a visual flag only.
  - done: false → shows pulsing/animated status text to indicate ongoing activity.
  - done: true → shows still (non-pulsing) status text.
- This flag does not clear the status row. To clear:
  - Emit a subsequent status with an empty description and done: true (preferred), see [MRS/AMS_MRS_v4_Module.py](MRS/AMS_MRS_v4_Module.py:2301).
  - Alternatively, use the "hidden": true option per [Reference/OW_Events.md](Reference/OW_Events.md).