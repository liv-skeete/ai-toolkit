# AMM Logging Standards v1.0

Purpose
- Provide a minimal, repeatable logging pattern for all modules that supports:
  - Standalone, per-module loggers (isolated from system/global logging)
  - Two-tier messages: standard (always) vs verbose (only when enabled)
  - Safe, predictable formatting and redaction
  - Visual markers aligned with the AMM emoji taxonomy
- Baseline drawn from mature modules:
  - [MRS/AMS_MRS_v4_Module.py](MRS/AMS_MRS_v4_Module.py)
  - [SMR/SMR_v1_Module.py](SMR/SMR_v1_Module.py)
- Emoji taxonomy reference:
  - [Reference/_Emoji_Taxonomy_1.2.md](Reference/_Emoji_Taxonomy_1.2.md)

1) Logger topology (per-module, isolated)
- Logger name pattern: ams_<module_key> (snake_case). Examples:
  - MRS: [logging.getLogger()](MRS/AMS_MRS_v4_Module.py:34)
  - SMR: [logging.getLogger()](SMR/SMR_v1_Module.py:26)
- Isolation:
  - [logger.propagate](MRS/AMS_MRS_v4_Module.py:35) = False
- Level:
  - [logger.setLevel()](MRS/AMS_MRS_v4_Module.py:36) = logging.INFO for both standard and verbose messages (use the verbose valve to gate details, not DEBUG level)
- Handler policy (consistent formatting; no duplicates):
  - If the logger has existing handlers, clear them, then add exactly one StreamHandler
    - [logger.hasHandlers()](MRS/AMS_MRS_v4_Module.py:39); [logger.handlers.clear()](MRS/AMS_MRS_v4_Module.py:40)
    - [logging.StreamHandler()](MRS/AMS_MRS_v4_Module.py:42)
    - [logging.Formatter()](MRS/AMS_MRS_v4_Module.py:43) using "%(levelname)s | %(name)s | %(message)s"
    - [handler.setFormatter()](MRS/AMS_MRS_v4_Module.py:44), [handler.setLevel()](MRS/AMS_MRS_v4_Module.py:45), [logger.addHandler()](MRS/AMS_MRS_v4_Module.py:46)
- Rationale
  - Clearing handlers ensures consistent format even after reloads; INFO-level preserves visibility in default deployments; verbose gating is handled in code, not by level.

2) Valve contract (configuration)
- Every module that logs must expose a boolean valve:
  - Valves.verbose_logging: bool = False
- SMR example usage:
  - [Filter._log_message()](SMR/SMR_v1_Module.py:112) gates verbose via valves.verbose_logging
- MRS example usage:
  - [Filter._log_message()](MRS/AMS_MRS_v4_Module.py:552)

3) Logging API (two-tier pattern)
- Implement a unified helper in the module:
  - def _log_message(self, standard_msg: Optional[str], verbose_msg: Optional[str] = None) - Always log standard_msg at INFO; log verbose_msg at INFO only if valves.verbose_logging is True
  - Reference implementations:
    - [Filter._log_message()](SMR/SMR_v1_Module.py:112)
    - [Filter._log_message()](MRS/AMS_MRS_v4_Module.py:552)
- Utilities (recommended)
  - Single-line truncation for payload snippets:
    - [Filter._format_log_content()](MRS/AMS_MRS_v4_Module.py:563)
  - Multi-line truncation for verbose blocks:
    - [Filter._truncate_log_lines()](MRS/AMS_MRS_v4_Module.py:538)
  - Safe message sanitizer for multimodal content:
    - [Filter._sanitize_message_for_logging()](MRS/AMS_MRS_v4_Module.py:575)

4) Content policy: non-verbose vs verbose
- Non-verbose (always on)
  - Log significant events and state transitions only:
    - Start/finish of major operations, selected counts, selection decisions
    - High-level outcomes (e.g., "Added N memories", "Routing complete", "No candidates")
    - Short identifiers only (e.g., short_id, collection name), no raw payloads
  - No prompts, no full payloads, no PII/secrets
  - Examples:
    - Vector search summary: [self._log_message(...)](MRS/AMS_MRS_v4_Module.py:1009)
    - Selection decisions: [self._log_message(...)](MRS/AMS_MRS_v4_Module.py:1163)
    - Router selection: [self._log_message(...)](SMR/SMR_v1_Module.py:585)
- Verbose (gated by valves.verbose_logging)
  - Include details that materially aid diagnosis:
    - Sanitized payload sizes and bounded snippets
    - Key parameters, chunk indices, elapsed timings
    - Truncated prompts/content (use _truncate_log_lines and _format_log_content)
  - Examples:
    - Connector payload summary (model, temp, max_tokens): [self._log_message(None, verbose_msg=...)](MRS/AMS_MRS_v4_Module.py:776)
    - Vector search verbose query block: [verbose_msg with truncated body](MRS/AMS_MRS_v4_Module.py:976)
    - Rerank chunk diagnostics: [verbose chunk info](MRS/AMS_MRS_v4_Module.py:845)
    - Classifier system prompt (truncated): [verbose prompt logging](SMR/SMR_v1_Module.py:285)

5) Placement guidelines (where to log)
- Module initialization
  - One-time, non-verbose: "Module initialized"
  - Verbose: paths/models detected, lazy switches
  - Example: [ChromaDB client now at...](MRS/AMS_MRS_v4_Module.py:687)
- Configuration changes (valves)
  - Non-verbose: "Updating module configuration"
  - Verbose: For each key, show safe display value with redaction
  - Example: [Filter.update_valves() logging loop](MRS/AMS_MRS_v4_Module.py:302)
- External calls (I/O boundaries)
  - Before call (verbose): model, timeouts, batch sizes, key arguments (sanitized)
  - After call (non-verbose): success summary; (verbose): timing and sizes
  - Examples:
    - Platform chat completion: [await generate_chat_completion(...)](MRS/AMS_MRS_v4_Module.py:783)
    - Classifier call timing: [elapsed_ms recorded](SMR/SMR_v1_Module.py:298)
- Loops and chunked work
  - Log per-chunk or per-phase summaries, not per-item floods
  - Examples:
    - Rerank chunk N/M: [chunk diagnostics](MRS/AMS_MRS_v4_Module.py:845)
    - Dedupe clusters count: [Found N duplicate cluster(s)](MRS/AMS_MRS_v4_Module.py:1907)
- Errors and warnings
  - Use logger.warning for recoverable oddities; logger.error(..., exc_info=True) for errors
  - Examples:
    - Timeout: [logger.error(... timeout ...)](MRS/AMS_MRS_v4_Module.py:789)
    - JSON decode failure: [logger.error with exc_info](MRS/AMS_MRS_v4_Module.py:919)
    - Router parse issues: [logger.error parse failure](SMR/SMR_v1_Module.py:194)

6) Data hygiene and redaction
- Secrets/keys/tokens/passwords: never log raw; mask like "ab***yz"
  - Pattern in valves update:
    - [Filter.update_valves() redaction](MRS/AMS_MRS_v4_Module.py:323)
- Prompts: log at most a short prefix (e.g., first 50 chars) or truncated multi-line via _truncate_log_lines
  - [Prompt prefix handling](MRS/AMS_MRS_v4_Module.py:332)
- Multimodal user content: convert images to "[Image]" and flatten text
  - [Filter._sanitize_message_for_logging()](MRS/AMS_MRS_v4_Module.py:575)
- Long single-line fields: use _format_log_content for compactness
  - [Filter._format_log_content()](MRS/AMS_MRS_v4_Module.py:563)

7) Visual markers (emoji) usage
- Follow [Reference/_Emoji_Taxonomy_1.2.md](Reference/_Emoji_Taxonomy_1.2.md)
- In status emissions and citation messages:
  - memory_lifecycle (status): 💭 🧠 💾 🗑️ 🔥 🤷‍♂️ 🏁 ✅ ⚠️ 🧹 💬 🤔
  - memory_bullets (citations): 🔍 💾 🗑️ ♻️ 🔥
  - Examples:
    - Status: "💭 Trying to remember...", "🏁 Memory processing complete" (see MRS status flows)
    - Citations: "🔍 [id | timestamp] content", "💾 content" etc.
- In logger messages:
  - Optional, but when used, match taxonomy for the event type; e.g., "🔥 Purging X memories"
  - Examples:
    - [Purged memory:🔥 ...](MRS/AMS_MRS_v4_Module.py:526)
    - [Sending 'Memories Read' citation](MRS/AMS_MRS_v4_Module.py:1754)

8) Canonical Python template (drop-in)

Note: This template follows the conventions demonstrated in MRS/SMR references above. It uses the two-tier _log_message helper and the recommended handler/formatter setup.

```python
# logger_setup.py (example snippet inside your module)
import logging
from typing import Optional, Any, List

# Standalone, per-module logger
logger = logging.getLogger("<module_key>")
logger.propagate = False
logger.setLevel(logging.INFO)

# Ensure exactly one stream handler with canonical formatter
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
handler.setLevel(logging.INFO)
logger.addHandler(handler)

class Valves:
    # Add to your existing Pydantic/BaseModel valves in real modules
    verbose_logging: bool = False

class Module:
    def __init__(self) -> None:
        self.valves = Valves()
        self._log_message("Module initialized")

    def _log_message(self, standard_msg: Optional[str], verbose_msg: Optional[str] = None) -> None:
        if standard_msg:
            logger.info(standard_msg)
        if getattr(self.valves, "verbose_logging", False) and verbose_msg:
            logger.info(verbose_msg)

    # Utility: safe, compact single-line
    def _format_log_content(self, text: str, max_len: int = 200) -> str:
        if not text:
            return ""
        s = (text.replace("\n", " ").strip())[: max_len + 1]
        return s if len(s) <= max_len else (s[: max_len - 3] + "...")

    # Utility: safe multi-line truncation for verbose blocks
    @staticmethod
    def _truncate_log_lines(text: str, max_lines: int = 1000) -> str:
        lines = text.split("\n")
        if len(lines) <= max_lines:
            return text
        return "\n".join(lines[:max_lines] + [f"... [truncated, {len(lines)-max_lines} more lines omitted]"])

    # Utility: sanitize multimodal message content
    @staticmethod
    def _sanitize_message_for_logging(content: Any) -> str:
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        parts.append("[Image]")
            return " ".join(p for p in parts if p)
        return str(content)

    # Example usage pattern
    def do_work(self, payload: dict) -> None:
        # Non-verbose: high-level
        self._log_message("Starting work")
        # Verbose: sanitized details
        self._log_message(None, verbose_msg=f"Payload keys={list(payload.keys())}")

        try:
            # ... perform work ...
            self._log_message("Work complete")
        except Exception as e:
            logger.error(f"Work failed: {e}", exc_info=True)
```

9) Minimal reviewer checklist
- Logger
  - Name is <module_key>; propagate=False; level=INFO
  - Handlers cleared; exactly one StreamHandler; canonical formatter "%(levelname)s | %(name)s | %(message)s"
- Valves
  - Valves.verbose_logging present and honored
- Messages
  - Uses _log_message(standard_msg, verbose_msg)
  - Non-verbose: significant events only; no payloads/PII
  - Verbose: helpful, sanitized details with truncation helpers
  - Errors use logger.error(..., exc_info=True)
- Hygiene
  - Secrets masked, prompts truncated, images sanitized
  - Chunk/loop logs are summarized (no floods)
- Visuals
  - Status/citations follow [Reference/_Emoji_Taxonomy_1.2.md](Reference/_Emoji_Taxonomy_1.2.md)
  - Optional emojis in logs match event type if used

10) Reference index (selected)
- Logger setup:
  - [logging.getLogger()](MRS/AMS_MRS_v4_Module.py:34), [logger.propagate](MRS/AMS_MRS_v4_Module.py:35), [logger.setLevel()](MRS/AMS_MRS_v4_Module.py:36)
  - [logger.hasHandlers()](MRS/AMS_MRS_v4_Module.py:39), [logger.handlers.clear()](MRS/AMS_MRS_v4_Module.py:40)
  - [logging.StreamHandler()](MRS/AMS_MRS_v4_Module.py:42), [logging.Formatter()](MRS/AMS_MRS_v4_Module.py:43), [handler.setFormatter()](MRS/AMS_MRS_v4_Module.py:44), [handler.setLevel()](MRS/AMS_MRS_v4_Module.py:45), [logger.addHandler()](MRS/AMS_MRS_v4_Module.py:46)
  - Alternative guard pattern: [logger.hasHandlers()](SMR/SMR_v1_Module.py:29)
- Two-tier helper:
  - [Filter._log_message()](SMR/SMR_v1_Module.py:112), [Filter._log_message()](MRS/AMS_MRS_v4_Module.py:552)
- Utilities:
  - [Filter._truncate_log_lines()](MRS/AMS_MRS_v4_Module.py:538)
  - [Filter._format_log_content()](MRS/AMS_MRS_v4_Module.py:563)
  - [Filter._sanitize_message_for_logging()](MRS/AMS_MRS_v4_Module.py:575)
- Examples of placement:
  - Connector payload diag: [verbose pre-call](MRS/AMS_MRS_v4_Module.py:776)
  - Vector search std/verbose: [standard + verbose query](MRS/AMS_MRS_v4_Module.py:976)
  - Chunk diagnostics: [rerank chunk](MRS/AMS_MRS_v4_Module.py:845)
  - Selection/fallback notes: [vector vs LLM decisions](MRS/AMS_MRS_v4_Module.py:1159)
  - Router model selection: [selected model](SMR/SMR_v1_Module.py:585)

Decision notes
- INFO-level logging with a verbose valve yields consistent behavior across modules and environments, avoids reliance on external logger level configuration, and keeps standard logs useful in production.
- Clearing handlers on construction ensures consistent formatting even after hot reloads or repeated instantiation.