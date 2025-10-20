"""
title: Context Manager Module
description: Truncate chat context with token limit and max turns; preserves critical MRS system memory. Log chat turn data with 'log_type'.
author: Cody
version: 1.1.0
date: 2025-10-10
changelog: Context_Manager/_changelog.md
"""

import tiktoken
import time
import logging
import json
from typing import Optional, Callable, Any, Awaitable, Dict, List, Tuple, Union
from pydantic import BaseModel, Field


# Logger configuration
logger = logging.getLogger("context_manager")
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler once
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    # Explicitly set handler level to match logger
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


def sanitize_log_message(message: str, max_length: int = 1000) -> str:
    """Sanitize and truncate log messages to prevent console overflow.
    
    Args:
        message: The log message to sanitize.
        max_length: Maximum length of the log message.
        
    Returns:
        Sanitized and truncated log message.
    """
    if not message:
        return ""
    
    # Truncate if too long
    if len(message) > max_length:
        message = message[:max_length] + "...[TRUNCATED]"
    
    return message


class Filter:
    """Context Manager module for Open WebUI.
    
    This filter truncates chat context with token limit and max turns while preserving
    critical MRS system memory. It also logs chat turn data with 'log_type'.
    """

    class Valves(BaseModel):
        """Configuration valves for the Context Manager Filter."""
        
        # Processing priority
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        
        # Status control
        SHOW_STATUS: bool = Field(
            default=True,
            description="Show real-time status updates via event emitter.",
        )
        
        VERBOSE_LOGGING: bool = Field(
            default=False,
            description="Enable verbose logging for detailed diagnostic output.",
        )
        
        # Context management settings
        MAX_TURNS: int = Field(
            default=20,
            description="Number of conversation turns to retain. Set '0' for unlimited",
        )
        
        TOKEN_LIMIT: int = Field(
            default=16000,
            description="Number of token limit to retain. Set '0' for unlimited",
        )

    def __init__(self):
        self.valves = self.Valves()
        
        # Set initial logging level based on VERBOSE_LOGGING setting
        if self.valves.VERBOSE_LOGGING:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            
        self.limit_exceeded = False
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.input_tokens = 0
        self.output_tokens = 0
        self.user = None
        self.model_base = None
        self.model_name = None
        self.start_time = None
        self.elapsed_time = None
        self.input_message_count = None

        # Internal tracking for trimming diagnostics
        self._turns_trim_info = None
        self._tokens_trim_info = None

    def _validate_input_data(self, body: Dict[str, Any], __user__: Optional[Dict[str, Any]]) -> bool:
        """Validate input data structure and content.
        
        Args:
            body: The message body from Open WebUI.
            __user__: User information dictionary from Open WebUI.
            
        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(body, dict):
            logger.warning("Body is not a dictionary")
            return False
            
        if "messages" not in body or not isinstance(body["messages"], list):
            logger.warning("Messages not found in body or not a list")
            return False
            
        return True

    def _is_pinned_system(self, msg: Dict[str, Any]) -> bool:
        """Return True if this is the MRS-injected system memory message that should be preserved.
        
        Args:
            msg: Message dictionary to check.
            
        Returns:
            True if message is a pinned system message, False otherwise.
        """
        try:
            if msg.get("role") != "system":
                return False
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.startswith("User Information (sorted by relevance):")
            return False
        except Exception as e:
            logger.debug(f"Error checking if message is pinned system: {str(e)}")
            return False

    def _extract_user_identifier(self, __user__: Optional[Dict[str, Any]]) -> Optional[str]:
        """Safely extract a user identifier for logging: prefer email, fall back to username or id.
        
        Args:
            __user__: User information dictionary from Open WebUI.
            
        Returns:
            User identifier string or None if not found.
        """
        if not __user__:
            return None
        try:
            if isinstance(__user__, dict):
                return __user__.get("email") or __user__.get("username") or __user__.get("id")
            return getattr(__user__, "email", None) or getattr(__user__, "username", None) or getattr(__user__, "id", None)
        except Exception as e:
            logger.debug(f"Error extracting user identifier: {str(e)}")
            return None

    def _infer_model_fields(self, body: Dict[str, Any], __model__: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """Infer model_base (id) and model_name with preference for SMR breadcrumb.
        
        Args:
            body: The message body from Open WebUI.
            __model__: Model information from Open WebUI.
            
        Returns:
            Tuple of (model_base, model_name).
        """
        model_id: Optional[str] = None
        model_name: Optional[str] = None
        try:
            meta = body.get("metadata") or {}
            if isinstance(meta, dict):
                router = meta.get("router") or {}
                if isinstance(router, dict):
                    # Prefer SMR breadcrumb
                    model_id = router.get("model_id") or model_id
                    model_name = router.get("model") or model_name  # route key as name if available
            # Fallbacks
            if not model_id:
                if isinstance(body, dict):
                    model_id = body.get("model") or model_id
                if not model_id and isinstance(__model__, dict):
                    model_id = __model__.get("id")
            if not model_name:
                if isinstance(__model__, dict):
                    model_name = __model__.get("name")
                if not model_name and model_id:
                    model_name = str(model_id)
        except Exception as e:
            logger.debug(f"Error inferring model fields: {str(e)}")
            if isinstance(body, dict):
                model_id = body.get("model", None)
            if isinstance(__model__, dict):
                model_id = model_id or __model__.get("id")
                model_name = __model__.get("name") or model_name
            if not model_name and model_id:
                model_name = str(model_id)
        return model_id, model_name

    def log_chat_turn(self) -> None:
        """Log data for a single chat turn."""
        if all(
            [
                self.user,
                self.model_base,
                self.model_name,
                self.input_tokens is not None,
                self.output_tokens is not None,
                self.elapsed_time is not None,
                self.input_message_count is not None,
            ]
        ):
            log_data = {
                "log_type": "chat_turn",
                "user": self.user,
                "model_base": self.model_base,
                "model_name": self.model_name,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "elapsed_seconds": round(self.elapsed_time, 0),
                "input_message_count": self.input_message_count,
            }
            # Sanitize log data before output
            sanitized_data = {k: sanitize_log_message(str(v)) for k, v in log_data.items()}
            logger.info(json.dumps(sanitized_data))

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __model__: Optional[Dict[str, Any]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Truncate chat context with token and turn limits; preserves critical MRS system memory.
        
        Args:
            body: The message body from Open WebUI.
            __event_emitter__: Function to emit status updates.
            __model__: Model information from Open WebUI.
            __user__: User information dictionary from Open WebUI.
            
        Returns:
            Modified body with truncated messages.
            
        Raises:
            RuntimeError: If context management fails.
            ValueError: If input validation fails.
        """
        # Update logger level based on valve setting
        if self.valves.VERBOSE_LOGGING:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            
        logger.info("Managing chat context...")
        
        try:
            # Validate input data
            if not self._validate_input_data(body, __user__):
                logger.warning("Invalid input data structure")
                if self.valves.SHOW_STATUS:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "error",
                                "description": "Invalid input data structure",
                                "done": True,
                            },
                        }
                    )
                # Continue with empty body rather than failing completely
                body = {"messages": []}

            # Send initial status update if enabled
            if self.valves.SHOW_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "in_progress",
                            "description": "🔍 Checking chat size...",
                            "done": False,
                        },
                    }
                )

            messages = body["messages"]
            chat_messages = messages[:]
            self.limit_exceeded = False
            # Reset trim diagnostics
            self._turns_trim_info = None
            self._tokens_trim_info = None

            # Pre-trim diagnostics
            user_ident = self._extract_user_identifier(__user__)
            try:
                pre_count = len(chat_messages)
                pre_tokens = sum(self.count_text_tokens(m) for m in chat_messages)
            except Exception as e:
                logger.debug(f"Error calculating pre-trim diagnostics: {str(e)}")
                pre_count = len(chat_messages)
                pre_tokens = 0
                
            logger.info(
                f"Context trimming started for user {user_ident or 'unknown'}: "
                f"messages={pre_count}, tokens={pre_tokens}, "
                f"max_turns={self.valves.MAX_TURNS}, token_limit={self.valves.TOKEN_LIMIT}"
            )

            # Apply trims
            chat_messages = self.truncate_turns(chat_messages)
            chat_messages = self.truncate_tokens(chat_messages)

            # Post-trim diagnostics
            try:
                post_count = len(chat_messages)
                post_tokens = sum(self.count_text_tokens(m) for m in chat_messages)
            except Exception as e:
                logger.debug(f"Error calculating post-trim diagnostics: {str(e)}")
                post_count = len(chat_messages)
                post_tokens = 0

            # Emit detailed trim logs if applied
            if isinstance(self._turns_trim_info, dict):
                info = self._turns_trim_info
                logger.info(
                    "Turn trim: "
                    f"current_turns={info.get('current_turns')}, "
                    f"max_turns={info.get('max_turns')}, "
                    f"removed_messages={info.get('removed')}, "
                    f"removed_turns={info.get('removed_turns')}, "
                    f"pinned_preserved={info.get('pinned_preserved')}"
                )
            if isinstance(self._tokens_trim_info, dict):
                info = self._tokens_trim_info
                if (info.get("dropped") or 0) > 0:
                    logger.info(
                        "Token trim: "
                        f"budget={info.get('budget')}, "
                        f"pinned_tokens={info.get('pinned_tokens')}, "
                        f"kept={info.get('kept')}, "
                        f"dropped={info.get('dropped')}"
                    )

            logger.info(
                f"Context trimming complete: messages={post_count}, tokens={post_tokens}, "
                f"limit_exceeded={self.limit_exceeded}"
            )
            
            await self.show_exceeded_status(__event_emitter__, len(chat_messages))
            self.init_log_data(user_ident, chat_messages)
            body["messages"] = chat_messages
            
            # Send completion status update if enabled
            if self.valves.SHOW_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "complete",
                            "description": "✅ Chat check complete",
                            "done": True,
                        },
                    }
                )

        except Exception as e:
            error_msg = f"Context management failed: {str(e)}"
            logger.error(error_msg)
            # Send error status update if enabled
            if self.valves.SHOW_STATUS:
                try:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "error",
                                "description": "Context management failed",
                                "done": True,
                            },
                        }
                    )
                except Exception as emitter_error:
                    logger.warning(f"Failed to send error status update: {str(emitter_error)}")
            raise RuntimeError(error_msg) from e

        return body

    def truncate_turns(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trim by conversation turns while preserving pinned system messages (e.g., MRS memory block).
        Only user/assistant roles count towards turn calculation; system messages are excluded.
        
        Args:
            messages: List of message dictionaries.
            
        Returns:
            Truncated list of messages.
        """
        result = messages
        self._turns_trim_info = None
        if self.valves.MAX_TURNS > 0 and messages:
            # Count only user/assistant messages
            ua_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
            current_turns = max(0, (len(ua_msgs) - 1) // 2)
            if current_turns > self.valves.MAX_TURNS:
                sent_msg_count = self.valves.MAX_TURNS * 2 + 1
                cut_start = len(messages) - sent_msg_count
                if cut_start < 0:
                    cut_start = 0
                # Preserve pinned system messages that appear before the cut
                pinned_prefix = [m for m in messages[:cut_start] if self._is_pinned_system(m)]
                tail = messages[cut_start:]
                # Messages removed are those before cut that were not pinned and not included in tail
                removed = cut_start - len(pinned_prefix)
                # Avoid duplicates if any pinned also exist in tail
                result = pinned_prefix + tail
                self.limit_exceeded = True
                self._turns_trim_info = {
                    "current_turns": current_turns,
                    "max_turns": self.valves.MAX_TURNS,
                    "removed": max(0, removed),
                    "removed_turns": max(0, current_turns - self.valves.MAX_TURNS),
                    "pinned_preserved": len(pinned_prefix),
                }
        return result

    def truncate_tokens(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Trim by token budget while always preserving:
        - Pinned system memory from MRS
        - The latest user message (never dropped)
        
        Args:
            messages: List of message dictionaries.
            
        Returns:
            Truncated list of messages.
        """
        filter_messages = messages
        self._tokens_trim_info = None
        if self.valves.TOKEN_LIMIT > 0 and messages:
            budget = self.valves.TOKEN_LIMIT
            # Collect pinned system messages and reserve their token budget
            pinned = [m for m in messages if self._is_pinned_system(m)]
            pinned_ids = set(id(m) for m in pinned)
            pinned_tokens = sum(self.count_text_tokens(m) for m in pinned)

            non_pinned_result = []
            current_toks = pinned_tokens

            for msg in reversed(messages):
                if id(msg) in pinned_ids:
                    continue  # already reserved
                toks = self.count_text_tokens(msg)
                role = msg.get("role", "")
                # Latest user message should not be dropped even if it exceeds the limit
                if (current_toks + toks > budget) and (role != "user"):
                    self.limit_exceeded = True
                    break
                non_pinned_result.insert(0, msg)
                current_toks += toks

            # Final ordering: pinned (front) + retained non-pinned in original order
            filter_messages = pinned + non_pinned_result

            dropped = max(0, len(messages) - len(filter_messages))
            self._tokens_trim_info = {
                "budget": budget,
                "pinned_tokens": pinned_tokens,
                "kept": len(filter_messages),
                "dropped": dropped,
            }
        return filter_messages

    def init_log_data(self, user_email: str, messages: List[Dict[str, Any]]) -> None:
        """Initialize logging data for chat turn.
        
        Args:
            user_email: User email identifier.
            messages: List of message dictionaries.
        """
        self.user = user_email
        self.input_tokens = 0
        self.start_time = time.time()
        for msg in messages:
            self.input_tokens += self.count_text_tokens(msg)
        self.input_message_count = len(messages)

    def outlet(
        self,
        body: Dict[str, Any],
        __model__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process outgoing message and log chat turn data.
        
        Args:
            body: The message body from Open WebUI.
            __model__: Model information from Open WebUI.
            
        Returns:
            Unmodified body.
        """
        self.output_tokens = self.count_text_tokens(body["messages"][-1])
        model_base, model_name = self._infer_model_fields(body, __model__)
        self.model_base = model_base
        self.model_name = model_name
        end_time = time.time()
        if self.start_time:
            self.elapsed_time = end_time - self.start_time
        self.log_chat_turn()
        return body

    async def show_exceeded_status(
        self, __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]], message_count: int
    ) -> None:
        """Show status message when context limits are exceeded.
        
        Args:
            __event_emitter__: Function to emit status updates.
            message_count: Number of messages being kept.
        """
        if self.limit_exceeded and self.valves.SHOW_STATUS:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Context limit reached - keeping last {message_count} messages",
                        "done": True,
                    },
                }
            )

    def count_text_tokens(self, msg: Dict[str, Any]) -> int:
        """Count tokens in message content.
        
        Args:
            msg: Message dictionary.
            
        Returns:
            Number of tokens in the message.
        """
        content = msg.get("content", "")
        total_tokens = 0

        if isinstance(content, list):
            # Handle multi-modal content
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    total_tokens += len(self.encoding.encode(text))
        elif isinstance(content, str):
            # Handle text-only content
            total_tokens = len(self.encoding.encode(content))
        else:
            # Handle unexpected content types
            total_tokens = 0

        return total_tokens
