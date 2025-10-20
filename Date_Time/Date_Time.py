"""
title: Date And Time Module
description: Retrieve and inject current date and time context into the model's context.
author: Cody
version: 1.3.0
date: 2025-10-09
changelog: Date_Time/_changelog.md
"""

import datetime
import logging
import os
from typing import Callable, Awaitable, Dict, Any, Optional
from pydantic import BaseModel, Field


# Logger configuration
logger = logging.getLogger("date_time")
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
    """Date and Time Context Injection module for Open WebUI.
    
    This filter retrieves the current date and time and injects it into the model's context.
    All information retrieval is configurable via the Valves settings.
    """

    class Valves(BaseModel):
        """Configuration valves for the Date and Time Filter."""
        
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
        
        # Time configuration
        TIMEZONE_OFFSET: float = Field(
            default=-7.0,
            description="Timezone offset in hours (e.g., -7 for UTC-7, 5.5 for UTC+5:30).",
        )

    def __init__(self):
        self.valves = self.Valves()
        
        # Set initial logging level based on VERBOSE_LOGGING setting
        if self.valves.VERBOSE_LOGGING:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def _validate_input_data(self, body: Dict[str, Any]) -> bool:
        """Validate input data structure and content.
        
        Args:
            body: The message body from Open WebUI.
            
        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(body, dict):
            logger.warning("Body is not a dictionary")
            return False
            
        return True

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __request__: Any = None,
        __user__: Optional[Dict[str, Any]] = None,
        __model__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve current date and time and inject it into the model's context.
        
        Args:
            body: The message body from Open WebUI.
            __event_emitter__: Function to emit status updates.
            __request__: Request information from Open WebUI.
            __user__: User information dictionary from Open WebUI.
            __model__: Model information from Open WebUI.
            
        Returns:
            Modified body with date and time information injected into system context.
            
        Raises:
            RuntimeError: If date and time retrieval fails.
            ValueError: If input validation fails.
        """
        # Update logger level based on valve setting
        if self.valves.VERBOSE_LOGGING:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            
        logger.info("Retrieving date and time information...")
        
        try:
            # Validate input data
            if not self._validate_input_data(body):
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
                body = {}

            # Send initial status update if enabled
            if self.valves.SHOW_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "in_progress",
                            "description": "🚚 Retrieving date and time...",
                            "done": False,
                        },
                    }
                )

            # Get current UTC time
            now_utc = datetime.datetime.utcnow()
            logger.debug(f"UTC time: {now_utc}")

            # Apply timezone offset
            total_offset_minutes = self.valves.TIMEZONE_OFFSET * 60
            now = now_utc + datetime.timedelta(minutes=total_offset_minutes)
            logger.debug(f"Local time with offset: {now}")

            # Format date and time components
            formatted_date = now.strftime("%B %d, %Y")
            formatted_time = now.strftime("%-I:%M%p").lower()
            day_of_week = now.strftime("%A")
            
            logger.debug(f"Formatted components - Date: {formatted_date}, Time: {formatted_time}, Day: {day_of_week}")

            # Create context string
            context = f"Current date is {day_of_week}, {formatted_date}, the user's time is {formatted_time}"
            logger.debug(f"Context string: {context}")
            
            # Sanitize context for logging
            sanitized_context = sanitize_log_message(context)

            # Send completion status update if enabled
            if self.valves.SHOW_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "complete",
                            "description": "🕒 Date and time retrieved",
                            "done": True,
                        },
                    }
                )

            # Inject into system context
            datetime_message = {
                "role": "system",
                "content": f"Time Context: {context}.",
            }
            
            # Add to messages
            if "messages" in body and isinstance(body["messages"], list):
                body["messages"].append(datetime_message)
            else:
                body["messages"] = [datetime_message]
                
            logger.info(f"Successfully injected date/time context into model: {sanitized_context}")

        except Exception as e:
            error_msg = f"Date and time retrieval failed: {str(e)}"
            logger.error(error_msg)
            # Send error status update if enabled
            if self.valves.SHOW_STATUS:
                try:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "error",
                                "description": "Date and time retrieval failed",
                                "done": True,
                            },
                        }
                    )
                except Exception as emitter_error:
                    logger.warning(f"Failed to send error status update: {str(emitter_error)}")
            raise RuntimeError(error_msg) from e

        return body
