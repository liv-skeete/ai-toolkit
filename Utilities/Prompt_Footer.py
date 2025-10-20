"""
title: Prompt Footer
description: Injects a configurable footer message into the system prompt. This module should have the highest priority number to ensure it runs last.
author: Cody
version: 1.0.1
date: 2025-09-07
"""

import os
import logging
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Callable, Awaitable

# Logger configuration
logger = logging.getLogger("prompt_footer")
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler once
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=99,
            description="Priority for the filter. Should be the highest value to run last."
        )
        footer_message: str = Field(
            default="",
            description="The message to inject. A blank line can act as a separator."
        )

    def __init__(self):
        self.valves = self.Valves(
            **{
                "footer_message": os.getenv("PROMPT_FOOTER_MESSAGE", ""),
            }
        )

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __request__: Any,
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        logger.info("Prompt Footer module executed.")

        if self.valves.footer_message:
            footer_message = {
                "role": "system",
                "content": self.valves.footer_message,
            }
            
            if "messages" in body and isinstance(body["messages"], list):
                body["messages"].append(footer_message)
            else:
                body["messages"] = [footer_message]
        
        return body