"""
title: User Information Retrieval Module
description: Retrieve and inject user information (ID, name, email, role) into the model's context.
author: Cody
version: 0.1.2
changes: Restructured valves into Valves and UserValves classes
"""

import asyncio
import logging
from typing import Callable, Awaitable, Dict
from pydantic import BaseModel, Field

# Logger configuration
logger = logging.getLogger("user_info")
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


class Filter:
    """User Information Retrieval module for Open WebUI."""

    class Valves(BaseModel):
        # Set processing priority
        priority: int = Field(
            default=1,
            description="Priority level for the filter operations.",
        )
        DISPLAY_EVENT_EMITTERS: bool = Field(
            default=True,
            description="Whether to display event emitters during user information retrieval.",
        )
        ALLOW_ID_RETRIEVAL: bool = Field(
            default=True, description="Whether to allow retrieval of user IDs."
        )
        ALLOW_NAME_RETRIEVAL: bool = Field(
            default=True, description="Whether to allow retrieval of user names."
        )
        ALLOW_EMAIL_RETRIEVAL: bool = Field(
            default=True, description="Whether to allow retrieval of user emails."
        )
        ALLOW_ROLE_RETRIEVAL: bool = Field(
            default=True, description="Whether to allow retrieval of user roles."
        )

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {},
    ) -> dict:
        """
        Retrieve user information and inject it into the model's context, respecting Valves settings.
        """
        logger.info("Starting user information retrieval")
        result = []
        retrieved_items = []

        try:
            # Determine whether to display event emitters
            display_event_emitters = self.valves.DISPLAY_EVENT_EMITTERS

            if display_event_emitters:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "in_progress",
                            "description": "Starting user information retrieval",
                            "done": False,
                        },
                    }
                )

            # Process user ID
            if self.valves.ALLOW_ID_RETRIEVAL and "id" in __user__:
                result.append(f"ID: {__user__['id']}")
                retrieved_items.append("ID")
                logger.debug("Retrieved user ID")

            # Process user name
            if self.valves.ALLOW_NAME_RETRIEVAL and "name" in __user__:
                result.append(f"User: {__user__['name']}")
                retrieved_items.append("name")
                logger.debug("Retrieved user name")

            # Process user email
            if self.valves.ALLOW_EMAIL_RETRIEVAL and "email" in __user__:
                result.append(f"Email: {__user__['email']}")
                retrieved_items.append("email")
                logger.debug("Retrieved user email")

            # Process user role
            if self.valves.ALLOW_ROLE_RETRIEVAL and "role" in __user__:
                result.append(f"Role: {__user__['role']}")
                retrieved_items.append("role")
                logger.debug("Retrieved user role")

            # Handle empty result case
            if not result:
                result = ["User: Unknown"]
                retrieved_items = ["no data"]
                logger.warning("No user information available")

            final_result = " | ".join(result)

            if display_event_emitters:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "complete",
                            "description": f"🧑 User information retrieval completed. Retrieved: {', '.join(retrieved_items)}",
                            "done": True,
                        },
                    }
                )

            # Inject into system context
            system_message = {
                "role": "system",
                "content": f"User Information: {final_result}",
            }
            body.setdefault("messages", []).append(system_message)
            logger.info("Successfully injected user info into context")

        except Exception as e:
            logger.error("Failed to retrieve user information: %s", str(e))
            raise

        return body
