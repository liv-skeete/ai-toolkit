"""
title: User Information Retrieval Module
description: Retrieve and inject user information (ID, name, email, role, bio, gender, date of birth) into the model's context.
author: Cody
version: 0.2.0
changes: Added bio, gender, and age (calculated from date of birth) retrieval; Removed last_active_at due to ephemeral nature
"""

import asyncio
import logging
from typing import Callable, Awaitable, Dict
from pydantic import BaseModel, Field
from datetime import datetime

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
        ALLOW_BIO_RETRIEVAL: bool = Field(
            default=True, description="Whether to allow retrieval of user biography."
        )
        ALLOW_GENDER_RETRIEVAL: bool = Field(
            default=True, description="Whether to allow retrieval of user gender."
        )
        ALLOW_DATE_OF_BIRTH_RETRIEVAL: bool = Field(
            default=True, description="Whether to allow retrieval of user date of birth."
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
        logger.info("🚚 Retreiving user information...")
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
                            "description": "🚚 Retreiving user information...",
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

            # Process user bio
            if self.valves.ALLOW_BIO_RETRIEVAL and "bio" in __user__ and __user__["bio"]:
                result.append(f"Bio: {__user__['bio']}")
                retrieved_items.append("bio")
                logger.debug("Retrieved user bio")

            # Process user gender
            if self.valves.ALLOW_GENDER_RETRIEVAL and "gender" in __user__ and __user__["gender"]:
                result.append(f"Gender: {__user__['gender']}")
                retrieved_items.append("gender")
                logger.debug("Retrieved user gender")

            # Process user date of birth
            if self.valves.ALLOW_DATE_OF_BIRTH_RETRIEVAL and "date_of_birth" in __user__ and __user__["date_of_birth"]:
                try:
                    # Calculate age from date of birth
                    dob_value = __user__["date_of_birth"]
                    if isinstance(dob_value, str):
                        dob = datetime.strptime(dob_value, "%Y-%m-%d").date()
                    else:
                        # Assume it's already a date object
                        dob = dob_value
                    
                    today = datetime.today().date()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    result.append(f"Age: {age}")
                    retrieved_items.append("age")
                    logger.debug(f"Calculated user age: {age}")
                except (ValueError, AttributeError) as e:
                    # Handle case where date format is incorrect
                    result.append(f"Date of Birth: {__user__['date_of_birth']}")
                    retrieved_items.append("date-of-birth")
                    logger.warning(f"Failed to parse date of birth, using raw value: {str(e)}")


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
                            "description": f"🧑 User information retrieved",
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
