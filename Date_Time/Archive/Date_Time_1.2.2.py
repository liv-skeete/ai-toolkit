"""
title: Date And Time Module
description: Gives model current date and time context for each message. Don't forget to adjust the timezone in the settings.
author: Cody
version: 1.2.2
date: 2025-07-31
changelog: Date_Time/_changelog.md
"""

import datetime
import os
import logging
from pydantic import BaseModel, Field
from typing import Callable, Awaitable, Any, Optional

# Logger configuration
logger = logging.getLogger("date_time")
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler once
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Priority for the filter.")
        timezone_offset: float = Field(
            default=-7,
            description="Timezone offset in hours (e.g., -7 for UTC-7, 5.5 for UTC+5:30).",
        )

    def __init__(self):
        self.valves = self.Valves(
            **{
                "timezone_offset": float(os.getenv("DATETIME_TIMEZONE_OFFSET", "-7.5")),
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
        logger.info("Adding date and time context to message.")
        now_utc = datetime.datetime.utcnow()

        total_offset_minutes = self.valves.timezone_offset * 60
        now = now_utc + datetime.timedelta(minutes=total_offset_minutes)

        formatted_date = now.strftime("%B %d, %Y")
        formatted_time = now.strftime("%-I:%M%p").lower()
        day_of_week = now.strftime("%A")

        context = f"Current date is {day_of_week}, {formatted_date}, the user's time is {formatted_time}"

        datetime_message = {
            "role": "system",
            "content": f"Time Context: {context}. ",
        }

        if "messages" in body and isinstance(body["messages"], list):
            body["messages"].append(datetime_message)
        else:
            body["messages"] = [datetime_message]
            
        logger.info("Date and time context added successfully.")
        return body
