"""
title: Memory Injection Filter
author: Raphael
author_url: https://github.com/raphael1-w
description: Inject user memories into system prompt, allowing selection of which model has access to memories. This works even if the user's memories setting is off.
required_open_webui_version: 0.5.0
version: 1.0.0
licence: MIT
"""

import json
from typing import Optional, Callable, Any
from pydantic import BaseModel, Field
from open_webui.models.memories import Memories
from datetime import datetime


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown state", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Filter:
    class Valves(BaseModel):
        PREPENDING_TEXT: str = Field(
            default="\n\nThe following is a list of stored memories, which are information related to the user, but are not part of this conversation. Consider the information within these memories as factual and use them to inform your responses, but do not bring up information from the memory bank unless it is directly related to the current conversation. Do not make up memories if the provided memory bank is empty. This is the list of memories: ",
            description="String to prepend before the JSON object of memories in the system prompt.",
        )
        SHOW_MEMORY_COUNT_EMITTER: bool = Field(
            default=False,
            description="Toggle activating an event emitter that shows how many memories are extracted.",
        )
        APPEND_ON_EMPTY: bool = Field(
            default=True,
            description="Inject prepending text and empty JSON object to system prompt even if memory list is empty.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> dict:
        print(f"inlet called: {body}")
        user_id = __user__.get("id")

        emitter = EventEmitter(__event_emitter__)

        if not user_id:
            print("User ID not provided.")
            return body

        user_memories = Memories.get_memories_by_user_id(user_id)

        num_memories = len(user_memories) if user_memories else 0

        # Conditionally emit memory count
        if self.valves.SHOW_MEMORY_COUNT_EMITTER and __event_emitter__:
            await emitter.emit(
                description=f"Extracted {num_memories} memories.",
                status="memory_extraction_complete",
                done=True,
            )

        memory_list = []
        if user_memories:
            for memory in user_memories:
                updated_at = memory.updated_at
                if isinstance(updated_at, int):
                    # Convert Unix timestamp to datetime object
                    updated_at = datetime.fromtimestamp(updated_at)

                updated_at_str = updated_at.isoformat() if updated_at else None

                memory_list.append(
                    {
                        "content": memory.content,
                        "updated_at": updated_at_str,
                    }
                )

        json_data = json.dumps(memory_list)

        # Inject into system prompt
        prepending_text = self.valves.PREPENDING_TEXT
        system_message = f"{prepending_text}\n{json_data}"

        # Conditionally append if memory list is empty
        if not user_memories and not self.valves.APPEND_ON_EMPTY:
            return body

        # Find the system message and append to it, if it exists
        for message in body["messages"]:
            if message["role"] == "system":
                message["content"] += f"\n{system_message}"
                print(f"Injected system message: {system_message}")
                return body

        # If no system message exists, create a new one
        body["messages"].insert(0, {"role": "system", "content": system_message})
        print(f"Injected system message: {system_message}")
        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Nothing is done in the outlet
        return body
