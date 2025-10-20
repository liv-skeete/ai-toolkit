"""
title: Auto-Memory Simplified (post-0.5)
author: shishka
version: as is
credits: caplescrest, Peter De-Ath (for their memory functions code)

# Posting as is, with the usage of !!Openrouter, reaplce if needed!! I have problems with ollama for some reason
    - Many of the current auto memory functions are not working after migration to 0.5 (may be not all), 
    - the point was to create a simplistic tool, that anyone can tweak, no huge prompts, adjust as you see fit.
    - extra point - i havent shared anything here yet, but i wanted to contribute, excuse "profanity", i'm fine with any critique

rip mark fisher
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Callable, Awaitable
from fastapi import Request
from open_webui.routers.memories import add_memory, AddMemoryForm
from open_webui.models.users import Users
import aiohttp
import json

class Valves(BaseModel):
    openrouter_api_url: str = Field(
        default="https://openrouter.ai/api/v1/chat/completions"
    )
    openrouter_api_key: str = Field(
        default=""
    )
    model: str = Field(default="openai/gpt-3.5-turbo")


class UserValves(BaseModel):
    show_status: bool = Field(default=True)


class Filter:
    def __init__(self):
        self.valves = Valves()
        self.user_valves = UserValves()

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __request__: Request,
        __user__: Optional[dict] = None,
    ) -> dict:
        print("outlet: Starting processing")

        if __event_emitter__:
            last_assistant_message = body["messages"][-1]
            user = Users.get_user_by_id(__user__["id"])

            if self.user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Adding to Memories", "done": False},
                    }
                )

            try:
                # Identify if the message should be added to memories
                should_add = await self.should_add_to_memory(
                    last_assistant_message["content"]
                )

                if should_add:
                    await add_memory(
                        request=__request__,
                        form_data=AddMemoryForm(
                            content=last_assistant_message["content"]
                        ),
                        user=user,
                    )

                    if self.user_valves.show_status:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": "Memory Saved", "done": True},
                            }
                        )
                else:
                    print("Message not identified as memory-worthy")

            except Exception as e:
                print(f"Error adding memory {str(e)}")
                if self.user_valves.show_status:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Error Adding Memory",
                                "done": True,
                            },
                        }
                    )
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "source": {"name": "Error:adding memory"},
                                "document": [str(e)],
                                "metadata": [{"source": "Add to Memory Action Button"}],
                            },
                        }
                    )

        return body

    async def should_add_to_memory(self, message: str) -> bool:
        system_prompt = "Analyze the following message and determine if it contains important information that should be remembered. Respond with 'true' if it should be added to memory, or 'false' if not."
        response = await self.query_openrouter_api(
            self.valves.model, system_prompt, message
        )
        return response.strip().lower() == "true"

    async def query_openrouter_api(
        self, model: str, system_prompt: str, prompt: str
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.openrouter_api_key}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.valves.openrouter_api_url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    json_content = await response.json()
            return json_content["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"query_openrouter_api: Error - {str(e)}")
            return "false"

    def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        print(f"inlet: Received body: {body}")
        print(f"inlet: User: {__user__}")
        return body
