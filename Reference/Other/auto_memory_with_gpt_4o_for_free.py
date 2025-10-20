"""
title: openai.ai Compatible OpenAI Pipe with auto memory
author: Raiyan Hasan
version: 0.1.0
license: MIT
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Callable, Awaitable, Any
import aiohttp
from fastapi.requests import Request
from open_webui.apps.webui.routers.memories import add_memory, AddMemoryForm
from open_webui.apps.webui.models.users import Users
import ast

from open_webui.main import webui_app


class Filter:
    class Valves(BaseModel):
        openai_api_url: str = Field(
            default="https://text.pollinations.ai/openai",
            description="Pollinations AI endpoint",
        )
        model: str = Field(
            default="pollinations/anywhere-v2",  # Default model for Pollinations
            description="Model to use to determine memory",
        )
        pass

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        print(f"inlet:{__name__}")
        print(f"inlet:body:{body}")
        print(f"inlet:user:{__user__}")
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        print(f"outlet:{__name__}")
        print(f"outlet:body:{body}")
        print(f"outlet:user:{__user__}")

        system_prompt = """You will be provided with a piece of text submitted by a user. Analyze the text to identify any information about the user that could be valuable to remember long-term. Do not include short-term information, such as the user's current query. You may infer interests based on the user's text.
            Extract only the useful information about the user and output it as a Python list of key details, where each detail is a string. If the text contains no useful information about the user, respond with an empty list ([]). Do not provide any commentary. Only provide the list.
            If the user explicitly requests to "remember" something, include that information in the output, even if it is not directly about the user. Do not store multiple copies of similar or overlapping information."""

        user_message = body["messages"][-2]["content"]
        memories = await self.query_pollinations_api(
            self.valves.model, system_prompt, user_message
        )
        if memories.startswith("[") and memories.endswith("]") and len(memories) != 2:
            user = Users.get_user_by_id(__user__["id"])
            result = await self.store_memory(memories, user)
            if __user__["valves"].show_status:
                if result:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Added memory",
                                "done": True,
                            },
                        }
                    )
                else:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Memory failed: {memories}",
                                "done": True,
                            },
                        }
                    )
        return body

    async def query_pollinations_api(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
    ) -> str:
        url = self.valves.openai_api_url
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 500,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    json_response = await response.json()

                    if not json_response.get("choices"):
                        raise ValueError("No choices in response")

                    return json_response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error querying Pollinations API: {str(e)}")
            return "[]"  # Return empty list on error

    async def store_memory(
        self,
        memories: str,
        user,
    ) -> bool:
        try:
            memory_list = ast.literal_eval(memories)
            for memory in memory_list:
                memory_obj = await add_memory(
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=AddMemoryForm(content=memory),
                    user=user,
                )
                print(f"Memory Added: {memory}")
            return True
        except Exception as e:
            print(f"Error adding memory {str(e)}")
            return False
