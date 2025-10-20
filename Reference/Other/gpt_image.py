"""
title: OpenAI Image Generator (GPTo\GPT-image-1)
description: Quick Pipe to enable image creation and editing with gpt-image-1
author: MorningBean.ai, contributions by MichaelMKenny and Coofdy
author_url: https://morningbean.ai
funding_url: FREE
version: 1.1.0
license: MIT
requirements: typing, pydantic, openai
environment_variables:
disclaimer: This pipe is provided as is without any guarantees.
            Please ensure that it meets your requirements.
            1.2.1 Performance Update
            1.2.0 Compatability with Open-Webui v0.6.7 allow setting BaseURL
            1.1.0 Removed Moderation on the editing endpoint
            0.4.0 Added support for multiple images in a single message
            0.3.2 Logic fix to only invoke editing when latets user message (prompt) contains an image.
            0.3.0 BugFix move to Non-Blocking
"""

import json
import random
import base64
import asyncio
import re
import tempfile
import os
import logging
from typing import List, AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
from openai import OpenAI


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEYS: str = Field(
            default="", description="OpenAI API Keys, comma-separated"
        )
        IMAGE_NUM: int = Field(default=2, description="Number of images (1-10)")
        IMAGE_SIZE: str = Field(
            default="1024x1024",
            description="Image size: 1024x1024, 1536x1024, 1024x1536, auto",
        )
        IMAGE_QUALITY: str = Field(
            default="auto", description="Image quality: high, medium, low, auto"
        )
        MODERATION: str = Field(
            default="auto", description="Moderation strictness: auto (default) or low"
        )
        BASE_URL: str = Field(
            default=None,
            description="Optional: Set OpenAI-compatible endpoint base URL (e.g. https://api.openai.com/v1 or a proxy URL). Leave empty for default.",
        )
        # Proxy-related fields have been removed

    def __init__(self):
        self.type = "manifold"
        self.name = "ChatGPT: "
        self.valves = self.Valves()
        self.emitter: Callable[[dict], Awaitable[None]] | None = None

    def _get_base_url(self) -> str | None:
        # Return base_url if set, else None
        val = getattr(self.valves, "BASE_URL", None)
        if val is not None and len(val.strip()) > 0:
            return val.strip()
        return None

    async def emit_status(self, message: str = "", done: bool = False):
        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )

    async def pipes(self) -> List[dict]:
        return [{"id": "gpt-image-1", "name": "GPT Image 1"}]

    def convert_message_to_prompt(self, messages: List[dict]) -> tuple[str, List[dict]]:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")

            # If content is a list (it can be mixed text and images)
            if isinstance(content, list):
                text_parts, image_data_list = [], []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            header, data = url.split(";base64,", 1)
                            mime = header.split("data:")[-1]
                            image_data_list.append({"mimeType": mime, "data": data})
                prompt = (
                    " ".join(text_parts).strip() or "Please edit the provided image(s)"
                )
                return prompt, image_data_list

            # If content is a plain string (search for embedded images in it)
            if isinstance(content, str):
                pattern = r"!\[[^\]]*\]\(data:([^;]+);base64,([^)]+)\)"
                matches = re.findall(pattern, content)
                image_data_list = [{"mimeType": m, "data": d} for m, d in matches]
                clean = (
                    re.sub(pattern, "", content).strip()
                    or "Please edit the provided image(s)"
                )
                return clean, image_data_list

        # Default case: No images found, return a default prompt
        return "Please edit the provided image(s)", []

    async def _run_blocking(self, fn: Callable, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    async def generate_image(
        self,
        prompt: str,
        model: str,
        n: int,
        size: str,
        quality: str,
    ) -> AsyncGenerator[str, None]:
        await self.emit_status("🖼️ Generating images...")
        key = random.choice(self.valves.OPENAI_API_KEYS.split(",")).strip()
        if not key:
            yield "Error: OPENAI_API_KEYS not set"
            return

        client = OpenAI(api_key=key, base_url=self._get_base_url())

        def _call_gen():
            return client.images.generate(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                quality=quality,
                moderation=self.valves.MODERATION,
            )

        try:
            resp = await self._run_blocking(_call_gen)
            for i, img in enumerate(resp.data, 1):
                yield f"![image_{i}](data:image/png;base64,{img.b64_json})"
            await self.emit_status("🎉 Image generation successful", done=True)
        except Exception as e:
            yield f"Error during image generation: {e}"
            await self.emit_status("❌ Image generation failed", done=True)

    async def edit_image(
        self,
        base64_images: List[dict],
        prompt: str,
        model: str,
        n: int,
        size: str,
        quality: str,
    ) -> AsyncGenerator[str, None]:
        await self.emit_status("✂️ Editing images...")
        key = random.choice(self.valves.OPENAI_API_KEYS.split(",")).strip()
        if not key:
            yield "Error: OPENAI_API_KEYS not set"
            return

        client = OpenAI(api_key=key, base_url=self._get_base_url())

        images_array = []
        for i, img_dict in enumerate(base64_images, start=1):
            try:
                data = base64.b64decode(img_dict["data"])
                if len(data) > 25 * 1024 * 1024:
                    raise ValueError("Image exceeds 25MB limit")

                suffix = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(img_dict["mimeType"])
                if not suffix:
                    raise ValueError(f"Unsupported format: {img_dict['mimeType']}")

                image = (f"file{i}", data, img_dict["mimeType"])

                images_array.append(image)
            except Exception as e:
                raise ValueError(f"Error decoding image: {e}")

        def _call_edit(images):
            return client.images.edit(
                model=model,
                image=images,
                prompt=prompt,
                n=n,
                size=size,
                extra_body={
                    "quality": quality,
                },
            )

        try:
            resp = await self._run_blocking(_call_edit, images_array)
            for i, img in enumerate(resp.data, 1):
                yield f"![image_{i}](data:image/png;base64,{img.b64_json})"
            await self.emit_status("🎉 Image edit successful", done=True)
        except Exception as e:
            yield f"Error during image edit: {e}"
            await self.emit_status("❌ Image edit failed", done=True)

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> AsyncGenerator[str, None]:
        self.emitter = __event_emitter__
        msgs = body.get("messages", [])
        model_id, n = "gpt-image-1", min(max(1, self.valves.IMAGE_NUM), 10)
        size, quality = self.valves.IMAGE_SIZE, self.valves.IMAGE_QUALITY
        prompt, imgs = self.convert_message_to_prompt(msgs)
        if imgs:
            async for out in self.edit_image(
                base64_images=imgs,
                prompt=prompt,
                model=model_id,
                n=n,
                size=size,
                quality=quality,
            ):
                yield out
        else:
            async for out in self.generate_image(
                prompt=prompt,
                model=model_id,
                n=n,
                size=size,
                quality=quality,
            ):
                yield out
