"""
title: Venice Image (Assistant-Only)
description: Image generation, editing, and upscaling via Venice.ai API. This version is designed to be called by the assistant directly.
author: Cody
version: 2.2.0
"""

import base64
import asyncio
import re
import logging
from typing import (
    List,
    AsyncGenerator,
    Callable,
    Awaitable,
    Optional,
    Dict,
    Any,
    Tuple,
    Literal,
)

import httpx
from pydantic import BaseModel, Field

# Logger configuration
logger = logging.getLogger("venice_image")
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
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations",
        )
         # System settings
        VENICE_BASE_URL: Optional[str] = Field(
            default="https://api.venice.ai/api/v1",
            description="Venice API base URL",
        )        
        VENICE_API_KEY: str = Field(
            default="", description="Venice API key (single key)"
        )
        EMBED_IMAGES_AS_URL: bool = Field(
            default=True,
            description="Upload images to pictshare and embed as URLs",
        )
        PICTSHARE_URL: Optional[str] = Field(
            default="http://pictshare:4000",
            description="URL for pictshare service (internal or public)",
        )
        TIMEOUT_SECONDS: int = Field(
            default=180, description="API request timeout in seconds"
        )
        # Image defaults
        SAFE_MODE: bool = Field(
            default=False, description="Safe-mode for image generation"
        )
        EMBED_EXIF_METADATA: bool = Field(
            default=False, description="Embed EXIF metadata"
        )
        HIDE_WATERMARK: bool = Field(
            default=True, description="Hide watermark if supported"
        )
        OUTPUT_FORMAT: Literal["webp", "jpeg", "png"] = Field(
            default="webp", description="webp | jpeg | png"
        )
        # Generation defaults
        MODEL: str = Field(
            default="hidream", description="Required: 'hidream' 'qwen-image'  'lustify-sdxl' 'venice-sd35' 'wai-Illustrious'"
        )
        STYLE_PRESET: Optional[str] = Field(
            default=None, description="Optional style preset for image generation"
        )
        NEGATIVE_PROMPT: Optional[str] = Field(
            default=None, description="Optional negative prompt"
        )
        IMAGE_NUM: int = Field(
            default=1, description="Number of images to generate"
        )    
        IMAGE_SIZE: str = Field(
            default="auto", description="WxH (e.g., 1024x1024) or 'auto'"
        )
        CFG_SCALE: float = Field(
            default=7.5,
            description="Prompt adherence 0-20",
        )
        LORA_STRENGTH: int = Field(
            default=50, description="LoRA strength (0-100)"
        )
        STEPS: int = Field(
            default=25, description="Steps for generation"
        )
        SEED: Optional[int] = Field(
            default=None, description="Optional deterministic seed"
        )
        # Upscale defaults
        ENHANCE: bool = Field(
            default=False, description="Enhance upscaled image"
        )
        ENHANCE_PROMPT: Optional[str] = Field(
            default=None, description="Optional enhance prompt for upscale"
        )
        UPSCALE_FACTOR: Literal[2, 4] = Field(
            default=2, description="Default upscale factor (2 or 4)"
        )
        ENHANCE_CREATIVITY: float = Field(
            default=0.5, description="Enhance creativity (0..1)"
        )
        REPLICATION: float = Field(
            default=0.35, description="How strongly lines and noise are preserved (0..1)"
        )

    def __init__(self):
        self.type = "filter"
        self.valves = self.Valves()
        self.emitter: Optional[Callable[[dict], Awaitable[None]]] = None

    def _get_base_url(self) -> str:
        val = getattr(self.valves, "VENICE_BASE_URL", None)
        return (
            val.strip()
            if isinstance(val, str) and val.strip()
            else "https://api.venice.ai/api/v1"
        )

    def _get_api_key(self) -> str:
        key = getattr(self.valves, "VENICE_API_KEY", "").strip()
        return key

    async def emit_status(self, message: str = "", done: bool = False):
        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )


    @staticmethod
    def _parse_size(size_str: str) -> Tuple[Optional[int], Optional[int]]:
        if not size_str or size_str.lower() == "auto":
            return None, None
        m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", size_str)
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    @staticmethod
    def _mime_for_format(fmt: str) -> str:
        f = (fmt or "").lower()
        if f == "png":
            return "image/png"
        if f in ("jpg", "jpeg"):
            return "image/jpeg"
        return "image/webp"

    def _validate_and_clean_b64(self, img_b64: str, image_num: int = 1) -> str:
        """Validate base64 image data and return clean version."""
        try:
            data = base64.b64decode(img_b64, validate=True)
            if len(data) > 25 * 1024 * 1024:
                raise ValueError(f"Image #{image_num} exceeds 25MB limit")
            return base64.b64encode(data).decode("utf-8")
        except Exception as e:
            logger.error(f"Error validating image #{image_num}: {e}")
            raise ValueError(f"Error validating image #{image_num}: {e}")

    async def _run_blocking(self, fn: Callable, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def _post_json_blocking(self, path: str, payload: Dict[str, Any], allow_binary: bool = False) -> Dict[str, Any]:
        base = self._get_base_url()
        key = self._get_api_key()
        if not key:
            logger.error("VENICE_API_KEY not set")
            raise RuntimeError("VENICE_API_KEY not set")
        url = f"{base}{path}"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        timeout = self.valves.TIMEOUT_SECONDS or 60
        with httpx.Client(timeout=timeout) as client:
            logger.info(f"POSTing to {url}")
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                logger.error(f"HTTP {resp.status_code} error from {url}")
                try:
                    err = resp.json()
                except Exception:
                    err = {"error": resp.text}
                raise RuntimeError(f"HTTP {resp.status_code} error: {err}")

            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    return resp.json()
                except Exception as e:
                    logger.error(f"Invalid JSON response: {e}")
                    raise RuntimeError(f"Invalid JSON response: {e}")
            if allow_binary:
                # Venice may return raw image bytes for edit/upscale
                img_b64 = base64.b64encode(resp.content).decode("utf-8")
                ct = content_type.split(";", 1)[0] if content_type else "image/png"
                return {"images": [img_b64], "__content_type": ct}
            logger.error(f"Expected JSON response but got content-type '{content_type}'")
            raise RuntimeError(f"Expected JSON response but got content-type '{content_type}'")

    def _upload_to_pictshare_blocking(self, b64_data: str) -> Optional[str]:
        pictshare_url = (self.valves.PICTSHARE_URL or "").strip()
        if not pictshare_url:
            logger.warning("PICTSHARE_URL is not configured.")
            return None

        url = f"{pictshare_url}/api/upload.php"
        logger.info(f"Uploading to pictshare: {url}")
        try:
            with httpx.Client(timeout=60) as client:
                try:
                    image_bytes = base64.b64decode(b64_data, validate=True)
                except (ValueError, TypeError):
                    logger.error("Invalid base64 data for pictshare upload.")
                    return None

                files = {"file": ("image.png", image_bytes, "image/png")}
                response = client.post(url, files=files)
                response.raise_for_status()
                res_json = response.json()
                if res_json.get("status") == "ok" and "url" in res_json:
                    logger.info(f"Pictshare upload successful: {res_json['url']}")
                    return res_json["url"]
                else:
                    err = res_json.get("error", "Unknown error")
                    logger.error(f"Pictshare API error: {err}")
                    return None
        except httpx.RequestError as e:
            logger.error(f"HTTP error uploading to pictshare: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during pictshare upload: {e}", exc_info=True)
            return None

    async def generate_image(
        self,
        prompt: str,
        n: int,
    ) -> AsyncGenerator[str, None]:
        await self.emit_status("⏳ Generating Image...")
        logger.info(f"Starting image generation for prompt: '{prompt}'")
        width, height = self._parse_size(self.valves.IMAGE_SIZE or "auto")
        fmt = (self.valves.OUTPUT_FORMAT or "webp").lower()
        mime = self._mime_for_format(fmt)
        payload_base: Dict[str, Any] = {
            "prompt": prompt or "",
            "model": self.valves.MODEL,
            "format": fmt,
            "return_binary": False,
            "safe_mode": bool(self.valves.SAFE_MODE),
            "cfg_scale": float(self.valves.CFG_SCALE),
            "lora_strength": int(self.valves.LORA_STRENGTH),
            "steps": int(self.valves.STEPS),
            "embed_exif_metadata": bool(self.valves.EMBED_EXIF_METADATA),
            "hide_watermark": bool(self.valves.HIDE_WATERMARK),
        }
        if width and height:
            payload_base["width"] = width
            payload_base["height"] = height
        if self.valves.NEGATIVE_PROMPT:
            payload_base["negative_prompt"] = self.valves.NEGATIVE_PROMPT
        if self.valves.SEED is not None:
            payload_base["seed"] = int(self.valves.SEED)
        if self.valves.STYLE_PRESET:
            payload_base["style_preset"] = self.valves.STYLE_PRESET

        try:
            variants = max(1, min(4, int(n)))
            payload_base["variants"] = variants
            logger.info(f"Requesting {variants} image(s)...")
            resp = await self._run_blocking(
                self._post_json_blocking, "/image/generate", payload_base, False
            )
            images = resp.get("images")
            if not isinstance(images, list) or not all(
                isinstance(x, str) for x in images
            ):
                logger.error(
                    "Invalid response from Venice generate: 'images' list missing"
                )
                raise RuntimeError(
                    "Invalid response from Venice generate: 'images' list missing"
                )

            logger.info(f"Received {len(images)} image(s) in response.")
            for img_b64 in images:
                b64 = "".join(img_b64.split())
                if self.valves.EMBED_IMAGES_AS_URL:
                    image_url = await self._run_blocking(
                        self._upload_to_pictshare_blocking, b64
                    )
                    if image_url:
                        yield f"![image]({image_url})[ System Generated Image: {image_url} ]"
                    else:
                        yield f"![image](data:{mime};base64,{b64})"
                else:
                    yield f"![image](data:{mime};base64,{b64})"
            logger.info(f"Successfully generated and emitted {len(images)} image(s).")
            await self.emit_status("🎉 Image generation successful", done=True)
        except Exception as e:
            logger.error(f"Error during image generation: {e}", exc_info=True)
            await self.emit_status("❌ Image generation failed", done=True)
            yield f"Error during image generation: {e}"

    async def edit_image(
        self,
        base64_images: List[str],
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        await self.emit_status("⏳ Editing Image...")
        logger.info(
            f"Starting image edit for {len(base64_images)} image(s) with prompt: '{prompt}'"
        )
        try:
            for i, img_b64 in enumerate(base64_images, start=1):
                logger.info(f"Processing image {i}/{len(base64_images)} for edit.")
                clean_b64 = self._validate_and_clean_b64(img_b64, i)
                
                payload = {
                    "prompt": prompt or "Edit the provided image",
                    "image": clean_b64,
                }

                resp = await self._run_blocking(
                    self._post_json_blocking, "/image/edit", payload, True
                )
                images = resp.get("images")
                if not isinstance(images, list) or not all(isinstance(x, str) for x in images):
                    logger.error("Invalid response from Venice edit: 'images' list missing")
                    raise RuntimeError("Invalid response from Venice edit: 'images' list missing")

                mime = resp.get("__content_type") or "image/png"
                logger.info(f"Received {len(images)} edited image(s) in response.")
                for out_b64 in images:
                    b64 = "".join(out_b64.split())
                    if self.valves.EMBED_IMAGES_AS_URL:
                        image_url = await self._run_blocking(
                            self._upload_to_pictshare_blocking, b64
                        )
                        if image_url:
                            yield f"![image_{i}]({image_url})[ System Generated Image: {image_url} ]"
                        else:
                            yield f"![image_{i}](data:{mime};base64,{b64})"
                    else:
                        yield f"![image_{i}](data:{mime};base64,{b64})"
            logger.info("Successfully edited and emitted all images.")
            await self.emit_status("🎉 Image edit successful", done=True)
        except Exception as e:
            logger.error(f"Error during image edit: {e}", exc_info=True)
            await self.emit_status("❌ Image edit failed", done=True)
            yield f"Error during image edit: {e}"

    async def upscale_images(
        self,
        base64_images: List[str],
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        await self.emit_status("⏳ Upscaling Image...")
        logger.info(f"Starting image upscale for {len(base64_images)} image(s).")
        try:
            scale = max(1, min(4, int(self.valves.UPSCALE_FACTOR or 2)))
            if scale not in (2, 4):
                scale = 2
            logger.info(f"Upscale factor set to {scale}x.")

            for i, img_b64 in enumerate(base64_images, start=1):
                logger.info(f"Processing image {i}/{len(base64_images)} for upscale.")
                clean_b64 = self._validate_and_clean_b64(img_b64, i)

                payload = {
                    "image": clean_b64,
                    "scale": scale,
                    "enhance": bool(self.valves.ENHANCE),
                    "enhanceCreativity": float(self.valves.ENHANCE_CREATIVITY),
                    "replication": float(self.valves.REPLICATION),
                }
                if self.valves.ENHANCE_PROMPT:
                    payload["enhancePrompt"] = self.valves.ENHANCE_PROMPT

                resp = await self._run_blocking(
                    self._post_json_blocking, "/image/upscale", payload, True
                )
                images = resp.get("images")
                if not isinstance(images, list) or not all(isinstance(x, str) for x in images):
                    logger.error("Invalid response from Venice upscale: 'images' list missing")
                    raise RuntimeError("Invalid response from Venice upscale: 'images' list missing")

                mime = resp.get("__content_type") or "image/png"
                logger.info(f"Received {len(images)} upscaled image(s) in response.")
                for out_b64 in images:
                    b64 = "".join(out_b64.split())
                    if self.valves.EMBED_IMAGES_AS_URL:
                        image_url = await self._run_blocking(
                            self._upload_to_pictshare_blocking, b64
                        )
                        if image_url:
                            yield f"![image_{i}]({image_url})[ System Generated Image: {image_url} ]"
                        else:
                            yield f"![image_{i}](data:{mime};base64,{b64})"
                    else:
                        yield f"![image_{i}](data:{mime};base64,{b64})"
            logger.info("Successfully upscaled and emitted all images.")
            await self.emit_status("🎉 Image upscale successful", done=True)
        except Exception as e:
            logger.error(f"Error during image upscale: {e}", exc_info=True)
            await self.emit_status("❌ Image upscale failed", done=True)
            yield f"Error during image upscale: {e}"

    async def inlet(self, body: dict, **kwargs) -> dict:
        # Pass inbound messages through without modification
        logger.info("Venice Image inlet pass-through.")
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> Dict[str, Any]:
        self.emitter = __event_emitter__
        logger.info("Venice Image outlet received a message.")
        
        assistant_message = next(
            (m for m in reversed(body["messages"]) if m.get("role") == "assistant"),
            None,
        )

        if not assistant_message or not isinstance(
            assistant_message.get("content"), str
        ):
            return body

        content = assistant_message.get("content", "")

        # Find all details blocks with invoke tags in the assistant's response
        matches = list(re.finditer(r"<details>\s*<summary>Image:.*?</summary>\s*<invoke>(.*?)</invoke>\s*</details>", content, re.DOTALL | re.IGNORECASE))
        if not matches:
            return body

        # Process each invoke block
        image_content = []
        for match in matches:
            try:
                params = {}
                # Extract XML tags: <tag>value</tag>
                invoke_content = match.group(1).strip()
                tag_pattern = re.compile(r"<([a-zA-Z_]+)>(.*?)</\1>", re.DOTALL)
                for tag, value in tag_pattern.findall(invoke_content):
                    params[tag.strip().lower()] = value.strip()
            except Exception as e:
                logger.error(f"Error parsing invoke block: {e}")
                continue

            logger.info(f"Successfully parsed detail block. Params: {params}")
            action = params.get("action", "generate")
            
            # Update valves from params
            for key, value in params.items():
                valve_key = key.strip().upper()
                if hasattr(self.valves, valve_key):
                    try:
                        current_value = getattr(self.valves, valve_key)
                        if isinstance(current_value, bool):
                            setattr(self.valves, valve_key, value.lower() in ['true', '1', 'yes'])
                        elif isinstance(current_value, int):
                            setattr(self.valves, valve_key, int(value))
                        elif isinstance(current_value, float):
                            setattr(self.valves, valve_key, float(value))
                        else:
                            setattr(self.valves, valve_key, value)
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Could not convert valve {valve_key} to correct type: {e}")

            if action == "generate":
                logger.info("Routing to generate_image.")
                variants = max(1, min(4, int(params.get("variants", self.valves.IMAGE_NUM))))
                async for out in self.generate_image(
                    prompt=params.get("prompt", ""), n=variants
                ):
                    assistant_message["content"] += f"\n{out}"
                    await self.emitter({"type": "message", "data": {"content": f"\n{out}"}})
            elif action in ["edit", "upscale"]:
                source_image_url = (
                    params.get("source_image_url")
                    or params.get("source_image")
                    or params.get("source_url")
                    or params.get("image_url")
                    or params.get("image")
                    or params.get("url")
                )
                if not source_image_url:
                    error_msg = f"Error: image URL is required for {action}. Use `source_image_url`, `image_url`, `image`, or `url`."
                    assistant_message["content"] += f"\n{error_msg}"
                    await self.emitter({"type": "message", "data": {"content": f"\n{error_msg}"}})
                else:
                    try:
                        async with httpx.AsyncClient(timeout=self.valves.TIMEOUT_SECONDS) as client:
                            response = await client.get(source_image_url)
                            response.raise_for_status()
                        
                        base64_image = base64.b64encode(response.content).decode('utf-8')
                        
                        if action == "edit":
                            logger.info("Routing to edit_image.")
                            async for out in self.edit_image(base64_images=[base64_image], prompt=params.get("prompt", "")):
                                assistant_message["content"] += f"\n{out}"
                                await self.emitter({"type": "message", "data": {"content": f"\n{out}"}})
                        else:
                            logger.info("Routing to upscale_images.")
                            async for out in self.upscale_images(base64_images=[base64_image], prompt=params.get("prompt", "")):
                                assistant_message["content"] += f"\n{out}"
                                await self.emitter({"type": "message", "data": {"content": f"\n{out}"}})

                    except httpx.HTTPStatusError as e:
                        error_msg = f"Error fetching source image: HTTP {e.response.status_code}"
                        assistant_message["content"] += f"\n{error_msg}"
                        await self.emitter({"type": "message", "data": {"content": f"\n{error_msg}"}})
                    except Exception as e:
                        error_msg = f"Error processing source image: {str(e)}"
                        assistant_message["content"] += f"\n{error_msg}"
                        await self.emitter({"type": "message", "data": {"content": f"\n{error_msg}"}})
            else:
                error_msg = f"Unknown action: {action}"
                assistant_message["content"] += f"\n{error_msg}"
                await self.emitter({"type": "message", "data": {"content": f"\n{error_msg}"}})

        logger.info("Processing finished!")
        return body