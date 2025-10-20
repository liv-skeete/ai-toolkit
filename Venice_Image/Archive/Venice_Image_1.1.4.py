"""
title: Venice Image
description: Image generation, editing, and upscaling via Venice.ai API
author: Cody
version: 1.1.4
date: 2025-09-25
changes:
   - v1.1.4 (2025-09-25):
       - Added `EMBED_IMAGES_AS_URL` valve to upload images to pictshare and embed URLs instead of base64.
       - Added `PICTSHARE_URL` valve for configuring the pictshare service endpoint.
   - v1.1.3 (2025-08-30):
       - Constrained OUTPUT_FORMAT to enum ['png', 'jpeg', 'webp'] to enable dropdown in UI.
   - v1.1.2 (2025-08-30):
       - Moved UPSCALE_FACTOR to admin Valves; user setting no longer overrides scale.
   - v1.1.1 (2025-08-30):
       - Fix: Respect per-user UserValves values passed via __user__.valves with user-first fallback resolution.
   - v1.1.0 (2025-08-30):
       - Refactored valves into UserValves (user-facing) and Valves (admin/advanced).
   - v1.0.0 (2025-08-19):
       - Initial release with Venice API integration for generate, edit, upscale, and styles.
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
logger = logging.getLogger("image_v1")
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler once
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


class Pipe:
    class Valves(BaseModel):
        VENICE_API_KEY: str = Field(
            default="", description="Venice API key (single key)"
        )
        VENICE_BASE_URL: Optional[str] = Field(
            default="https://api.venice.ai/api/v1",
            description="Venice API base URL",
        )
        OUTPUT_FORMAT: Literal["webp", "jpeg", "png"] = Field(
            default="webp", description="webp | jpeg | png"
        )
        SAFE_MODE: bool = Field(
            default=False, description="Safe-mode for image generation"
        )
        EMBED_EXIF_METADATA: bool = Field(
            default=False, description="Embed EXIF metadata"
        )
        HIDE_WATERMARK: bool = Field(
            default=True, description="Hide watermark if supported"
        )
        UPSCALE_FACTOR: Literal[2, 4] = Field(
            default=2, description="Default upscale factor (2 or 4)"
        )
        UPSCALE_KEYWORDS: str = Field(
            default="upscale",
            description="Keyword that triggers upscale when images are present",
        )
        # Networking
        TIMEOUT_SECONDS: int = Field(
            default=180, description="API equest timeout in seconds"
        )
        PICTSHARE_URL: Optional[str] = Field(
            default="http://pictshare:4000",
            description="URL for pictshare service (internal or public)",
        )
        EMBED_IMAGES_AS_URL: bool = Field(
            default=False,
            description="Upload images to pictshare and embed as URLs",
        )

    class UserValves(BaseModel):
        # Generation defaults
        MODEL: str = Field(
            default="hidream", description="Required: 'hidream' 'qwen-image'  'lustify-sdxl' 'venice-sd35' 'wai-Illustrious'"
        )
        STYLE_PRESET: Optional[str] = Field(
            default=None, description="Optional: 'Abstract' 'Anime' 'Dreamscape' 'Impressionist' 'Minimalist' 'Monochrome' 'Photographic' 'Watercolor'"
        )
        NEGATIVE_PROMPT: Optional[str] = Field(
            default=None, description="Optional negative prompt"
        )
        IMAGE_NUM: int = Field(default=1, description="Number of images to generate")
        IMAGE_SIZE: str = Field(
            default="auto", description="WxH (e.g., 1024x1024) or 'auto'"
        )
        CFG_SCALE: float = Field(
            default=7.5,
            description="Prompt adherence 0-20",
        )
        LORA_STRENGTH: int = Field(default=50, description="LoRA strength (0-100)")
        STEPS: int = Field(default=30, description="Steps for generation")
        SEED: Optional[int] = Field(
            default=None, description="Optional deterministic seed"
        )
        # Upscale defaults
        ENHANCE: bool = Field(default=False, description="Enhance upscaled image")
        ENHANCE_PROMPT: Optional[str] = Field(
            default=None, description="Optional enhance prompt for upscale"
        )
        ENHANCE_CREATIVITY: float = Field(
            default=0.5, description="Enhance creativity (0..1)"
        )

    def __init__(self):
        self.type = "manifold"
        self.name = "Venice: "
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.emitter: Optional[Callable[[dict], Awaitable[None]]] = None
        self._styles_cache: Optional[List[str]] = None

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

    async def pipes(self) -> List[dict]:
        return [{"id": "sol-image", "name": "Sol Image"}]

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

    def _get_user_valve(self, __user__: Optional[dict], key: str, default):
        """
        Resolve an effective user setting value.
        - Prefer __user__['valves'].<key> if present (supports attr and dict access)
        - Fall back to provided default.
        """
        try:
            if (
                __user__
                and isinstance(__user__, dict)
                and "valves" in __user__
                and __user__["valves"] is not None
            ):
                v = __user__["valves"]
                # Attribute access (Pydantic model or dict-like)
                if hasattr(v, key):
                    val = getattr(v, key)
                elif isinstance(v, dict) and key in v:
                    val = v[key]
                else:
                    val = None
                if val is not None:
                    return val
        except Exception as e:
            logger.debug(f"_get_user_valve: failed to read '{key}': {e}")
        return default

    def _upscale_intent_and_factor(self, text: str) -> Tuple[bool, Optional[int]]:
        if not text:
            return False, None
        kws = [
            k.strip().lower()
            for k in (self.valves.UPSCALE_KEYWORDS or "").split(",")
            if k.strip()
        ]
        t = text.lower()
        intent = any(k in t for k in kws)
        factor = 4 if "4x" in t else (2 if "2x" in t else None)
        return intent, factor

    def convert_message_to_prompt(self, messages: List[dict]) -> tuple[str, List[dict]]:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")

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
                prompt = " ".join(text_parts).strip() or ""
                return prompt, image_data_list

            if isinstance(content, str):
                pattern = r"!\[[^\]]*\]\(data:([^;]+);base64,([^)]+)\)"
                matches = re.findall(pattern, content)
                image_data_list = [{"mimeType": m, "data": d} for m, d in matches]
                clean = re.sub(pattern, "", content).strip()
                return clean, image_data_list

        return "", []

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

    def _get_json_blocking(self, path: str) -> Dict[str, Any]:
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
            logger.info(f"GETting from {url}")
            resp = client.get(url, headers=headers)
            if resp.status_code >= 400:
                logger.error(f"HTTP {resp.status_code} error from {url}")
                try:
                    err = resp.json()
                except Exception:
                    err = {"error": resp.text}
                raise RuntimeError(f"HTTP {resp.status_code} error: {err}")
            try:
                return resp.json()
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                raise RuntimeError(f"Invalid JSON response: {e}")

    def _upload_to_pictshare_blocking(self, b64_data: str) -> Optional[str]:
        pictshare_url = (self.valves.PICTSHARE_URL or "").strip()
        if not pictshare_url:
            logger.warning("PICTSHARE_URL is not configured.")
            return None

        url = f"{pictshare_url}/api/upload.php"
        logger.info(f"Uploading to pictshare: {url}")
        try:
            with httpx.Client(timeout=60) as client:
                # The PHP script expects a multipart file upload with the name 'file'
                try:
                    image_bytes = base64.b64decode(b64_data, validate=True)
                except (ValueError, TypeError):
                    logger.error("Invalid base64 data for pictshare upload.")
                    return None

                files = {"file": ("image.png", image_bytes, "image/png")}
                response = client.post(url, files=files)
                response.raise_for_status()  # Raises on 4xx/5xx
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

    async def fetch_styles(self) -> List[str]:
        if self._styles_cache is not None:
            logger.info("Returning cached styles.")
            return self._styles_cache
        try:
            logger.info("Fetching styles from Venice API.")
            data = await self._run_blocking(self._get_json_blocking, "/image/styles")
            styles = data.get("data", [])
            if isinstance(styles, list):
                self._styles_cache = [s for s in styles if isinstance(s, str)]
                logger.info(
                    f"Successfully fetched and cached {len(self._styles_cache)} styles."
                )
            else:
                self._styles_cache = []
                logger.warning("Styles data was not a list.")
        except Exception as e:
            logger.error(f"Failed to fetch styles: {e}")
            self._styles_cache = []
        return self._styles_cache

    async def generate_image(
        self,
        prompt: str,
        n: int,
        __user__: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        await self.emit_status("⏳ Generating Image...")
        logger.info(f"Starting image generation for prompt: '{prompt}'")
        # Build common payload
        width, height = self._parse_size(
            (self._get_user_valve(__user__, "IMAGE_SIZE", self.user_valves.IMAGE_SIZE))
            or "auto"
        )
        fmt = (self.valves.OUTPUT_FORMAT or "webp").lower()
        mime = self._mime_for_format(fmt)
        payload_base: Dict[str, Any] = {
            "prompt": prompt or "",
            "model": self._get_user_valve(__user__, "MODEL", self.user_valves.MODEL),
            "format": fmt,
            "return_binary": False,
            "safe_mode": bool(self.valves.SAFE_MODE),
            "cfg_scale": float(
                self._get_user_valve(__user__, "CFG_SCALE", self.user_valves.CFG_SCALE)
            ),
            "lora_strength": int(
                self._get_user_valve(
                    __user__, "LORA_STRENGTH", self.user_valves.LORA_STRENGTH
                )
            ),
            "steps": int(
                self._get_user_valve(__user__, "STEPS", self.user_valves.STEPS)
            ),
            "embed_exif_metadata": bool(self.valves.EMBED_EXIF_METADATA),
            "hide_watermark": bool(self.valves.HIDE_WATERMARK),
        }
        if width and height:
            payload_base["width"] = width
            payload_base["height"] = height
        neg = self._get_user_valve(
            __user__, "NEGATIVE_PROMPT", self.user_valves.NEGATIVE_PROMPT
        )
        if neg:
            payload_base["negative_prompt"] = neg
        seed_val = self._get_user_valve(__user__, "SEED", self.user_valves.SEED)
        if seed_val is not None:
            payload_base["seed"] = int(seed_val)
        style = self._get_user_valve(
            __user__, "STYLE_PRESET", self.user_valves.STYLE_PRESET
        )
        if style:
            payload_base["style_preset"] = style

        try:
            count = max(1, min(10, int(n)))
            emitted = 0
            for i in range(count):
                logger.info(f"Requesting image {i + 1}/{count}...")
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
                    # Ensure base64 cleanliness (strip whitespace/newlines)
                    b64 = "".join(img_b64.split())
                    if self.valves.EMBED_IMAGES_AS_URL:
                        image_url = await self._run_blocking(
                            self._upload_to_pictshare_blocking, b64
                        )
                        if image_url:
                            yield f"![image]({image_url})"
                        else:
                            # Fallback to base64 if upload fails
                            yield f"![image]({f'data:{mime};base64,'}{b64})"
                    else:
                        yield f"![image]({f'data:{mime};base64,'}{b64})"
                    emitted += 1
            logger.info(f"Successfully generated and emitted {emitted} image(s).")
            await self.emit_status("🎉 Image generation successful", done=True)
        except Exception as e:
            logger.error(f"Error during image generation: {e}", exc_info=True)
            await self.emit_status("❌ Image generation failed", done=True)
            yield f"Error during image generation: {e}"

    async def edit_image(
        self,
        base64_images: List[dict],
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        await self.emit_status("⏳ Editing Image...")
        logger.info(
            f"Starting image edit for {len(base64_images)} image(s) with prompt: '{prompt}'"
        )
        try:
            for i, img_dict in enumerate(base64_images, start=1):
                logger.info(f"Processing image {i}/{len(base64_images)} for edit.")
                # Validate input image
                try:
                    data = base64.b64decode(img_dict.get("data", ""), validate=True)
                    if len(data) > 25 * 1024 * 1024:
                        raise ValueError("Image exceeds 25MB limit")
                    # We only need to send base64, not data URI
                    img_b64 = base64.b64encode(data).decode("utf-8")
                except Exception as e:
                    logger.error(f"Error decoding image #{i}: {e}")
                    raise ValueError(f"Error decoding image #{i}: {e}")

                payload = {
                    "prompt": prompt or "Edit the provided image",
                    "image": img_b64,
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
                            yield f"![image_{i}]({image_url})"
                        else:
                            yield f"![image_{i}]({f'data:{mime};base64,'}{b64})"
                    else:
                        yield f"![image_{i}]({f'data:{mime};base64,'}{b64})"
            logger.info("Successfully edited and emitted all images.")
            await self.emit_status("🎉 Image edit successful", done=True)
        except Exception as e:
            logger.error(f"Error during image edit: {e}", exc_info=True)
            await self.emit_status("❌ Image edit failed", done=True)
            yield f"Error during image edit: {e}"

    async def upscale_images(
        self,
        base64_images: List[dict],
        prompt: str,
        __user__: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        await self.emit_status("⏳ Upscaling Image...")
        logger.info(f"Starting image upscale for {len(base64_images)} image(s).")
        try:
            # Determine factor from prompt if available
            intent, factor_from_prompt = self._upscale_intent_and_factor(prompt or "")
            scale = factor_from_prompt or max(
                1, min(4, int(self.valves.UPSCALE_FACTOR or 2))
            )
            if scale not in (2, 4):
                scale = 2
            logger.info(f"Upscale factor set to {scale}x.")

            for i, img_dict in enumerate(base64_images, start=1):
                logger.info(f"Processing image {i}/{len(base64_images)} for upscale.")
                try:
                    data = base64.b64decode(img_dict.get("data", ""), validate=True)
                    if len(data) > 25 * 1024 * 1024:
                        raise ValueError("Image exceeds 25MB limit")
                    img_b64 = base64.b64encode(data).decode("utf-8")
                except Exception as e:
                    logger.error(f"Error decoding image #{i}: {e}")
                    raise ValueError(f"Error decoding image #{i}: {e}")

                payload = {
                    "image": img_b64,
                    "scale": scale,
                    "enhance": bool(
                        self._get_user_valve(
                            __user__, "ENHANCE", self.user_valves.ENHANCE
                        )
                    ),
                    "enhanceCreativity": float(
                        self._get_user_valve(
                            __user__,
                            "ENHANCE_CREATIVITY",
                            self.user_valves.ENHANCE_CREATIVITY,
                        )
                    ),
                }
                enhance_prompt = self._get_user_valve(
                    __user__, "ENHANCE_PROMPT", self.user_valves.ENHANCE_PROMPT
                )
                if enhance_prompt:
                    payload["enhancePrompt"] = enhance_prompt

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
                            yield f"![image_{i}]({image_url})"
                        else:
                            yield f"![image_{i}]({f'data:{mime};base64,'}{b64})"
                    else:
                        yield f"![image_{i}]({f'data:{mime};base64,'}{b64})"
            logger.info("Successfully upscaled and emitted all images.")
            await self.emit_status("🎉 Image upscale successful", done=True)
        except Exception as e:
            logger.error(f"Error during image upscale: {e}", exc_info=True)
            await self.emit_status("❌ Image upscale failed", done=True)
            yield f"Error during image upscale: {e}"

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __user__: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        self.emitter = __event_emitter__
        logger.info("Processing started...")
        msgs = body.get("messages", [])
        prompt, imgs = self.convert_message_to_prompt(msgs)
        logger.info(f"Found {len(imgs)} image(s) and prompt: '{prompt}'")

        # Routing
        if imgs:
            upscale_intent, _ = self._upscale_intent_and_factor(prompt)
            if upscale_intent:
                logger.info("Upscale intent detected, routing to upscale_images.")
                async for out in self.upscale_images(
                    base64_images=imgs, prompt=prompt, __user__=__user__
                ):
                    yield out
            else:
                logger.info(
                    "Image(s) present without upscale intent, routing to edit_image."
                )
                async for out in self.edit_image(base64_images=imgs, prompt=prompt):
                    yield out
        else:
            logger.info("No images found, routing to generate_image.")
            n = max(
                1,
                min(
                    10,
                    int(
                        self._get_user_valve(
                            __user__, "IMAGE_NUM", self.user_valves.IMAGE_NUM
                        )
                    ),
                ),
            )
            async for out in self.generate_image(prompt=prompt, n=n, __user__=__user__):
                yield out
        logger.info("Processing finished!")
