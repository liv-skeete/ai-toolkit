"""
title: Venice Image
description: Image generation, editing, and upscaling via Venice.ai API
author: Cody
version: 1.1.5
date: 2025-10-09
changelog: Venice_Image/_changelog.md
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
    """
    Venice Image Pipe for Open WebUI.
    
    This pipe provides image generation, editing, and upscaling capabilities
    through the Venice.ai API. It processes user messages that contain
    image prompts and base64-encoded images.
    
    The pipe supports three main operations:
    1. Generate - Create new images from text prompts
    2. Edit - Modify existing images based on textual instructions
    3. Upscale - Increase image resolution with optional enhancement
    
    All operations support configuration through the Valves system, allowing
    customization of API endpoints, default parameters, and output options.
    """
    
    class Valves(BaseModel):
        """
        Configuration valves for the Venice Image Pipe.
        
        These settings control the behavior of the pipe, including API endpoints,
        default parameters for image operations, and output options.
        
        Configuration Categories:
        - System settings: API endpoints, keys, and timeouts
        - Image defaults: Output format and quality settings
        - Upscale defaults: Parameters for image upscaling
        """
        VENICE_API_KEY: str = Field(
            default="", description="Venice API key (single key)"
        )
        VENICE_BASE_URL: Optional[str] = Field(
            default="https://api.venice.ai/api/v1",
            description="Venice API base URL",
        )
         # Networking
        EMBED_IMAGES_AS_URL: bool = Field(
            default=False,
            description="Upload images to pictshare and embed as URLs",
        )        
        PICTSHARE_URL: Optional[str] = Field(
            default="http://pictshare:4000",
            description="URL for pictshare service (internal or public)",
        ) 
        TIMEOUT_SECONDS: int = Field(
            default=180, description="API request timeout in seconds"
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


    class UserValves(BaseModel):
        """
        User-facing configuration valves for the Venice Image Pipe.
        
        These settings allow users to customize their image generation experience.
        
        Configuration Categories:
        - Generation defaults: Parameters for image creation
        - Upscale defaults: Parameters for image upscaling
        """
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

    def _validate_and_clean_b64(self, img_b64: str, image_num: int = 1) -> str:
        """Validate base64 image data and return clean version.
        
        Args:
            img_b64: Base64 encoded image data
            image_num: Image number for error reporting
            
        Returns:
            Clean base64 encoded image data
            
        Raises:
            ValueError: If image validation fails
        """
        try:
            # Validate and decode base64 data
            data = base64.b64decode(img_b64, validate=True)
            
            # Check file size limit (25MB)
            if len(data) > 25 * 1024 * 1024:
                error_msg = f"Image #{image_num} exceeds 25MB limit ({len(data)} bytes)"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Re-encode to ensure clean data
            return base64.b64encode(data).decode("utf-8")
        except base64.binascii.Error as e:
            error_msg = f"Invalid base64 data for image #{image_num}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Error validating image #{image_num}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e

    async def _process_image_output(self, b64: str, mime: str, image_num: int = 1) -> str:
        """Process image output by uploading to pictshare or embedding as base64.
        
        Args:
            b64: Base64 encoded image data
            mime: MIME type of the image
            image_num: Image number for identification
            
        Returns:
            Image markdown string
        """
        try:
            # Clean base64 data
            clean_b64 = "".join(b64.split())
            
            # Upload to pictshare if configured
            if self.valves.EMBED_IMAGES_AS_URL:
                image_url = await self._run_blocking(
                    self._upload_to_pictshare_blocking, clean_b64
                )
                if image_url:
                    return f"![image_{image_num}]({image_url})"
                else:
                    # Fallback to base64 if upload fails
                    return f"![image_{image_num}](data:{mime};base64,{clean_b64})"
            else:
                return f"![image_{image_num}](data:{mime};base64,{clean_b64})"
        except Exception as e:
            error_msg = f"Error processing image output #{image_num}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _post_json_blocking(self, path: str, payload: Dict[str, Any], allow_binary: bool = False) -> Dict[str, Any]:
        """Send a POST request with JSON payload to the Venice API.
        
        Args:
            path: API endpoint path
            payload: JSON payload to send
            allow_binary: Whether to allow binary response
            
        Returns:
            Response data as dictionary
            
        Raises:
            RuntimeError: If API request fails
        """
        base = self._get_base_url()
        key = self._get_api_key()
        
        # Validate API key
        if not key:
            error_msg = "VENICE_API_KEY not set - image operations will fail"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        url = f"{base}{path}"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        
        # Get timeout with fallback
        timeout = self.valves.TIMEOUT_SECONDS or 60
        if timeout < 1:
            timeout = 60  # Minimum 60 seconds
            
        try:
            with httpx.Client(timeout=timeout) as client:
                logger.info(f"POSTing to {url} with timeout {timeout}s")
                resp = client.post(url, headers=headers, json=payload)
                
                # Handle HTTP errors
                if resp.status_code >= 400:
                    error_msg = f"HTTP {resp.status_code} error from {url}"
                    logger.error(error_msg)
                    
                    # Try to extract error details from response
                    try:
                        err = resp.json()
                        error_details = err.get("error", err) if isinstance(err, dict) else str(err)
                    except Exception:
                        error_details = resp.text
                        
                    raise RuntimeError(f"{error_msg}: {error_details}")

                # Process response based on content type
                content_type = resp.headers.get("content-type", "").lower()
                
                if "application/json" in content_type:
                    try:
                        return resp.json()
                    except Exception as e:
                        error_msg = f"Invalid JSON response from {url}: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        raise RuntimeError(error_msg) from e
                        
                if allow_binary:
                    # Venice may return raw image bytes for edit/upscale
                    try:
                        img_b64 = base64.b64encode(resp.content).decode("utf-8")
                        ct = content_type.split(";", 1)[0] if content_type else "image/png"
                        return {"images": [img_b64], "__content_type": ct}
                    except Exception as e:
                        error_msg = f"Error processing binary response from {url}: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        raise RuntimeError(error_msg) from e
                        
                error_msg = f"Expected JSON response from {url} but got content-type '{content_type}'"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except httpx.TimeoutException as e:
            error_msg = f"Request to {url} timed out after {timeout}s"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
        except httpx.RequestError as e:
            error_msg = f"Network error during request to {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error during request to {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _get_json_blocking(self, path: str) -> Dict[str, Any]:
        """Send a GET request to the Venice API.
        
        Args:
            path: API endpoint path
            
        Returns:
            Response data as dictionary
            
        Raises:
            RuntimeError: If API request fails
        """
        base = self._get_base_url()
        key = self._get_api_key()
        
        # Validate API key
        if not key:
            error_msg = "VENICE_API_KEY not set - image operations will fail"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        url = f"{base}{path}"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        
        # Get timeout with fallback
        timeout = self.valves.TIMEOUT_SECONDS or 60
        if timeout < 1:
            timeout = 60  # Minimum 60 seconds
            
        try:
            with httpx.Client(timeout=timeout) as client:
                logger.info(f"GETting from {url} with timeout {timeout}s")
                resp = client.get(url, headers=headers)
                
                # Handle HTTP errors
                if resp.status_code >= 400:
                    error_msg = f"HTTP {resp.status_code} error from {url}"
                    logger.error(error_msg)
                    
                    # Try to extract error details from response
                    try:
                        err = resp.json()
                        error_details = err.get("error", err) if isinstance(err, dict) else str(err)
                    except Exception:
                        error_details = resp.text
                        
                    raise RuntimeError(f"{error_msg}: {error_details}")

                # Process JSON response
                try:
                    return resp.json()
                except Exception as e:
                    error_msg = f"Invalid JSON response from {url}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise RuntimeError(error_msg) from e
                    
        except httpx.TimeoutException as e:
            error_msg = f"Request to {url} timed out after {timeout}s"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
        except httpx.RequestError as e:
            error_msg = f"Network error during request to {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error during request to {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _upload_to_pictshare_blocking(self, b64_data: str) -> Optional[str]:
        """Upload base64 image data to pictshare service.
        
        Args:
            b64_data: Base64 encoded image data
            
        Returns:
            Image URL if successful, None otherwise
        """
        pictshare_url = (self.valves.PICTSHARE_URL or "").strip()
        
        # Validate pictshare URL configuration
        if not pictshare_url:
            warning_msg = "PICTSHARE_URL is not configured - images will be embedded as base64 data"
            logger.warning(warning_msg)
            return None

        url = f"{pictshare_url}/api/upload.php"
        logger.info(f"Uploading to pictshare: {url}")
        
        try:
            with httpx.Client(timeout=60) as client:
                try:
                    # Validate and decode base64 data
                    image_bytes = base64.b64decode(b64_data, validate=True)
                except (base64.binascii.Error, ValueError, TypeError) as e:
                    error_msg = f"Invalid base64 data for pictshare upload: {str(e)}"
                    logger.error(error_msg)
                    return None

                # Prepare file for upload
                files = {"file": ("image.png", image_bytes, "image/png")}
                
                # Upload file
                response = client.post(url, files=files)
                response.raise_for_status()
                
                # Parse response
                res_json = response.json()
                
                # Check if upload was successful
                if res_json.get("status") == "ok" and "url" in res_json:
                    image_url = res_json["url"]
                    logger.info(f"Pictshare upload successful: {image_url}")
                    return image_url
                else:
                    err = res_json.get("error", "Unknown error")
                    error_msg = f"Pictshare API error: {err}"
                    logger.error(error_msg)
                    return None
                    
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code} error uploading to pictshare: {str(e)}"
            logger.error(error_msg)
            return None
            
        except httpx.RequestError as e:
            error_msg = f"Network error uploading to pictshare: {str(e)}"
            logger.error(error_msg)
            return None
            
        except Exception as e:
            error_msg = f"Unexpected error during pictshare upload: {str(e)}"
            logger.error(error_msg, exc_info=True)
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
        """Generate images using the Venice API.
        
        Args:
            prompt: Text prompt for image generation
            n: Number of images to generate
            __user__: User context for per-user valve values
            
        Yields:
            Image markdown strings
        """
        await self.emit_status("⏳ Generating Image...")
        logger.info(f"Starting image generation for prompt: '{prompt}'")
        
        try:
            # Validate and sanitize inputs
            if not prompt or not prompt.strip():
                error_msg = "Prompt is required for image generation"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Validate prompt length (1-1500 characters)
            prompt = prompt.strip()
            if len(prompt) > 1500:
                prompt = prompt[:1500]
                logger.warning(f"Prompt truncated to 1500 characters: {prompt[:50]}...")
            elif len(prompt) < 1:
                error_msg = "Prompt must be between 1 and 1500 characters"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Parse image dimensions
            width, height = self._parse_size(
                (self._get_user_valve(__user__, "IMAGE_SIZE", self.user_valves.IMAGE_SIZE))
                or "auto"
            )
            
            # Validate width and height if specified (0 < x <= 1280)
            if width is not None:
                if width <= 0 or width > 1280:
                    logger.warning(f"Width {width} is out of range (0 < x <= 1280), using default")
                    width = None
            if height is not None:
                if height <= 0 or height > 1280:
                    logger.warning(f"Height {height} is out of range (0 < x <= 1280), using default")
                    height = None
            
            # Validate output format
            fmt = (self.valves.OUTPUT_FORMAT or "webp").lower()
            if fmt not in ["webp", "jpeg", "png"]:
                logger.warning(f"Invalid format '{fmt}', using default 'webp'")
                fmt = "webp"
            mime = self._mime_for_format(fmt)
            
            # Validate CFG scale (0 < x <= 20)
            cfg_scale = float(
                self._get_user_valve(__user__, "CFG_SCALE", self.user_valves.CFG_SCALE)
            )
            if cfg_scale <= 0 or cfg_scale > 20:
                logger.warning(f"CFG scale {cfg_scale} is out of range (0 < x <= 20), using default 7.5")
                cfg_scale = 7.5
                
            # Validate LoRA strength (0 <= x <= 100)
            lora_strength = int(
                self._get_user_valve(
                    __user__, "LORA_STRENGTH", self.user_valves.LORA_STRENGTH
                )
            )
            if lora_strength < 0 or lora_strength > 100:
                logger.warning(f"LoRA strength {lora_strength} is out of range (0 <= x <= 100), using default 50")
                lora_strength = 50
                
            # Validate steps (0 < x <= 50)
            steps = int(
                self._get_user_valve(__user__, "STEPS", self.user_valves.STEPS)
            )
            if steps <= 0 or steps > 50:
                logger.warning(f"Steps {steps} is out of range (0 < x <= 50), using default 30")
                steps = 30
                
            # Validate seed range (-999999999 <= x <= 999999999)
            seed = None
            seed_val = self._get_user_valve(__user__, "SEED", self.user_valves.SEED)
            if seed_val is not None:
                seed_val = int(seed_val)
                if seed_val < -999999999 or seed_val > 999999999:
                    logger.warning(f"Seed {seed_val} is out of range (-999999999 <= x <= 999999999), using random seed")
                else:
                    seed = seed_val
            
            # Get negative prompt and validate length (max 1500 characters)
            negative_prompt = self._get_user_valve(
                __user__, "NEGATIVE_PROMPT", self.user_valves.NEGATIVE_PROMPT
            )
            if negative_prompt and len(negative_prompt) > 1500:
                negative_prompt = negative_prompt[:1500]
                logger.warning(f"Negative prompt truncated to 1500 characters: {negative_prompt[:50]}...")
            
            # Build base payload
            payload_base: Dict[str, Any] = {
                "prompt": prompt,
                "model": self._get_user_valve(__user__, "MODEL", self.user_valves.MODEL),
                "format": fmt,
                "return_binary": False,
                "safe_mode": bool(self.valves.SAFE_MODE),
                "cfg_scale": cfg_scale,
                "lora_strength": lora_strength,
                "steps": steps,
                "embed_exif_metadata": bool(self.valves.EMBED_EXIF_METADATA),
                "hide_watermark": bool(self.valves.HIDE_WATERMARK),
            }
            
            # Add optional parameters
            if width and height:
                payload_base["width"] = width
                payload_base["height"] = height
            if negative_prompt:
                payload_base["negative_prompt"] = negative_prompt
            if seed is not None:
                payload_base["seed"] = seed
            style = self._get_user_valve(
                __user__, "STYLE_PRESET", self.user_valves.STYLE_PRESET
            )
            if style:
                payload_base["style_preset"] = style

            # Validate and set number of variants
            try:
                variants = max(1, min(10, int(n)))
                payload_base["variants"] = variants
            except (ValueError, TypeError) as e:
                error_msg = f"Invalid number of images requested: {n}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e

            logger.info(f"Requesting {variants} image(s)...")
            
            # Make API request
            resp = await self._run_blocking(
                self._post_json_blocking, "/image/generate", payload_base, False
            )
            
            # Validate response
            images = resp.get("images")
            if not isinstance(images, list):
                error_msg = "Invalid response from Venice generate: 'images' is not a list"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            if not all(isinstance(x, str) for x in images):
                error_msg = "Invalid response from Venice generate: not all items are strings"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info(f"Received {len(images)} image(s) in response.")
            
            # Process each image
            for img_b64 in images:
                try:
                    yield await self._process_image_output(img_b64, mime)
                except Exception as e:
                    error_msg = f"Error processing generated image: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    yield error_msg
                    
            logger.info(f"Successfully generated and emitted {len(images)} image(s).")
            await self.emit_status("🎉 Image generation successful", done=True)
            
        except ValueError as e:
            error_msg = f"Invalid input for image generation: {str(e)}"
            logger.error(error_msg)
            await self.emit_status("❌ Image generation failed: Invalid input", done=True)
            yield error_msg
            
        except RuntimeError as e:
            error_msg = f"API error during image generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self.emit_status("❌ Image generation failed: API error", done=True)
            yield error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during image generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self.emit_status("❌ Image generation failed", done=True)
            yield error_msg

    async def edit_image(
        self,
        base64_images: List[dict],
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        """Edit images using the Venice API.
        
        Args:
            base64_images: List of base64 encoded images to edit
            prompt: Text prompt for image editing
            
        Yields:
            Edited image markdown strings
        """
        await self.emit_status("⏳ Editing Image...")
        logger.info(
            f"Starting image edit for {len(base64_images)} image(s) with prompt: '{prompt}'"
        )
        
        try:
            # Validate inputs
            if not base64_images:
                error_msg = "No images provided for editing"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if not isinstance(base64_images, list):
                error_msg = "Images must be provided as a list"
                logger.error(error_msg)
                raise TypeError(error_msg)

            for i, img_dict in enumerate(base64_images, start=1):
                logger.info(f"Processing image {i}/{len(base64_images)} for edit.")
                
                try:
                    # Validate prompt for editing (max 1500 characters)
                    edit_prompt = prompt or "Edit the provided image"
                    if len(edit_prompt) > 1500:
                        edit_prompt = edit_prompt[:1500]
                        logger.warning(f"Edit prompt for image {i} truncated to 1500 characters: {edit_prompt[:50]}...")
                    
                    # Validate and clean base64 image data
                    clean_b64 = self._validate_and_clean_b64(img_dict.get("data", ""), i)
                    
                    # Build payload
                    payload = {
                        "prompt": edit_prompt,
                        "image": clean_b64,
                    }

                    # Make API request
                    resp = await self._run_blocking(
                        self._post_json_blocking, "/image/edit", payload, True
                    )
                    
                    # Validate response
                    images = resp.get("images")
                    if not isinstance(images, list):
                        error_msg = f"Invalid response from Venice edit for image {i}: 'images' is not a list"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                        
                    if not all(isinstance(x, str) for x in images):
                        error_msg = f"Invalid response from Venice edit for image {i}: not all items are strings"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

                    # Get MIME type
                    mime = resp.get("__content_type") or "image/png"
                    logger.info(f"Received {len(images)} edited image(s) in response.")
                    
                    # Process each edited image
                    for out_b64 in images:
                        try:
                            yield await self._process_image_output(out_b64, mime, i)
                        except Exception as e:
                            error_msg = f"Error processing edited image {i}: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            yield error_msg
                            
                except ValueError as e:
                    error_msg = f"Invalid image data for image {i}: {str(e)}"
                    logger.error(error_msg)
                    yield error_msg
                    continue  # Continue with other images
                    
                except RuntimeError as e:
                    error_msg = f"API error editing image {i}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    yield error_msg
                    continue  # Continue with other images
                    
            logger.info("Successfully edited and emitted all images.")
            await self.emit_status("🎉 Image edit successful", done=True)
            
        except ValueError as e:
            error_msg = f"Invalid input for image editing: {str(e)}"
            logger.error(error_msg)
            await self.emit_status("❌ Image edit failed: Invalid input", done=True)
            yield error_msg
            
        except TypeError as e:
            error_msg = f"Type error in image editing: {str(e)}"
            logger.error(error_msg)
            await self.emit_status("❌ Image edit failed: Type error", done=True)
            yield error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during image edit: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self.emit_status("❌ Image edit failed", done=True)
            yield error_msg

    async def upscale_images(
        self,
        base64_images: List[dict],
        prompt: str,
        __user__: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Upscale images using the Venice API.
        
        Args:
            base64_images: List of base64 encoded images to upscale
            prompt: Text prompt for image upscaling (optional)
            __user__: User context for per-user valve values
            
        Yields:
            Upscaled image markdown strings
        """
        await self.emit_status("⏳ Upscaling Image...")
        logger.info(f"Starting image upscale for {len(base64_images)} image(s).")
        
        try:
            # Validate inputs
            if not base64_images:
                error_msg = "No images provided for upscaling"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if not isinstance(base64_images, list):
                error_msg = "Images must be provided as a list"
                logger.error(error_msg)
                raise TypeError(error_msg)

            # Validate and set upscale factor
            try:
                scale = max(1, min(4, int(self.valves.UPSCALE_FACTOR or 2)))
                if scale not in (2, 4):
                    scale = 2
            except (ValueError, TypeError) as e:
                error_msg = f"Invalid upscale factor: {self.valves.UPSCALE_FACTOR}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
                
            logger.info(f"Upscale factor set to {scale}x.")

            for i, img_dict in enumerate(base64_images, start=1):
                logger.info(f"Processing image {i}/{len(base64_images)} for upscale.")
                
                try:
                    # Validate and clean base64 image data
                    clean_b64 = self._validate_and_clean_b64(img_dict.get("data", ""), i)
                    
                    # Validate enhance creativity (0 <= x <= 1)
                    enhance_creativity = float(
                        self._get_user_valve(
                            __user__,
                            "ENHANCE_CREATIVITY",
                            self.user_valves.ENHANCE_CREATIVITY,
                        )
                    )
                    if enhance_creativity < 0 or enhance_creativity > 1:
                        logger.warning(f"Enhance creativity {enhance_creativity} is out of range (0 <= x <= 1), using default 0.5")
                        enhance_creativity = 0.5
                        
                    # Get enhance prompt and validate length (max 1500 characters)
                    enhance_prompt = self._get_user_valve(
                        __user__, "ENHANCE_PROMPT", self.user_valves.ENHANCE_PROMPT
                    )
                    if enhance_prompt and len(enhance_prompt) > 1500:
                        enhance_prompt = enhance_prompt[:1500]
                        logger.warning(f"Enhance prompt for image {i} truncated to 1500 characters: {enhance_prompt[:50]}...")

                    # Build payload
                    payload = {
                        "image": clean_b64,
                        "scale": scale,
                        "enhance": bool(
                            self._get_user_valve(
                                __user__, "ENHANCE", self.user_valves.ENHANCE
                            )
                        ),
                        "enhanceCreativity": enhance_creativity,
                    }
                    if enhance_prompt:
                        payload["enhancePrompt"] = enhance_prompt

                    # Make API request
                    resp = await self._run_blocking(
                        self._post_json_blocking, "/image/upscale", payload, True
                    )
                    
                    # Validate response
                    images = resp.get("images")
                    if not isinstance(images, list):
                        error_msg = f"Invalid response from Venice upscale for image {i}: 'images' is not a list"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                        
                    if not all(isinstance(x, str) for x in images):
                        error_msg = f"Invalid response from Venice upscale for image {i}: not all items are strings"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

                    # Get MIME type
                    mime = resp.get("__content_type") or "image/png"
                    logger.info(f"Received {len(images)} upscaled image(s) in response.")
                    
                    # Process each upscaled image
                    for out_b64 in images:
                        try:
                            yield await self._process_image_output(out_b64, mime, i)
                        except Exception as e:
                            error_msg = f"Error processing upscaled image {i}: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            yield error_msg
                            
                except ValueError as e:
                    error_msg = f"Invalid image data for image {i}: {str(e)}"
                    logger.error(error_msg)
                    yield error_msg
                    continue  # Continue with other images
                    
                except RuntimeError as e:
                    error_msg = f"API error upscaling image {i}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    yield error_msg
                    continue  # Continue with other images
            
            logger.info("Successfully upscaled and emitted all images.")
            await self.emit_status("🎉 Image upscale successful", done=True)
            
        except ValueError as e:
            error_msg = f"Invalid input for image upscaling: {str(e)}"
            logger.error(error_msg)
            await self.emit_status("❌ Image upscale failed: Invalid input", done=True)
            yield error_msg
            
        except TypeError as e:
            error_msg = f"Type error in image upscaling: {str(e)}"
            logger.error(error_msg)
            await self.emit_status("❌ Image upscale failed: Type error", done=True)
            yield error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during image upscale: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self.emit_status("❌ Image upscale failed", done=True)
            yield error_msg

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
