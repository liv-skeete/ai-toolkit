"""
title: RSS News Module
description: Fetches RSS/Atom feeds synchronously
author: Cody
version: 1.1.0
date: 2025-10-10
changelog: Rss_News/_changelog.md
"""

import html
import ipaddress
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field




# Standalone logger configuration
logger = logging.getLogger("rss_news")
logger.propagate = False
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Filter:
    """Minimal RSS News Filter; fetches configured RSS/Atom feeds and injects headlines."""

    class Valves(BaseModel):
        """Pydantic settings for feed URLs and display options."""

        priority: int = Field(default=0, description="Priority level")
        show_links: bool = Field(default=True)
        per_feed_timeout: int = Field(default=8)
        max_total_articles_display: int = Field(default=12)
        article_description_length: int = Field(default=240)
        max_article_age_days: int = Field(default=2)
        user_agent: str = Field(
            default="MinimalRSS/3.0 (+https://example.invalid/rss)",
            description="User agent string for RSS feed requests"
        )
        max_bytes: int = Field(
            default=2_000_000,
            description="Maximum bytes to read from each feed response"
        )
        verbose_logging: bool = Field(
            default=False,
            description="Enable verbose logging for debugging purposes"
        )
        show_status: bool = Field(
            default=True,
            description="Show status updates via event emitter"
        )
        rss_url: str = Field(
            default=(
                "https://feeds.bbci.co.uk/news/world/rss.xml,"
                "http://rss.cnn.com/rss/cnn_topstories.rss,"
                "https://abcnews.go.com/abcnews/topstories"
            ),
            description="Comma-separated RSS/Atom feed URLs.",
        )
        news_keywords: str = Field(
            default="show news, show headlines, latest news",
            description="Comma-separated keywords to trigger news fetching.",
        )
        assistant_instructions: str = Field(
            default="Summarize the headlines above. Do not invent content.  Include links in the output **only** if full link is provided in context.",
            description="Instructions for the assistant when presenting news.",
        )

    def __init__(self) -> None:
        """Initialize the RSS News Filter with default configuration."""
        self.valves = self.Valves()
        self.processing_news = False
        self.script_version = "3.0.0"
        # Set logger level based on verbose_logging valve
        logger.setLevel(logging.DEBUG if self.valves.verbose_logging else logging.INFO)
        
    def _validate_valves(self) -> None:
        """Validate valve configuration values."""
        # Validate per_feed_timeout
        if self.valves.per_feed_timeout <= 0:
            self._log("Invalid per_feed_timeout value, using default of 8 seconds", level="WARNING")
            self.valves.per_feed_timeout = 8
            
        # Validate max_total_articles_display
        if self.valves.max_total_articles_display <= 0:
            self._log("Invalid max_total_articles_display value, using default of 12", level="WARNING")
            self.valves.max_total_articles_display = 12
            
        # Validate article_description_length
        if self.valves.article_description_length <= 0:
            self._log("Invalid article_description_length value, using default of 240", level="WARNING")
            self.valves.article_description_length = 240
            
        # Validate max_article_age_days
        if self.valves.max_article_age_days < 0:
            self._log("Invalid max_article_age_days value, using default of 2", level="WARNING")
            self.valves.max_article_age_days = 2

    def _log(self, msg: str, level: str = "INFO") -> None:
        """Helper for logging.
        
        Args:
            msg: The message to log
            level: The logging level (default: "INFO")
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        # If verbose_logging is disabled, only log INFO and higher levels
        if not self.valves.verbose_logging and log_level < logging.INFO:
            return
        logger.log(log_level, msg)
        
    def _sanitize_input(self, input_str: str) -> str:
        """Sanitize input string to prevent injection attacks.
        
        Args:
            input_str: Input string to sanitize
            
        Returns:
            Sanitized string
        """
        # Remove any control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', input_str)
        # Limit length to prevent excessive logging
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000] + "...[truncated]"
        return sanitized

    def _fetch_feed_bytes(self, url: str) -> Optional[bytes]:
        """Fetch RSS feed content as bytes with safety checks and size limits.

        Args:
            url: The RSS feed URL to fetch

        Returns:
            Raw feed content as bytes, or None if fetch fails or is unsafe
        """
        if not _is_safe_url(url):
            self._log(f"Skipping unsafe URL: {url}", level="WARNING")
            return None
        headers: Dict[str, str] = {
            "User-Agent": self.valves.user_agent,
            "Accept": (
                "application/rss+xml, application/atom+xml, "
                "application/xml;q=0.9, */*;q=0.8"
            ),
            "Accept-Encoding": "identity",  # keep it simple
        }
        req = Request(url, headers=headers, method="GET")
        try:
            with urlopen(req, timeout=self.valves.per_feed_timeout) as resp:
                cl_hdr = resp.getheader("Content-Length")
                if cl_hdr:
                    try:
                        if int(cl_hdr) > self.valves.max_bytes:
                            self._log(
                                f"Content-Length too large ({cl_hdr}) for {url}",
                                level="WARNING",
                            )
                            return None
                    except ValueError:
                        pass
                data = resp.read(self.valves.max_bytes + 1)
                if len(data) > self.valves.max_bytes:
                    self._log(
                        f"Feed exceeded {self.valves.max_bytes} bytes: {url}", level="WARNING"
                    )
                    return None
                return data
        except HTTPError as e:
            self._log(f"HTTP error for {url}: {e}", level="WARNING")
        except URLError as e:
            self._log(f"URL error for {url}: {e}", level="WARNING")
        except TimeoutError as e:
            self._log(f"Timeout error for {url}: {e}", level="WARNING")
        except Exception as e:
            self._log(f"Generic fetch error for {url}: {e}", level="ERROR")
        return None

    def _parse_feed(self, content: bytes, source_name: str) -> List[Dict[str, Any]]:
        """Parse RSS/Atom feed content and extract article information.

        Args:
            content: Raw feed content as bytes
            source_name: Name of the feed source for logging

        Returns:
            List of article dictionaries with title, link, description, etc.
        """
        articles: List[Dict[str, Any]] = []
        try:
            # ElementTree accepts both str and bytes; pass bytes directly
            root = ET.fromstring(content)
        except ET.ParseError as e:
            self._log(f"XML parse error for {source_name}: {e}", level="ERROR")
            return articles
        except Exception as e:
            self._log(f"Unexpected error parsing XML for {source_name}: {e}", level="ERROR")
            return articles

        # Collect item/entry elements ignoring namespaces
        items = [
            elem for elem in root.iter() if _localname(elem.tag) in {"item", "entry"}
        ]

        for idx, item in enumerate(items):
            # Title
            title: str = ""
            for child in item:
                if _localname(child.tag) == "title" and (child.text or "").strip():
                    title = child.text.strip()
                    break
            if not title:
                # Some entries may place title in alternative tags; skip if none
                self._log(
                    f"Skipping item {idx+1} for {source_name} due to missing title",
                    level="INFO",
                )
                continue

            # Link
            link: str = "No link"
            links = [c for c in item if _localname(c.tag) == "link"]
            chosen: Optional[str] = None
            if links:
                # Prefer rel=alternate text/html if present (Atom)
                for ln in links:
                    rel = (ln.attrib.get("rel") or "").lower()
                    typ = (ln.attrib.get("type") or "").lower()
                    href = ln.attrib.get("href")
                    if rel in {"alternate", ""} and (not typ or "html" in typ) and href:
                        chosen = href.strip()
                        break
                if not chosen:
                    # Fallback: first link with href or text
                    href = links[0].attrib.get("href")
                    chosen = (href or (links[0].text or "").strip()) or "No link"
            link = chosen or link

            # Description / summary
            description: str = ""
            for child in item:
                ln = _localname(child.tag)
                if ln in {"description", "summary"} and (child.text or "").strip():
                    description = _strip_html(child.text or "")
                    break
                # Support content:encoded or <content>
                if ln in {"encoded", "content"} and (child.text or "").strip():
                    description = _strip_html(child.text or "")
                    break
            if (
                description
                and len(description) > self.valves.article_description_length
            ):
                description = (
                    description[: self.valves.article_description_length - 3] + "..."
                )

            # Date
            raw_date: str = ""
            for child in item:
                ln = _localname(child.tag)
                if (
                    ln in {"pubDate", "published", "updated"}
                    and (child.text or "").strip()
                ):
                    raw_date = child.text.strip()
                    break
            dt = _parse_datetime_fallback(raw_date)

            articles.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "source": source_name,
                    "parsed_datetime": dt,
                    "time_ago": _time_ago(dt),
                    "formatted_date": _format_date(dt),
                }
            )

        return articles

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> dict:
        """Process incoming messages and inject RSS news when news queries are detected.

        Args:
            body: The message body containing user messages
            __user__: User information (unused)
            __event_emitter__: Event emitter for status updates (unused)

        Returns:
            Modified message body with RSS news injected if applicable
        """
        # Mark optional parameters as used to satisfy linters
        _ = __user__, __event_emitter__

        messages = body.get("messages", [])
        if not messages:
            return body

        # Minimal intent detection
        last = messages[-1].get("content")
        last_text: str = ""
        if isinstance(last, str):
            last_text = last.lower()
        elif isinstance(last, list):
            for item in last:
                if isinstance(item, dict) and item.get("type") == "text":
                    last_text = (item.get("text") or "").lower()
                    break
        if not last_text:
            return body

        news_keywords = [
            kw.strip() for kw in self.valves.news_keywords.split(",") if kw.strip()
        ]
        if not any(kw in last_text for kw in news_keywords):
            return body

        self.processing_news = True
        self._validate_valves()
        self._log(f"News Query Detected - Fetching RSS... Query: {self._sanitize_input(last_text)}", level="INFO")

        # Collect feed URLs (stdlib-safe)
        feed_urls: List[str] = []
        if self.valves.rss_url and self.valves.rss_url.strip():
            feed_urls = [u.strip() for u in self.valves.rss_url.split(",") if u.strip()]
        # Deduplicate and keep only valid-scheme URLs
        allowed_schemes = {"http", "https"}
        feed_urls = sorted(
            list(
                {
                    u
                    for u in feed_urls
                    if (urlparse(u).scheme in allowed_schemes and urlparse(u).netloc)
                }
            )
        )
        if not feed_urls:
            self._log("No valid RSS feed URLs configured.", level="WARNING")
            return body

        # Fetch sequentially (simple and robust)
        all_articles: List[Dict[str, Any]] = []
        start = time.time()
        for i, feed_url in enumerate(feed_urls, start=1):
            source = urlparse(feed_url).netloc
            self._log(f"Fetching {i}/{len(feed_urls)}: {source}", level="INFO")
            data = self._fetch_feed_bytes(feed_url)
            if not data:
                continue
            try:
                items = self._parse_feed(data, source)
                all_articles.extend(items)
                self._log(f"🔄 {source}: {len(items)} articles", level="INFO")
            except Exception as e:
                self._log(f"Parse error for {source}: {e}", level="ERROR")

        # Filter and sort
        cutoff: Optional[datetime] = None
        if self.valves.max_article_age_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(
                days=self.valves.max_article_age_days
            )

        filtered: List[Dict[str, Any]] = []
        for a in all_articles:
            dt = a.get("parsed_datetime")
            if cutoff and dt:
                if dt >= cutoff:
                    filtered.append(a)
            else:
                # Keep items without date (place them last when sorting)
                filtered.append(a)

        filtered.sort(
            key=lambda x: x.get("parsed_datetime")
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        if len(filtered) > self.valves.max_total_articles_display:
            filtered = filtered[: self.valves.max_total_articles_display]

        elapsed = time.time() - start
        self._log(
            (
                f"Total articles fetched before display: {len(filtered)} "
                f"(in {elapsed:.1f}s)"
            ),
            level="INFO",
        )

        if not filtered:
            failure_content = (
                "❌ RSS News Unavailable\n"
                f"Attempted to fetch news at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, "
                "but no articles were retrieved.\n"
                "This could be due to network issues, invalid RSS feed URLs, "
                "or feeds with no recent content."
            )
            messages.insert(0, {"role": "system", "content": failure_content})
            body["messages"] = messages
            self._log("No articles to display.", level="WARNING")
            return body

        # Build compact system message
        header = (
            f"📰 Current News Headlines ({len(filtered)} articles, "
            f"max {self.valves.max_article_age_days} days old):\n"
        )
        entries: List[str] = []
        for art in filtered:
            line = f"**{art['title']}** ({art['time_ago']})"
            if art["description"]:
                line += f"\n    {art['description']}"

            info: List[str] = []
            if self.valves.show_links and art["link"] != "No link":
                info.append(f"Source: {art['link']}")
            else:
                info.append(f"Source: {art['source']}")
            if art["formatted_date"] != "Unknown date":
                info.append(f"Published: {art['formatted_date']}")

            entries.append(line + f"\n    *{' - '.join(info)}*")

        body_text = "\n\n".join(entries)
        system_msg = (
            header + "\n" + body_text + "\n" + self.valves.assistant_instructions
        )

        messages.insert(0, {"role": "system", "content": system_msg})
        body["messages"] = messages
        self._log(
            f"✅ News loaded: {len(filtered)} articles ({elapsed:.1f}s)", level="INFO"
        )
        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None, __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None) -> dict:
        """Clean up after processing news queries.

        Args:
            body: The message body
            __user__: User information (unused)
            __event_emitter__: Event emitter (unused)

        Returns:
            Unmodified message body
        """
        # Mark optional parameters as used to satisfy linters
        _ = __user__, __event_emitter__

        if self.processing_news:
            self._log("Resetting news processing state.", level="INFO")
        self.processing_news = False
        return body


def _localname(tag: str) -> str:
    """Extract local name from namespaced tag.
    
    Args:
        tag: XML tag which may contain namespace
        
    Returns:
        Local name portion of the tag
    """
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _strip_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace from text.
    
    Args:
        text: Text that may contain HTML tags
        
    Returns:
        Cleaned text with HTML removed and whitespace normalized
    """
    # Remove HTML tags and normalize whitespace
    no_tags = re.sub(r"<[^>]+>", "", text or "")
    return re.sub(r"\s+", " ", html.unescape(no_tags)).strip()


def _parse_iso8601(s: str) -> Optional[datetime]:
    """Basic ISO-8601 handling without external deps.
    
    Args:
        s: String representation of datetime in ISO-8601 format
        
    Returns:
        Parsed datetime object in UTC, or None if parsing fails
    """
    if not s:
        return None
    s = s.strip()
    try:
        if s.endswith("Z"):
            # Python fromisoformat doesn't handle 'Z' directly
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def _parse_datetime_fallback(s: str) -> Optional[datetime]:
    """Try RFC2822 via email.utils, then ISO-8601 fallback.
    
    Args:
        s: String representation of datetime
        
    Returns:
        Parsed datetime object in UTC, or None if parsing fails
    """
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        if dt is None:
            return _parse_iso8601(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return _parse_iso8601(s)


def _format_date(dt: Optional[datetime]) -> str:
    """Format datetime as a readable string.
    
    Args:
        dt: Datetime object to format
        
    Returns:
        Formatted date string or "Unknown date" if None
    """
    return dt.strftime("%b %d, %Y") if dt else "Unknown date"


def _time_ago(dt: Optional[datetime]) -> str:
    """Convert datetime to human-readable time ago string.
    
    Args:
        dt: Datetime object to convert
        
    Returns:
        Human-readable time ago string
    """
    if not dt:
        return "recently"
    now = datetime.now(timezone.utc)
    diff = now - dt
    if diff.total_seconds() < 0:
        return dt.strftime("%b %d, %Y (%Z)")
    if diff.total_seconds() < 60:
        return "just now"
    minutes = int(diff.total_seconds() / 60)
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hr ago"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''} ago"


def _is_safe_url(url: str) -> bool:
    """Lightweight SSRF guard without DNS resolution.
    
    Args:
        url: URL to check for safety
        
    Returns:
        True if URL is safe, False otherwise
        
    Security checks:
    - Allow only http/https
    - Reject localhost and IP literals in private/loopback/link-local/
      reserved/multicast ranges
    """
    try:
        allowed_schemes = {"http", "https"}
        p = urlparse(url)
        if p.scheme not in allowed_schemes:
            return False
        host = p.hostname or ""
        if not host:
            return False
        if host.lower() == "localhost":
            return False
        try:
            ip = ipaddress.ip_address(host)
            if not ip.is_global:
                return False
        except ValueError:
            # Not an IP literal; allow hostnames (no DNS resolution here for simplicity)
            pass
        return True
    except Exception:
        return False
