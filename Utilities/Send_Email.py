"""
title: Send Email Module
description: Allows users to send themself the current assistant message as an email.
author: Cody
version: 0.1.2
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM0YzRjNGMiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLW1haWwiPjxyZWN0IHdpZHRoPSIyMCIgaGVpZ2h0PSIxNiIgeD0iMiIgeT0iNCIgcng9IjIiLz48cGF0aCBkPSJtMjIgNy0xMCA3TDIgNyIvPjwvc3ZnPg==
changes:
  - 0.1.2: Fixed issue where user object was a tuple instead of a dict, causing an AttributeError.
"""

from pydantic import BaseModel, Field
from typing import Optional
import boto3
import re
import asyncio
import logging
from botocore.exceptions import ClientError

from fastapi.requests import Request
from open_webui.routers.users import Users

# Module-specific logger configuration
logger = logging.getLogger("send_email")
logger.propagate = False  # Prevent propagation to root logger
logger.setLevel(logging.INFO)
# Configure handler at module level per logging guide
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def convert_formatted_text_to_html(text):
    """
    Converts formatted text (markdown-like) to HTML.

    Handles:
    - **bold text**
    - *italic text*
    - - bullet points
    - 1. numbered lists
    - # headers
    - links [text](url)

    Parameters:
        text (str): The formatted text to convert

    Returns:
        str: HTML version of the text
    """
    if not text:
        return ""

    html = text

    # Convert headers (# Header)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)

    # Convert bold (**text**)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)

    # Convert italics (*text*)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

    # Convert bullet points (- item)
    bullet_list_pattern = r"((?:^- .+\n?)+)"

    def replace_bullet_list(match):
        items = match.group(1).split("\n")
        list_items = ""
        for item in items:
            if item.strip().startswith("- "):
                list_items += f"<li>{item.strip()[2:]}</li>"
        return f"<ul>{list_items}</ul>"

    html = re.sub(bullet_list_pattern, replace_bullet_list, html, flags=re.MULTILINE)

    # Convert numbered lists (1. item)
    numbered_list_pattern = r"((?:^\d+\. .+\n?)+)"

    def replace_numbered_list(match):
        items = match.group(1).split("\n")
        list_items = ""
        for item in items:
            if re.match(r"^\d+\. ", item.strip()):
                content = re.sub(r"^\d+\. ", "", item.strip())
                list_items += f"<li>{content}</li>"
        return f"<ol>{list_items}</ol>"

    html = re.sub(
        numbered_list_pattern, replace_numbered_list, html, flags=re.MULTILINE
    )

    # Convert links ([text](url))
    html = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', html)

    # Convert newlines to <br> tags
    html = html.replace("\n", "<br>")

    return html


# Cache for SES clients to avoid recreating them
_ses_client_cache = {}


def get_ses_client(
    aws_region="us-east-1", aws_access_key_id=None, aws_secret_access_key=None
):
    """
    Get or create an SES client with caching.

    Parameters:
        aws_region (str, optional): The AWS region where SES is configured. Defaults to "us-east-1".
        aws_access_key_id (str, optional): AWS default access key.
        aws_secret_access_key (str, optional): AWS secret access key.

    Returns:
        boto3.client: The SES client
    """
    # Suppress AWS SDK logs below WARNING
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Create a cache key based on the region and credentials
    cache_key = f"{aws_region}:{aws_access_key_id}"

    # Return cached client if it exists
    if cache_key in _ses_client_cache:
        return _ses_client_cache[cache_key]

    # Create a new client and cache it
    client = boto3.client(
        "ses",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    _ses_client_cache[cache_key] = client
    return client


def send_email_ses(
    sender: str,
    recipient: str,
    subject: str,
    body_text: str,
    body_html=None,
    sender_name=None,
    aws_region="us-east-1",
    aws_access_key_id=None,
    aws_secret_access_key=None,
):
    """
    Sends an email using Amazon SES.

    Parameters:
        sender (str): The email address of the sender. Must be verified in Amazon SES.
        recipient (str): The email address of the recipient. Must be verified in Amazon SES if using the sandbox environment.
        subject (str): The subject of the email.
        body_text (str): The plain-text body of the email.
        body_html (str, optional): The HTML body of the email. Defaults to None.
        sender_name (str, optional): The display name of the sender (e.g., "AI Assistant"). Defaults to None.
        aws_region (str, optional): The AWS region where SES is configured. Defaults to "us-east-1".
        aws_access_key_id (str, optional): AWS default access key.
        aws_secret_access_key (str, optional): AWS secret access key.

    Returns:
        dict: Response from the send_email call.

    Raises:
        ClientError: If there is an error sending the email.
    """
    # Get or create an SES client (using cache)
    ses_client = get_ses_client(
        aws_region=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Format the sender with display name if provided
    formatted_sender = sender
    if sender_name:
        # Format as "Display Name <email@example.com>"
        formatted_sender = f"{sender_name} <{sender}>"

    # Define the email body format based on input
    if body_html:
        body = {"Text": {"Data": body_text}, "Html": {"Data": body_html}}
    else:
        body = {"Text": {"Data": body_text}}

    try:
        # Send the email
        response = ses_client.send_email(
            Source=formatted_sender,
            Destination={
                "ToAddresses": [recipient],
            },
            Message={"Subject": {"Data": subject}, "Body": body},
        )
        return response
    except ClientError as e:
        logger.error("SES send failure: %s", e.response["Error"]["Message"])
        raise


class Action:
    class Valves(BaseModel):
        show_status: bool = Field(
            default=True,
            description="Show status of the action. When disabled, only citation summaries will be shown.",
        )
        FROM_EMAIL: str = Field(
            default="",
            description="The email address to send from",
        )
        DEFAULT_AI_NAME: str = Field(
            default="AI Assistant",
            description="The default name of the AI to display as the email sender",
        )
        default_subject: str = Field(
            default="Message from AI Assistant",
            description="Default subject line for emails",
        )
        AWS_ACCESS_KEY_ID: str = Field(
            default="",
            description="AWS ACCESS KEY",
        )
        AWS_SECRET_ACCESS_KEY: str = Field(
            default="",
            description="AWS SECRET KEY",
        )
        AWS_REGION: str = Field(
            default="us-east-1",
            description="AWS region for SES",
        )

    def __init__(self):
        self.valves = self.Valves()
        # Handler configuration moved to module level

    async def action(
        self,
        body: dict,
        __request__: Request,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        logger.info("Initializing email action")

        if not __event_emitter__:
            return

        try:
            last_assistant_message = body["messages"][-1]

            # Validate user object and get email
            if not __user__ or not isinstance(__user__, tuple) or len(__user__) == 0:
                logger.error(
                    "User information is not available or in an unexpected format."
                )
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "source": {"name": "Error: sending email"},
                            "document": ["User session information is not available"],
                            "metadata": [{"source": "Send as Email Action Button"}],
                        },
                    }
                )
                return

            user_info = __user__[0]
            if not isinstance(user_info, dict):
                logger.error(
                    f"Expected user_info to be a dict but got {type(user_info)}"
                )
                return

            recipient_email = user_info.get("email")

            if not recipient_email:
                logger.error("No email found in user profile")
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "source": {"name": "Error:sending email"},
                            "document": [
                                "No email address found in your user profile. Please update your profile with an email address."
                            ],
                            "metadata": [{"source": "Send as Email Action Button"}],
                        },
                    }
                )
                return

            # Check for message content
            if not last_assistant_message or "content" not in last_assistant_message:
                logger.error("No assistant message found to send")
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "source": {"name": "Error: sending email"},
                            "document": ["No assistant message found to send"],
                            "metadata": [{"source": "Send as Email Action Button"}],
                        },
                    }
                )
                return

            message_content = last_assistant_message["content"]

            # Validate SES configuration
            if (
                not self.valves.FROM_EMAIL
                or not self.valves.AWS_ACCESS_KEY_ID
                or not self.valves.AWS_SECRET_ACCESS_KEY
            ):
                logger.error(
                    "Missing required SES configuration: FROM_EMAIL or AWS credentials"
                )
                # Emit error event and return
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "source": {"name": "Error: Missing Configuration"},
                            "document": [
                                "Missing required SES configuration: FROM_EMAIL or AWS credentials"
                            ],
                            "metadata": [{"source": "Send as Email Action Button"}],
                        },
                    }
                )
                return

            html_content = convert_formatted_text_to_html(message_content)

            # Send the email
            response = send_email_ses(
                sender=self.valves.FROM_EMAIL,
                recipient=recipient_email,
                subject=self.valves.default_subject,
                body_text=message_content,
                body_html=html_content,
                sender_name=self.valves.DEFAULT_AI_NAME,
                aws_region=self.valves.AWS_REGION,
                aws_access_key_id=self.valves.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=self.valves.AWS_SECRET_ACCESS_KEY,
            )

            logger.info(
                "Email sent to %s with message ID: %s",
                recipient_email,
                response.get("MessageId"),
            )

            # Only show status if show_status is enabled
            if self.valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Email Sent", "done": True},
                    }
                )

            # Always show citation for transaction summary
            await __event_emitter__(
                {
                    "type": "citation",
                    "data": {
                        "source": {"name": "Email sent"},
                        "document": [f"Email successfully sent to {recipient_email}"],
                        "metadata": [{"source": "Send as Email Action Button"}],
                    },
                }
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", "Unknown error")
            logger.error("SES ClientError: %s - %s", error_code, error_message)
            await __event_emitter__(
                {
                    "type": "citation",
                    "data": {
                        "source": {"name": "Error:sending email"},
                        "document": [f"Failed to send email: {error_message}"],
                        "metadata": [{"source": "Send as Email Action Button"}],
                    },
                }
            )
        except Exception as e:
            logger.exception("An unexpected error occurred in send_email action")
            await __event_emitter__(
                {
                    "type": "citation",
                    "data": {
                        "source": {"name": "Error:sending email"},
                        "document": [f"An unexpected error occurred: {e}"],
                        "metadata": [{"source": "Send as Email Action Button"}],
                    },
                }
            )
