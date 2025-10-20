"""
title: User Information Module
description: Retrieve and inject user information (ID, name, email, role, bio, gender, date of birth) into the model's context.
author: Cody
version: 1.0.1
date: 2025-10-09
changelog: User_Info/_changelog.md
"""

import asyncio
import logging
from typing import Callable, Awaitable, Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

# Logger configuration
logger = logging.getLogger("user_info")
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler once
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    # Explicitly set handler level to match logger
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


def sanitize_log_message(message: str, max_length: int = 1000) -> str:
    """Sanitize and truncate log messages to prevent console overflow.
    
    Args:
        message: The log message to sanitize.
        max_length: Maximum length of the log message.
        
    Returns:
        Sanitized and truncated log message.
    """
    if not message:
        return ""
    
    # Truncate if too long
    if len(message) > max_length:
        message = message[:max_length] + "...[TRUNCATED]"
    
    return message


class Filter:
    """User Information Retrieval module for Open WebUI.
    
    This filter retrieves user information and injects it into the model's context.
    All information retrieval is configurable via the Valves settings.
    """

    class Valves(BaseModel):
        """Configuration valves for the User Information Filter."""
        
        # Processing priority
        priority: int = Field(
            default=1,
            description="Priority level for the filter operations.",
        )
        
        # Status control
        SHOW_STATUS: bool = Field(
            default=True,
            description="Show real-time status updates via event emitter.",
        )
        
        # Information retrieval controls
        ALLOW_ID_RETRIEVAL: bool = Field(
            default=True,
            description="Whether to allow retrieval of user IDs."
        )
        ALLOW_NAME_RETRIEVAL: bool = Field(
            default=True,
            description="Whether to allow retrieval of user names."
        )
        ALLOW_EMAIL_RETRIEVAL: bool = Field(
            default=True,
            description="Whether to allow retrieval of user emails."
        )
        ALLOW_ROLE_RETRIEVAL: bool = Field(
            default=True,
            description="Whether to allow retrieval of user roles."
        )
        ALLOW_BIO_RETRIEVAL: bool = Field(
            default=True,
            description="Whether to allow retrieval of user biography."
        )
        ALLOW_GENDER_RETRIEVAL: bool = Field(
            default=True,
            description="Whether to allow retrieval of user gender."
        )
        ALLOW_DATE_OF_BIRTH_RETRIEVAL: bool = Field(
            default=True,
            description="Whether to allow retrieval of user date of birth (converted to age)."
        )

    def __init__(self):
        self.valves = self.Valves()

    def _validate_user_data(self, user_data: Dict[str, Any]) -> bool:
        """Validate user data structure and content.
        
        Args:
            user_data: User information dictionary to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(user_data, dict):
            logger.warning("User data is not a dictionary")
            return False
        
        # Check for required fields if any retrieval is enabled
        if self.valves.ALLOW_ID_RETRIEVAL and "id" in user_data and not user_data["id"]:
            logger.debug("User ID is empty")
        
        return True

    def _process_user_field(
        self,
        user_data: Dict[str, Any],
        field_name: str,
        display_name: str,
        result: List[str],
        retrieved_items: List[str],
        allow_retrieval: bool
    ) -> None:
        """Process a single user field and add it to results if available.
        
        Args:
            user_data: The user data dictionary.
            field_name: The field name in the user data.
            display_name: The display name for the field.
            result: The list to append results to.
            retrieved_items: The list to track retrieved items.
            allow_retrieval: Whether retrieval is allowed for this field.
        """
        if allow_retrieval and field_name in user_data and user_data[field_name]:
            result.append(f"{display_name}: {user_data[field_name]}")
            retrieved_items.append(field_name)
            logger.debug(f"Retrieved user {field_name}")

    def _calculate_age(self, dob_value: Any) -> Optional[int]:
        """Calculate age from date of birth value.
        
        Args:
            dob_value: Date of birth value (string or date object).
            
        Returns:
            Calculated age or None if calculation fails.
        """
        try:
            if isinstance(dob_value, str):
                dob = datetime.strptime(dob_value, "%Y-%m-%d").date()
            else:
                # Assume it's already a date object
                dob = dob_value
            
            today = datetime.today().date()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to calculate age from date of birth: {str(e)}")
            return None

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]],
        __user__: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Retrieve user information and inject it into the model's context.
        
        Args:
            body: The message body from Open WebUI.
            __event_emitter__: Function to emit status updates.
            __user__: User information dictionary from Open WebUI.
            
        Returns:
            Modified body with user information injected into system context.
            
        Raises:
            RuntimeError: If user information retrieval fails.
            ValueError: If input validation fails.
        """
            
        logger.info("Retrieving user information...")
        result: List[str] = []
        retrieved_items: List[str] = []

        try:
            # Validate input user data
            if not self._validate_user_data(__user__):
                logger.warning("Invalid user data structure")
                if self.valves.SHOW_STATUS:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "error",
                                "description": "Invalid user data structure",
                                "done": True,
                            },
                        }
                    )
                # Continue with empty user info rather than failing completely
                __user__ = {}

            # Send initial status update if enabled
            if self.valves.SHOW_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "in_progress",
                            "description": "🚚 Retrieving user information...",
                            "done": False,
                        },
                    }
                )

            # Process user ID
            self._process_user_field(__user__, "id", "ID", result, retrieved_items, self.valves.ALLOW_ID_RETRIEVAL)

            # Process user name
            self._process_user_field(__user__, "name", "User", result, retrieved_items, self.valves.ALLOW_NAME_RETRIEVAL)

            # Process user email
            self._process_user_field(__user__, "email", "Email", result, retrieved_items, self.valves.ALLOW_EMAIL_RETRIEVAL)

            # Process user role
            self._process_user_field(__user__, "role", "Role", result, retrieved_items, self.valves.ALLOW_ROLE_RETRIEVAL)

            # Process user bio
            self._process_user_field(__user__, "bio", "Bio", result, retrieved_items, self.valves.ALLOW_BIO_RETRIEVAL)

            # Process user gender
            self._process_user_field(__user__, "gender", "Gender", result, retrieved_items, self.valves.ALLOW_GENDER_RETRIEVAL)

            # Process user date of birth
            if self.valves.ALLOW_DATE_OF_BIRTH_RETRIEVAL and "date_of_birth" in __user__ and __user__["date_of_birth"]:
                age = self._calculate_age(__user__["date_of_birth"])
                if age is not None:
                    result.append(f"Age: {age}")
                    retrieved_items.append("age")
                    logger.debug(f"Calculated user age: {age}")
                else:
                    # Fallback to showing date of birth if age calculation fails
                    result.append(f"Date of Birth: {__user__['date_of_birth']}")
                    retrieved_items.append("date-of-birth")
                    logger.warning("Failed to calculate age, showing raw date of birth")


            # Handle empty result case
            if not result:
                result = ["User: Unknown"]
                retrieved_items = ["no data"]
                logger.warning("No user information available")

            final_result = " | ".join(result)
            
            # Sanitize the final result to prevent log overflow
            sanitized_result = sanitize_log_message(final_result)

            # Send completion status update if enabled
            if self.valves.SHOW_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "complete",
                            "description": f"🧑 User information retrieved",
                            "done": True,
                        },
                    }
                )

            # Inject into system context
            system_message = {
                "role": "system",
                "content": f"User Information: {final_result}",
            }
            body.setdefault("messages", []).append(system_message)
            logger.info(f"Successfully injected user info into context: {sanitized_result}")

        except Exception as e:
            error_msg = f"User information retrieval failed: {str(e)}"
            logger.error(error_msg)
            # Send error status update if enabled
            if self.valves.SHOW_STATUS:
                try:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "status": "error",
                                "description": "User information retrieval failed",
                                "done": True,
                            },
                        }
                    )
                except Exception as emitter_error:
                    logger.warning(f"Failed to send error status update: {str(emitter_error)}")
            raise RuntimeError(error_msg) from e

        return body
