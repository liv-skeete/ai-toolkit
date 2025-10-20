# Module Logging Implementation Guide

## Core Principles
1. **Isolation**  
   - Each module gets its own logger
   - Prevent propagation to root logger
   ```python
   logger = logging.getLogger('Module_Name')
   logger.propagate = False
   ```

2. **Structured Formatting**  
   Standard format for all modules:
   ```python
   '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   ```

3. **Noise Control**  
   - Default level: INFO
   - Third-party libs set to WARNING
   ```python
   logging.getLogger('boto3').setLevel(logging.WARNING)
   ```

4. **Error Tracking**  
   - Use `exception()` for errors
   - Include context (IDs, non-sensitive data)

## Implementation Checklist
1. **Logger Setup**  
   ```python
   import logging
   logger = logging.getLogger('Your_Module_Name')
   logger.propagate = False  # Critical: prevent propagation to root logger
   logger.setLevel(logging.INFO)
   ```

2. **Handler Configuration**  
   ```python
   # Remove any existing handlers to ensure clean configuration
   for handler in logger.handlers[:]:
       logger.removeHandler(handler)
       
   # Configure handler
   handler = logging.StreamHandler()
   formatter = logging.Formatter(
       '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   handler.setFormatter(formatter)
   # Explicitly set handler level to match logger
   handler.setLevel(logging.INFO)
   logger.addHandler(handler)
   ```

3. **Third-Party Control**  
   Add to module initialization:
   ```python
   # Silence common libraries that might log at INFO level
   logging.getLogger('aiohttp').setLevel(logging.WARNING)
   logging.getLogger('asyncio').setLevel(logging.WARNING)
   logging.getLogger('open_webui').setLevel(logging.WARNING)
   logging.getLogger('urllib3').setLevel(logging.WARNING)
   logging.getLogger('requests').setLevel(logging.WARNING)
   logging.getLogger('httpx').setLevel(logging.WARNING)
   logging.getLogger('httpcore').setLevel(logging.WARNING)
   ```

4. **Message Taxonomy**  
   | Level    | Usage Scenario                          |
   |----------|-----------------------------------------|
   | DEBUG    | Detailed troubleshooting                |
   | INFO     | Service milestones                      |
   | WARNING  | Unexpected but recoverable conditions   |
   | ERROR    | Failed operations                       |
   | CRITICAL | System-level failures                   |

5. **Resource Management**  
   For modules using external resources (e.g., HTTP sessions):
   ```python
   import atexit
   import asyncio
   
   # Module-level resource management
   _session = None
   
   async def get_session():
       """Get or create a shared resource."""
       global _session
       if _session is None or _session.closed:
           logger.debug("Creating new resource")
           _session = create_resource()
       return _session
   
   async def close_session():
       """Close the shared resource if it exists."""
       global _session
       if _session and not _session.closed:
           logger.debug("Closing resource")
           await _session.close()
           _session = None
   
   def cleanup():
       """Cleanup function to ensure resource is closed on module unload."""
       global _session
       if _session and not _session.closed:
           logger.debug("Running cleanup for resource")
           try:
               loop = asyncio.get_event_loop()
           except RuntimeError:
               loop = asyncio.new_event_loop()
               asyncio.set_event_loop(loop)
           
           if loop.is_running():
               loop.create_task(close_session())
           else:
               loop.run_until_complete(close_session())
   
   # Register cleanup function to run on module unload
   atexit.register(cleanup)
   ```

## Code Examples
**Basic Template**
```python
import logging

# Logger configuration
logger = logging.getLogger('New_Module')
logger.propagate = False
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Configure handler
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# Silence common libraries
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Usage
logger.info("Service initialized")
```

**Error Handling**
```python
try:
    risky_operation()
except Exception as e:
    logger.error("Failed to perform operation - %s", error_code)
    logger.exception("Error details: ")
```

**Resource Management Example (HTTP Client)**
```python
import logging
import asyncio
import atexit
import aiohttp

# Logger configuration
logger = logging.getLogger('http_client_module')
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

# Module-level session management
_session = None

async def get_session():
    """Get or create a shared aiohttp ClientSession."""
    global _session
    if _session is None or _session.closed:
        logger.debug("Creating new aiohttp ClientSession")
        _session = aiohttp.ClientSession()
    return _session

async def close_session():
    """Close the shared aiohttp ClientSession if it exists."""
    global _session
    if _session and not _session.closed:
        logger.debug("Closing aiohttp ClientSession")
        await _session.close()
        _session = None

def cleanup():
    """Cleanup function to ensure session is closed on module unload."""
    global _session
    if _session and not _session.closed:
        logger.debug("Running cleanup for aiohttp ClientSession")
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            logger.debug("Event loop is running, creating cleanup task")
            loop.create_task(close_session())
        else:
            logger.debug("Running event loop to close session")
            loop.run_until_complete(close_session())

# Register cleanup function to run on module unload
atexit.register(cleanup)

# Example usage in a class
class Client:
    async def make_request(self, url):
        session = await get_session()  # Use shared session
        try:
            async with session.get(url) as response:
                return await response.text()
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None
```

## Maintenance Guidelines
1. **Naming Convention**  
   - Use lowercase with underscores for module logger names
   - Match logger name to module functionality

2. **Level Management**  
   ```python
   # Temporary debug override
   logger.setLevel(logging.DEBUG)
   ```

3. **Security**  
   - Never log credentials/PII
   - Sanitize data before logging
   ```python
   logger.info("User action: %s", user_id[:4]+"****")  # Masked ID
   ```

4. **Rotation** (When using files)
   ```python
   from logging.handlers import RotatingFileHandler
   handler = RotatingFileHandler(
       'app.log', maxBytes=1e6, backupCount=3
   )
   ```

5. **Troubleshooting Code Logging Issues**
   If module logs are appearing in system logs despite `propagate=False`:
   
   - Ensure logger name is consistent throughout the module
   - Check for direct calls to `logging.xxx()` instead of `logger.xxx()`
   - Verify handler levels match logger level
   - Consider using a NullHandler for complete isolation:
     ```python
     logger.addHandler(logging.NullHandler())
     ```
   - For modules with external resources (HTTP clients, etc.), implement proper resource management with cleanup
