# Automatic Memory Manager: Clean Slate Logging Plan

## 1. Current Issues

The current logging implementation in the Automatic Memory Manager has several critical issues:

- **Stack dumps** appear in logs at all log levels (DEBUG to CRITICAL)
- These dumps occur on **every memory operation**
- The current `__repr__` and `__str__` methods are not preventing full code dumps
- Log level filtering is not working effectively
- Large data structures are being logged without proper truncation
- Too much code is dedicated to logging functionality

## 2. Goals for New Logging System

1. **Minimalism**: Keep logging code to an absolute minimum
2. **Clarity**: Make logs concise, readable, and useful for debugging
3. **Control**: Provide effective log level filtering
4. **Performance**: Minimize logging overhead
5. **Maintainability**: Make it easy to add new logging without causing issues

## 3. Core Principles

1. **Separation of Concerns**: 
   - Logging should be a thin layer that doesn't interfere with core functionality
   - Logging code should be isolated and easy to maintain

2. **Fail-Safe Design**:
   - Logging should never cause exceptions or errors
   - If logging fails, it should fail silently and not affect the main code

3. **Minimal Footprint**:
   - Logging code should be a small percentage of the overall codebase
   - Avoid complex logging frameworks or extensive helper methods

4. **Effective Defaults**:
   - Default log formats should be useful without configuration
   - Objects should have sensible string representations by default

## 4. Implementation Approach

### Phase 1: Strip Existing Logging

1. Remove all existing logging-related code:
   - Remove all `logger.xxx()` calls
   - Remove logging-specific helper methods
   - Remove custom `__repr__` and `__str__` methods
   - Remove log level configuration from valves

2. Identify essential logging points:
   - Entry/exit of key methods
   - Important state changes
   - Error conditions
   - API calls

### Phase 2: Implement Minimal Logging Framework

1. Create a simple logging proxy:
   ```python
   class LogProxy:
       def __init__(self, name, level=logging.INFO):
           self._logger = logging.getLogger(name)
           self._logger.setLevel(level)
           
       def _safe_log(self, level, msg, *args):
           """Log safely, preventing object dumps."""
           if not self._logger.isEnabledFor(level):
               return
               
           # Convert args to safe strings
           safe_args = []
           for arg in args:
               safe_args.append(self._safe_str(arg))
               
           self._logger.log(level, msg, *safe_args)
           
       def _safe_str(self, obj, max_len=100):
           """Convert any object to a safe string representation."""
           try:
               if obj is None:
                   return "None"
                   
               if isinstance(obj, str):
                   if len(obj) > max_len:
                       return obj[:max_len] + "..."
                   return obj
                   
               if isinstance(obj, (list, tuple)):
                   return f"{type(obj).__name__}[{len(obj)} items]"
                   
               if isinstance(obj, dict):
                   return f"dict[{len(obj)} items]"
                   
               # Use class name and basic info
               return f"{obj.__class__.__name__}(...)"
           except:
               return "<unprintable>"
               
       def debug(self, msg, *args):
           self._safe_log(logging.DEBUG, msg, *args)
           
       def info(self, msg, *args):
           self._safe_log(logging.INFO, msg, *args)
           
       def warning(self, msg, *args):
           self._safe_log(logging.WARNING, msg, *args)
           
       def error(self, msg, *args):
           self._safe_log(logging.ERROR, msg, *args)
           
       def critical(self, msg, *args):
           self._safe_log(logging.CRITICAL, msg, *args)
   ```

2. Add minimal string representations to key classes:
   ```python
   class Filter:
       def __str__(self):
           return f"AMM Filter v{self.version}"
           
   class MemoryOperation:
       def __str__(self):
           content_preview = ""
           if self.content:
               content_preview = (self.content[:20] + "...") if len(self.content) > 20 else self.content
           return f"{self.operation} Memory: {content_preview}"
   ```

3. Add a simple log level valve:
   ```python
   # In the Valves class
   log_level: str = Field(
       default="INFO",
       description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
   )
   ```

### Phase 3: Add Essential Logging Points

1. **Entry/Exit of Key Methods**:
   ```python
   # At the start of a method
   logger.debug("Starting %s", method_name)
   
   # At the end of a method
   logger.debug("Completed %s", method_name)
   ```

2. **Important State Changes**:
   ```python
   logger.info("Memory state changed: %s", state_description)
   ```

3. **Error Conditions**:
   ```python
   logger.error("Error in %s: %s", context, error_description)
   ```

4. **API Calls**:
   ```python
   logger.debug("API call to %s", endpoint)
   ```

### Phase 4: Add Log Configuration

1. Add a method to configure the logger:
   ```python
   def _configure_logger(self):
       """Configure the logger based on the valve settings."""
       level_name = self.valves.log_level.upper()
       level = getattr(logging, level_name, logging.INFO)
       self._logger.setLevel(level)
   ```

2. Call this method when valves are updated:
   ```python
   def update_valves(self, new_valves):
       # Update valves
       for key, value in new_valves.items():
           if hasattr(self.valves, key):
               setattr(self.valves, key, value)
               
       # Reconfigure logger if log level changed
       if "log_level" in new_valves:
           self._configure_logger()
   ```

## 5. Testing Strategy

1. **Verify Log Output**:
   - Check that logs are concise and readable
   - Ensure no stack dumps occur at any log level
   - Verify that log level filtering works correctly

2. **Performance Testing**:
   - Measure the impact of logging on memory operations
   - Ensure logging doesn't significantly slow down the system

3. **Error Handling**:
   - Test that logging errors don't affect the main code
   - Verify that invalid objects don't cause logging failures

## 6. Implementation Timeline

1. **Phase 1 (Strip Existing Logging)**: 1 day
2. **Phase 2 (Implement Minimal Framework)**: 1 day
3. **Phase 3 (Add Essential Logging Points)**: 1-2 days
4. **Phase 4 (Add Log Configuration)**: 1 day
5. **Testing and Refinement**: 1-2 days

Total estimated time: 5-7 days

## 7. Success Criteria

1. No stack dumps in logs at any log level
2. Logging code is less than 10% of the total codebase
3. Logs are concise, readable, and useful for debugging
4. Log level filtering works correctly
5. Logging has minimal impact on performance

## 8. Implementation Status

### Phase 1: Strip Existing Logging - COMPLETED

- ✅ Removed all `logger.xxx()` calls
- ✅ Removed logging-specific helper methods (`_safe_log_content`, `_safe_log_exception`)
- ✅ Removed custom `__repr__` and `__str__` methods from `MemoryOperation` and `Filter` classes
- ✅ Removed log level configuration from valves (`log_level` and `log_truncate_length` fields)
- ✅ Removed logger initialization and configuration code
- ✅ Removed `import logging` statement
- ✅ Replaced exception handling logging with silent passes where appropriate

The code now has a clean slate with all logging functionality removed. This provides a solid foundation for implementing the new minimal logging framework as outlined in the plan. The core functionality of the Automatic Memory Manager remains intact, but without any of the problematic logging code that was causing stack dumps and excessive output.

Next steps:
- Proceed with Phase 2: Implement Minimal Logging Framework
- Add the `LogProxy` class
- Add minimal string representations to key classes
- Add a simple log level valve