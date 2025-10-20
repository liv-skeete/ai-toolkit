# RSS News Module Changelog

## v1.1.0 (2025-10-10)

### Refactored for Open WebUI Standards

- **Code Quality & Readability**
  - Added comprehensive type annotations throughout the module
  - Improved code organization and structure
  - Added detailed docstrings for all functions and classes
  - Removed global constants and externalized to Valves configuration

- **Configuration & Logging**
  - Added `verbose_logging` valve for configurable log verbosity
  - Added `show_status` valve for status updates control
  - Added `user_agent` and `max_bytes` valves for RSS fetching configuration
  - Improved logger configuration with proper level handling

- **Error Handling & Validation**
  - Enhanced error handling with specific exception types
  - Added valve validation to ensure valid configuration values
  - Added input sanitization to prevent injection attacks
  - Improved error messages with better context

- **Security**
  - Maintained SSRF protection with URL validation
  - Added input sanitization for logging user data
  - Ensured no sensitive data is logged

- **Performance**
  - Maintained existing performance characteristics
  - Added validation to prevent excessive resource usage

## v1.0.0 (2025-08-29)

- Initial release