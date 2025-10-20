# Changelog

## v1.0.1 (2025-10-09)

### Removed
- VERBOSE_LOGGING valve (not needed for simple module)
- Associated logging level configuration logic

### Changed
- Simplified valve configuration section
- Updated version to 1.0.1

## v1.0.0 (2025-10-09)

### Added
- Production code quality refactoring
- Type hints for all functions and classes
- Google-style docstrings with comprehensive documentation
- SHOW_STATUS valve for configurable event emitter control
- Log sanitization and truncation utilities
- Comprehensive input validation
- Improved error handling with context

### Changed
- Refactored user data processing with helper methods
- Enhanced date of birth processing with age calculation
- Improved module structure and organization
- Updated version to 1.0.0
- Standardized valve parameter naming (ALL_CAPS)

## v0.3.0 (2025-10-09)

### Added
- Production code quality refactoring
- Type hints for all functions and classes
- Google-style docstrings with comprehensive documentation
- Verbose logging valve for configurable debug output
- Show status valve for configurable event emitter control
- Log sanitization and truncation utilities
- Comprehensive input validation
- Improved error handling with context

### Changed
- Refactored user data processing with helper methods
- Enhanced date of birth processing with age calculation
- Improved module structure and organization
- Updated version to 0.3.0

## v0.2.0 (2025-10-09)

### Added
- User biography retrieval with privacy controls
- User gender retrieval with privacy controls
- User age calculation from date of birth (privacy-preserving)
- New valve settings for bio, gender, and date of birth retrieval

### Removed
- Last active timestamp retrieval (improves prompt caching efficiency)

### Changed
- Updated version to 0.2.0
- Improved date of birth processing with proper error handling
- Enhanced module description to reflect new capabilities