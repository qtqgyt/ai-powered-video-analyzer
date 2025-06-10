# ADR-003: Configuration Management

## Status
Accepted

## Context
System needs flexible configuration for models, paths, and processing options.

## Decision
- Use centralized settings in [`settings.py`](src/config/settings.py)
- Support environment variables for configuration
- Use class-based settings management

## Consequences
+ Easy to modify settings
+ Environment-specific configuration
+ Clear configuration hierarchy
- Need to manage multiple configuration sources