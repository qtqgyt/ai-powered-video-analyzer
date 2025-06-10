# ADR-006: Error Handling and Logging

## Status
Accepted

## Context
Need consistent error handling and logging across the application.

## Decision
- Centralized logging setup ([`logging_setup.py`](src/utils/logging_setup.py))
- Structured error handling hierarchy
- Debug logging for AI operations

## Consequences
+ Consistent error handling
+ Better debugging capabilities
+ Structured logging
- Need to maintain logging standards