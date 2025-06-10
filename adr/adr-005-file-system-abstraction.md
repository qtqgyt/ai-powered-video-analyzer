# ADR-005: File System Abstraction

## Status
Accepted

## Context
System needs to handle various file operations across different platforms.

## Decision
- Create utility module for file operations ([`file_system.py`](src/utils/file_system.py))
- Use pathlib for cross-platform compatibility
- Centralize file handling logic

## Consequences
+ Better cross-platform support
+ Consistent file handling
+ Easier testing
- Additional abstraction layer