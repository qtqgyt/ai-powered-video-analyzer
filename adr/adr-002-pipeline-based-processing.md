# ADR-002: Pipeline-based Processing

## Status
Accepted

## Context
Video analysis requires multiple sequential steps with different types of processing.

## Decision
- Implement a pipeline pattern in [`VideoAnalysisPipeline`](src/core/pipeline.py)
- Use step-by-step processing (audio extraction, transcription, object detection, etc.)
- Make steps optional/configurable
- Return structured results dictionary

## Consequences
+ Clear processing flow
+ Easy to modify/skip steps
+ Results are well-structured
- Sequential processing may impact performance