# AI-Powered Video Analyzer

A Python-based video analysis tool that uses multiple AI models to extract insights from video content, including transcription, object detection, and scene description.

## Overview

This project is based on the original work by [Arash Sajjadi](https://github.com/arashsajjadi/ai-powered-video-analyzer). It has been modified and enhanced to meet specific requirements while maintaining the core concept of AI-powered video analysis.

## Features

- Speech transcription using OpenAI's Whisper
- Object detection 
- Scene description (planned)
- Audio event detection (planned)
- Content summarization

## Prerequisites

Before running the video analyzer, ensure:

1. Ollama is installed and running:
2. The Ollama service is running at http://localhost:11434
3. Required models are downloaded:
   - gemma3 (default summarization model)
   - YOLOv8 (object detection)

## Installation

1. Clone this repository:
2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e .
```

## Usage

Basic usage example:
```bash
python -m src.cli analyze video "path/to/video.mp4" --transcription-language en
```

## Project Structure

```
ai-powered-video-analyzer/
├── src/
│   ├── ai_models/       # AI model implementations
│   ├── core/           # Core processing logic
│   ├── cli/            # Command-line interface
│   └── config/         # Configuration management
├── tests/              # Test suite
└── setup.py           # Package configuration
```

## Configuration

Configuration is managed through environment variables and/or a settings file. See `src/config/settings.py` for available options.

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Credits

- Original concept and implementation by [Arash Sajjadi](https://github.com/arashsajjadi/ai-powered-video-analyzer)
- This project uses several open-source AI models:
  - [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition

## License

This project maintains the same license as the original work by Arash Sajjadi. Please refer to the original repository for license details.




