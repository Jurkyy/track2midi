# Track2MIDI

A Python tool that converts audio songs into MIDI drum tracks.

## Features

- Convert audio files (WAV, MP3) to MIDI drum tracks
- Detect drum hits and classify them (kick, snare, hi-hat)
- Adjustable sensitivity for detection
- Simple and intuitive GUI interface

## Installation

```bash
# Install dependencies and the package in development mode
poetry install

# Run the application
poetry run track2midi
```

## Development

```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy .

# Linting
poetry run ruff check .
``` 