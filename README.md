# Track2MIDI

A Python tool that converts audio songs into MIDI drum tracks.

## Features

- Convert audio files (WAV, MP3) to MIDI drum tracks
- Detect drum hits and classify them (kick, snare, hi-hat, etc.)
- Machine learning based drum classification
- Adjustable sensitivity for detection
- Simple and intuitive GUI interface

## Installation

```bash
# Install dependencies and the package in development mode
poetry install

# Run the application
poetry run track2midi
```

## Machine Learning Classification

Track2MIDI uses a Random Forest classifier to identify different drum types based on audio features. By default, the application will use a synthetic training dataset, but you can train your own model:

```bash
# Train the model with synthetic data (from command line)
poetry run train-model --samples 2000

# Or use the GUI's "Train Model" button in the Machine Learning section
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