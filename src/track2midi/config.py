"""Configuration settings for the Track2MIDI application."""

# Audio processing settings
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_SENSITIVITY = 1.0
DEFAULT_PPQN = 480  # Pulses Per Quarter Note (MIDI resolution)

# MIDI drum mapping (General MIDI standard)
GM_DRUM_MAP = {
    "kick": 36,  # Bass Drum 1
    "snare": 38,  # Acoustic Snare
    "hihat_closed": 42,  # Closed Hi-Hat
    "hihat_open": 46,  # Open Hi-Hat
    "crash": 49,  # Crash Cymbal 1
    "ride": 51,  # Ride Cymbal 1
    "tom_low": 45,  # Low Tom
    "tom_mid": 47,  # Mid Tom
    "tom_high": 50,  # High Tom
}

# Feature extraction settings
ONSET_WINDOW_DURATION_MS = 100  # Window duration for feature extraction around onsets
MIN_ONSET_DISTANCE_MS = 50  # Minimum time between detected onsets

# Classification settings
DRUM_TEMPLATES = {
    "kick": {
        "spectral_centroid_mean": 150,
        "spectral_centroid_std": 50,
        "spectral_bandwidth_mean": 200,
        "spectral_bandwidth_std": 100,
    },
    "snare": {
        "spectral_centroid_mean": 1500,
        "spectral_centroid_std": 300,
        "spectral_bandwidth_mean": 1000,
        "spectral_bandwidth_std": 500,
    },
    "hihat_closed": {
        "spectral_centroid_mean": 3000,
        "spectral_centroid_std": 500,
        "spectral_bandwidth_mean": 2000,
        "spectral_bandwidth_std": 1000,
    },
} 