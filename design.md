# Design Document: Audio to MIDI Drum Track Converter (Python Prototype)

**Project Goal:** To design a prototype of a software tool using Python that converts audio songs into MIDI drum tracks.
**Based on Plan:** `testament.md`
**Date:** May 19, 2025

---

## 1. Introduction

This document outlines the software design for a prototype of the Audio to MIDI Drum Track Converter. Python has been chosen as the primary programming language due to its extensive libraries for audio analysis, machine learning, MIDI manipulation, and rapid prototyping capabilities. This design focuses on achieving the Minimum Viable Product (MVP) features outlined in `testament.md`, emphasizing simplicity and modularity.

---

## 2. Core Python Libraries and Justification

The following Python libraries are proposed for the prototype:

* **Audio Handling & Input/Output:**
    * **Soundfile:** For loading and saving audio files. It provides access to the raw audio data as NumPy arrays and information like sample rate, number of channels, and encoding.
        * *Justification:* A widely used library for reading/writing audio files, based on `libsndfile`. It's robust and supports many formats.
* **Numerical Operations & Audio Analysis:**
    * **NumPy:** For efficient numerical computation, especially array manipulation for audio data. It will be used for tasks like mono conversion, normalization, and implementing basic audio analysis algorithms (e.g., RMS energy, simple onset detection, basic spectral features from FFT).
        * *Justification:* Fundamental for scientific computing in Python. Many audio analysis tasks can be built using NumPy operations on the raw data provided by `soundfile`.
* **MIDI Generation:**
    * **Mido:** For creating and manipulating MIDI messages and files.
        * *Justification:* Lightweight, straightforward API for direct MIDI event creation, suitable for MVP. (PrettyMIDI is an alternative for more complex MIDI tasks later).
* **User Interface (GUI):**
    * **Tkinter:** For building the graphical user interface.
        * *Justification:* Built into Python's standard library, making it easy to get started without external dependencies for a simple MVP GUI. PyQt could be considered for more complex UIs later.
* **Machine Learning (Optional for MVP, placeholder for future):**
    * **Scikit-learn:** If ML-based classification is pursued (e.g., SVM, Random Forest).
        * *Justification:* Comprehensive and easy-to-use library for classical ML algorithms. (For MVP, a rule-based or template-matching approach will be prioritized).
    * **Librosa (Optional, for specific advanced features):** While the goal is to minimize dependencies, `librosa` could be considered for specific, complex features like MFCCs or advanced onset/beat tracking if implementing them with NumPy proves too time-consuming for the MVP or later stages.
* **Configuration:**
    * **JSON module (built-in):** For managing drum templates or simple configurations.
        * *Justification:* Easy to use, human-readable format for simple key-value configurations.

---

## 3. Module-wise Design (Python Specifics)

The software will be broken down into the following Python modules:

### 3.1. Audio Input & Preprocessing Module (`audio_processor.py`)

* **Purpose:** Handles audio file loading, decoding, and initial preparation.
* **Key Functions/Classes:**
    * `load_audio(file_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, float]:`
        * Uses `soundfile.read(file_path)` to load audio, which returns `(audio_data_np_array, native_sample_rate_float)`.
        * If `target_sr` is provided and different from `native_sample_rate_float`:
            * Resampling would be needed. `soundfile` itself does not resample. For MVP, consider processing at native sample rate, or implement/use a simple resampling function (e.g., using `numpy.interp` or a dedicated library like `resampy` if high quality is essential later). This function might initially skip resampling or raise a warning if `target_sr` is set.
        * Converts audio to mono: If `audio_data.ndim > 1` and `audio_data.shape[1] > 1` (e.g., stereo), it's converted to mono (e.g., `audio_data = audio_data.mean(axis=1)`).
        * Returns a tuple: `(mono_audio_data_np_array, sample_rate_float)`.
    * `normalize_audio(audio_data: np.ndarray) -> np.ndarray:`
        * Normalizes the audio waveform using NumPy (e.g., peak normalization: `audio_data / np.max(np.abs(audio_data))`, or RMS-based normalization).
* **Data Structures:**
    * `np.ndarray`: For storing audio waveform data.

### 3.2. Analysis Engine Module (`analysis_engine.py`)

* **Purpose:** Core logic for detecting rhythmic events and classifying drum sounds.
* **Key Functions/Classes:**
    * **Onset Detection (MVP: Basic Energy-based):**
        * `detect_onsets(audio_data: np.ndarray, sr: float, sensitivity_factor: float = 1.0, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:`
            * Calculates a simple novelty function, e.g., by looking at changes in short-term RMS energy across frames.
            * Uses `NumPy` for frame-wise energy calculation and peak picking in the novelty curve. The `sensitivity_factor` would adjust the peak picking threshold.
            * Returns `np.ndarray` of onset times in seconds.
            * (Note: More sophisticated methods like those in `librosa` might be considered for future enhancements.)
    * **Tempo and Beat Estimation (MVP: Simplified/Placeholder):**
        * `estimate_tempo(audio_data: np.ndarray, sr: float) -> Optional[float]:`
            * For MVP, this might be a very basic estimation (e.g., based on average onset intervals if available and somewhat regular) or return a default/user-provided value.
            * Reliable tempo estimation from arbitrary audio is complex and might rely on more advanced algorithms (e.g., from `librosa` or custom autocorrelation-based methods with `NumPy`) in future iterations.
            * Returns tempo in BPM (float) or `None`.
    * **Feature Extraction (MVP: Basic Features with NumPy):**
        * `extract_features_for_onset(audio_data: np.ndarray, sr: float, onset_time: float, window_duration_ms: int = 100) -> Dict[str, any]:`
            * Extracts an audio segment (window) around `onset_time` using `NumPy`.
            * Calculates basic features for this segment using `NumPy`:
                * RMS energy (for velocity estimation).
                * Basic spectral features: e.g., spectral centroid, spectral spread, calculated from an FFT of the window (using `numpy.fft.rfft`).
                * (Note: MFCCs and other advanced spectral features like those in `librosa` would require more complex implementations or using `librosa` for future enhancements).
            * Returns a dictionary: `{'rms': float, 'spectral_centroid': float, ...}`.
    * **Drum Sound Classification (MVP: Rule-based/Template Matching):**
        * `DrumTemplates`: Class or dictionary to hold characteristic feature values for different drum types.
            * Example template structure (loaded from `drum_templates.json`):
                ```json
                {
                    "kick": {"spectral_centroid_mean": 150, "spectral_centroid_std": 50, "rms_min": 0.2},
                    "snare": {"spectral_centroid_mean": 1500, "rms_min": 0.1},
                    "hihat_closed": {"spectral_centroid_mean": 3000, "rms_min": 0.05}
                }
                ```
        * `load_drum_templates(template_file_path: str = "drum_templates.json") -> Dict:`
            * Loads drum sound characteristic templates from a JSON file.
        * `classify_drum_hit(features: Dict[str, any], templates: Dict) -> str:`
            * Compares extracted `features` against `templates` using a similarity metric (e.g., weighted Euclidean distance of feature means, or a set of `if/elif` rules based on feature ranges).
            * Returns a string label (e.g., "kick", "snare", "hihat_closed", "unknown").
            * For MVP, this will be simple rules (e.g., kick if low spectral centroid, hi-hat if high).
* **Data Structures:**
    * `np.ndarray`: For onset times, feature vectors.
    * `List[Dict[str, any]]`: To store features for each detected onset.
    * `Dict`: For drum templates.

### 3.3. MIDI Generation Module (`midi_generator.py`)

* **Purpose:** Converts detected and classified drum events into a standard MIDI file.
* **Key Functions/Classes:**
    * `GM_DRUM_MAP: Dict[str, int] = {"kick": 36, "snare": 38, "hihat_closed": 42, ...}` (Constant)
    * `DetectedDrumEvent = TypedDict('DetectedDrumEvent', {'time_seconds': float, 'label': str, 'amplitude': float})` (Using `typing.TypedDict` for clarity)
    * `create_midi_file(drum_events: List[DetectedDrumEvent], tempo_bpm: Optional[float], output_path: str, ppqn: int = 480, default_tempo: float = 120.0) -> None:`
        * Initializes `mido.MidiFile(ticks_per_beat=ppqn)`.
        * Creates a `mido.MidiTrack`.
        * Uses `tempo_bpm` if provided and valid, otherwise uses `default_tempo`.
        * Adds a tempo meta message: `mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(current_tempo_bpm))`.
        * For each `event` in `drum_events`:
            * Map `event['label']` to a MIDI note number using `GM_DRUM_MAP`.
            * Convert `event['amplitude']` (e.g., RMS energy from feature extraction, normalized 0-1) to MIDI velocity (0-127). Clamp values.
            * Calculate MIDI event time in ticks: `delta_time_ticks = mido.second2tick(event['time_seconds'] - previous_event_time_seconds, ticks_per_beat=ppqn, tempo=mido.bpm2tempo(tempo_bpm))`. Mido handles absolute vs delta times when adding messages.
            * Create `mido.Message('note_on', note=midi_note, velocity=midi_velocity, time=calculated_tick_time_from_start_or_delta)`.
            * Create `mido.Message('note_off', note=midi_note, velocity=0, time=calculated_tick_time_from_start_or_delta + short_duration_ticks)`. (For drums, `note_off` time is often just after `note_on` with a short fixed duration).
        * Saves the MIDI file using `mid.save(output_path)`.
* **Data Structures:**
    * `List[DetectedDrumEvent]`: Input list of classified drum hits.
    * `mido.MidiFile`, `mido.MidiTrack`, `mido.Message` objects.

### 3.4. User Interface (UI) Module (`main_gui.py`)

* **Purpose:** Provides the graphical interface for user interaction.
* **Key Classes:**
    * `DrumConverterApp(tk.Tk):`
        * Inherits from `tkinter.Tk`.
        * **Widgets:**
            * `tk.Button` for "Load Audio", "Convert to MIDI", "Save MIDI".
            * `tk.Label` to display file names, status messages.
            * `ttk.Progressbar` for visual feedback during processing.
            * `tk.Scale` or `tk.Entry` for sensitivity parameter.
        * **Instance Variables:**
            * `self.audio_file_path: str`
            * `self.output_midi_path: str`
            * `self.sensitivity_var: tk.DoubleVar`
        * **Methods:**
            * `_load_audio_file(self):` Opens file dialog, stores path.
            * `_save_midi_file(self):` Opens save dialog, stores path.
            * `_start_conversion(self):`
                * Retrieves parameters from UI elements.
                * Runs the conversion pipeline (audio loading, analysis, MIDI generation) in a separate thread (`threading.Thread`) to prevent UI freezing.
                * Updates progress bar and status label.
            * `_conversion_thread_target(self):` The actual conversion logic called by the thread.
                * Calls functions from `audio_processor`, `analysis_engine`, `midi_generator`.
                * Handles exceptions and updates UI with success/failure.
* **Data Structures:**
    * Tkinter variable classes (`tk.StringVar`, `tk.DoubleVar`) to link UI elements with internal state.

### 3.5. Configuration Module (`config.py` or `drum_templates.json`)

* **Purpose:** Store application settings and drum sound definitions.
* `config.py`:
    * `DEFAULT_SAMPLE_RATE = 22050`
    * `DEFAULT_SENSITIVITY = 1.0`
    * `GM_DRUM_MAP = {"kick": 36, ...}` (can also be in `midi_generator.py`)
    * `DEFAULT_PPQN = 480`
* `drum_templates.json` (for rule-based classification):
    * As described in section 3.2.

---

## 4. Main Application Flow (`main_script.py` or within `main_gui.py`)

1.  Initialize `DrumConverterApp` (Tkinter main window).
2.  User clicks "Load Audio": `_load_audio_file()` is called.
3.  User adjusts sensitivity (if available).
4.  User clicks "Convert to MIDI": `_start_conversion()` is called.
    * A new thread is spawned, executing `_conversion_thread_target()`.
    * Inside the thread:
        a.  `audio_data, sr = audio_processor.load_audio(self.audio_file_path, target_sr=config.DEFAULT_SAMPLE_RATE)` (Note: `target_sr` handling depends on `load_audio` implementation, might process at native SR)
        b.  `audio_data = audio_processor.normalize_audio(audio_data)`
        c.  `onset_times_seconds = analysis_engine.detect_onsets(audio_data, sr, sensitivity_factor=self.sensitivity_var.get())`
        d.  `tempo_bpm = analysis_engine.estimate_tempo(audio_data, sr)` (May be `None`)
        e.  `templates = analysis_engine.load_drum_templates()` (if using templates)
        f.  `detected_drum_events: List[DetectedDrumEvent] = []`
        g.  For each `t_onset` in `onset_times_seconds`:
            i.  `features = analysis_engine.extract_features_for_onset(audio_data, sr, t_onset)`
            ii. `label = analysis_engine.classify_drum_hit(features, templates)`
            iii.If `label != "unknown"`:
                `amplitude = features['rms']` (or other relevant amplitude measure)
                `detected_drum_events.append({'time_seconds': t_onset, 'label': label, 'amplitude': amplitude})`
        h.  If `detected_drum_events` is not empty:
            `self.output_midi_path = ...` (prompt user to save or auto-generate name)
            `midi_generator.create_midi_file(detected_drum_events, tempo_bpm, self.output_midi_path, ppqn=config.DEFAULT_PPQN)`
            Update UI: "Conversion successful. MIDI saved to X."
        i.  Else:
            Update UI: "No drum events detected."
    * UI thread updates progress based on messages from the worker thread.
5.  Tkinter main loop (`app.mainloop()`) keeps the UI running.

---

## 5. Data Flow (Python Objects)

`file_path: str`
  `-> (audio_processor)` `audio_data: np.ndarray, sr: float`
  `-> (analysis_engine)` `onset_times_seconds: np.ndarray`
  `-> (analysis_engine)` `features_per_onset: List[Dict[str, any]]`
  `-> (analysis_engine)` `classified_hits: List[Dict{'time_seconds': float, 'label': str, 'amplitude': float}]`
  `-> (analysis_engine)` `tempo_bpm: Optional[float]`
  `-> (midi_generator)` `mido_object: mido.MidiFile`
  `->` `.mid file (disk)`

---

## 6. Error Handling Strategy

* Use `try-except` blocks extensively:
    * `audio_processor.py`: For `FileNotFoundError`, audio decoding errors (e.g., from `soundfile`).
    * `analysis_engine.py`: For numerical errors during feature extraction with `NumPy`.
    * `midi_generator.py`: For file writing errors (`IOError`).
    * `main_gui.py`: To catch errors from backend modules and display user-friendly messages via `tkinter.messagebox`.
* Log detailed errors to the console for debugging.
* The UI thread should not perform blocking operations; these must be delegated to worker threads.

---

## 7. Directory Structure (Prototype)

```
audio_to_midi_drum_converter/
├── main_gui.py                 # Main application and UI logic (Tkinter)
├── audio_processor.py          # Audio loading and preprocessing
├── analysis_engine.py          # Onset detection, feature extraction, classification
├── midi_generator.py           # MIDI file creation
├── config.py                   # Application settings, constants (or use JSON)
├── drum_templates.json         # (If using rule-based classification templates)
├── tests/                      # (Future placeholder for unit tests)
│   └── ...
├── README.md
├── testament.md
└── design.md
```

---

## 8. Future Enhancements

This section outlines potential improvements and new features that can be added to the Python prototype after the MVP is established and validated. These enhancements aim to improve accuracy, usability, and the range of capabilities.

### 8.1. Advanced Drum Sound Classification
* **Machine Learning Integration:**
    * Transition from rule-based/template matching to more robust machine learning models using `scikit-learn`.
    * Train classifiers (e.g., SVM, Random Forest, Gradient Boosting) on a labeled dataset of drum sounds (e.g., extracted onsets from IDMT-SMT-DRUMS, ENST-Drums, or a custom dataset).
    * Features for ML: MFCCs, spectral contrast, chroma features, zero-crossing rate, etc., extracted from raw audio data (provided by `soundfile`) using libraries like `librosa` or custom `NumPy` implementations.
    * Consider Deep Learning (e.g., CNNs with `TensorFlow/Keras` or `PyTorch`) if spectrograms are used directly as input for classification. This would require a more substantial dataset and training infrastructure.
* **Expanded Drum Kit Detection:**
    * Train models to identify a wider range of drum instruments: open hi-hat, various toms (high, mid, low), crash cymbals, ride cymbals.
    * Investigate methods to distinguish between different articulations (e.g., ride bell vs. ride bow, rimshot vs. regular snare).
* **User-Trainable Models:**
    * Allow users to provide their own labeled samples of drum hits to fine-tune or retrain the classification model, improving accuracy for specific drum kits or recording styles.

### 8.2. Improved MIDI Output and Control
* **Velocity Dynamics:**
    * Refine velocity mapping. Instead of direct RMS, explore perceptually weighted energy or machine learning models to predict MIDI velocity more accurately from audio features.
* **Quantization Options:**
    * Implement more sophisticated quantization in `midi_generator.py`. This might involve beat tracking, for which libraries like `librosa` (`librosa.beat.beat_track`) could be used to get precise beat locations.
    * Offer user-selectable quantization grids (e.g., 8th, 16th, 32nd notes, triplets) and strength (how strictly notes snap to the grid).
* **Humanization:**
    * Add options for subtle, random variations in timing and velocity to make the MIDI output sound less robotic.
* **Note Duration for Sustained Sounds:**
    * For cymbals or open hi-hats, attempt to estimate the decay time from the audio signal to set more appropriate MIDI note durations, rather than a fixed short duration. This is challenging but can improve realism.

### 8.3. Enhanced Audio Processing
* **Source Separation Preprocessing:**
    * Integrate an optional drum stem isolation step using a pre-trained source separation model like Spleeter (via its Python library) or Demucs. This would be part of `audio_processor.py`. The processed stem would then be read by `soundfile`.
    * Processing only the drum stem can significantly improve the accuracy of onset detection and classification by reducing interference from other instruments.
* **Advanced Onset Detection:**
    * Explore more advanced onset detection functions (e.g., from `librosa` or other libraries like `madmom`), or refine custom `NumPy`-based algorithms.
* **Tempo Mapping for Variable Tempos:**
    * Move beyond a single global tempo. Use beat tracking algorithms (e.g., `librosa.beat.beat_track` or similar from other libraries/custom code) to get beat timings throughout the song and generate a tempo map. This would allow the MIDI to follow tempo changes in the original audio. The `midi_generator.py` would need to insert multiple tempo change meta-messages.

### 8.4. User Interface and Experience (UX) Improvements
* **GUI Upgrade:**
    * Consider migrating from Tkinter to a more feature-rich GUI framework like PyQt or Kivy for a more polished and professional look and feel, especially if adding more complex features.
* **Visual Feedback:**
    * Display the audio waveform (e.g., using `matplotlib` embedded in the GUI or a dedicated plotting library compatible with the chosen GUI framework).
    * Visualize detected onsets and classified drum hits on the waveform.
* **Parameter Presets:**
    * Allow users to save and load sets of parameters (sensitivity, classification thresholds, quantization settings) as presets for different genres or types of audio.
* **Basic MIDI Preview/Editor:**
    * A very simple piano roll style view (even text-based or basic graphical) to see the generated MIDI notes.
    * Allow for rudimentary editing (deleting notes, changing type) before saving. This is a significant undertaking.

### 8.5. Performance and Packaging
* **Optimization:**
    * Profile Python code (using `cProfile`) to identify bottlenecks in `analysis_engine.py` or feature extraction.
    * If necessary, critical sections could be rewritten in Cython or C/C++ (using libraries like `pybind11`) for speed.
* **Packaging and Distribution:**
    * Use tools like PyInstaller, cx_Freeze, or Briefcase to package the Python application into a standalone executable for easier distribution on different operating systems.

### 8.6. Broader Functionality
* **Support for More Audio Formats:**
    * Extend `audio_processor.py` to support formats like OGG, FLAC, etc., leveraging `soundfile`'s capabilities (which relies on the underlying `libsndfile` library for broad format support).
* **Batch Processing:**
    * Allow users to process multiple audio files in a queue.