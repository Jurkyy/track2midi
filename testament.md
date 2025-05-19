# Plan: Audio to MIDI Drum Track Converter

**Project Goal:** To develop a software tool capable of analyzing audio songs, identifying drum sounds, and converting these into a standard MIDI drum track.

**Date:** May 19, 2025

---

## 1. Define the Scope and Objectives of the Tool

### 1.1. Objectives
* **Primary Objective:** Accurately convert polyphonic audio (full songs) into a MIDI file representing the percussive elements (drum track).
* **Secondary Objective:** Provide a user-friendly interface for easy operation.
* **Tertiary Objective:** Allow for some level of user customization or parameter tweaking to improve results for different audio sources.

### 1.2. Scope
* **Input:**
    * Common audio file formats (e.g., WAV, MP3, FLAC).
    * Focus on songs with clearly discernible drum sounds.
* **Output:**
    * Standard MIDI File (SMF, .mid) Type 1 (multiple tracks, with one dedicated to drums).
    * MIDI events corresponding to General MIDI (GM) standard drum mapping (e.g., Kick on MIDI note 36, Snare on 38, etc.).
* **Core Functionality:**
    * Audio loading and preprocessing.
    * Beat detection and onset detection.
    * Drum sound identification (e.g., kick, snare, hi-hat, cymbals, toms).
    * Mapping identified drum sounds and their timing to MIDI note events.
    * MIDI file export.
* **Out of Scope (for initial version):**
    * Real-time audio-to-MIDI conversion.
    * Conversion of melodic or harmonic content.
    * Advanced drum pattern generation or embellishment (focus on transcription).
    * Transcription of nuanced techniques like ghost notes, flams, or complex cymbal articulations (unless achievable with chosen methods).
    * Built-in audio editing capabilities.

### 1.3. Specific Features and Capabilities
* **V1.0 (Minimum Viable Product - MVP):**
    * Support for WAV and MP3 input.
    * Detection of basic drum kit elements: Kick, Snare, Hi-Hat (closed/open if distinguishable).
    * Export to a Type 1 MIDI file with a single drum track.
    * Basic progress indication during processing.
    * Adjustable sensitivity/threshold for detection (global or per instrument).
* **V1.X (Potential Enhancements):**
    * Detection of additional drum elements (e.g., Toms, Crash Cymbals, Ride Cymbals).
    * Basic quantization options (e.g., to 8th notes, 16th notes).
    * Velocity estimation for MIDI notes based on audio amplitude.
    * Option to choose different MIDI drum mapping presets.
    * Visual feedback of detected onsets on a waveform.

---

## 2. Research Existing Technologies and Methods

### 2.1. Audio Analysis Techniques
* **Onset Detection:**
    * Energy-based methods (sudden increases in signal energy).
    * Spectral flux (changes in the frequency spectrum).
    * Phase-based methods.
    * Machine learning-based onset detectors.
    * *Libraries:* Librosa, Aubio, Essentia, Madmom.
* **Beat Tracking & Tempo Estimation:**
    * Autocorrelation methods.
    * Comb filter methods.
    * Dynamic programming approaches.
    * *Libraries:* Librosa, Madmom.
* **Source Separation (Drum Isolation):**
    * Techniques to isolate the drum track from the rest of the mix.
    * Non-negative Matrix Factorization (NMF).
    * Harmonic/Percussive Source Separation (HPSS).
    * Deep Learning models (e.g., Spleeter, Demucs). This could be a preprocessing step.
* **Drum Sound Classification/Identification:**
    * **Template Matching:** Comparing spectral or temporal features of detected onsets against pre-defined templates of drum sounds.
    * **Feature Extraction:** MFCCs (Mel-Frequency Cepstral Coefficients), spectral centroids, zero-crossing rates, chroma features.
    * **Machine Learning Classifiers:**
        * Support Vector Machines (SVM).
        * Random Forests.
        * K-Nearest Neighbors (KNN).
        * Convolutional Neural Networks (CNNs) – for spectrogram analysis.
        * Recurrent Neural Networks (RNNs) – for sequential data.
    * *Datasets for training (if ML):* IDMT-SMT-DRUMS, ENST-Drums, MedleyDB.

### 2.2. Existing Software & Libraries
* **Commercial Software:** Ableton Live (Audio to MIDI), Logic Pro X (Drummer Track, transient detection), Melodyne (Polyphonic direct note access).
* **Open Source Tools/Libraries:**
    * **Librosa (Python):** Comprehensive audio analysis.
    * **Madmom (Python):** Strong focus on beat, downbeat, and onset tracking.
    * **Essentia (C++/Python):** Large library for audio analysis and MIR tasks.
    * **Aubio (C/Python):** Onset detection, pitch tracking, beat tracking.
    * **Spleeter (Python/TensorFlow):** Source separation (can isolate drums).
    * **PrettyMIDI (Python):** MIDI file creation and manipulation.
    * **Mido (Python):** MIDI message and file handling.

### 2.3. MIDI Standards
* **General MIDI (GM) Drum Map:** Understand the standard note assignments for drum sounds.
* **MIDI File Format:** Structure of .mid files (header, track chunks, events).

---

## 3. Determine Software Architecture and Choose Languages/Frameworks

### 3.1. Software Architecture
* **Modular Design:**
    1.  **Audio Input & Preprocessing Module:** Handles file loading, decoding, resampling, potentially source separation (e.g., isolating drums).
    2.  **Analysis Engine Module:**
        * Onset Detection Sub-module.
        * Beat & Tempo Estimation Sub-module.
        * Feature Extraction Sub-module.
        * Drum Sound Classification Sub-module.
    3.  **MIDI Generation Module:** Converts classified drum events into MIDI messages and constructs the MIDI file.
    4.  **User Interface (UI) Module:** Provides user interaction for file selection, parameter adjustment, and output.
    5.  **Configuration/Settings Module:** Manages user preferences and parameters.

* **Data Flow:**
    Audio File -> Input Module -> (Optional Source Separation) -> Analysis Engine (Onsets -> Features -> Classification) -> MIDI Generation Module -> MIDI File.

### 3.2. Programming Languages
* **Core Processing (Analysis Engine):**
    * **Python:** Recommended for rapid prototyping due to its extensive libraries for audio analysis (Librosa, Madmom, Scikit-learn, TensorFlow/PyTorch for ML) and MIDI (Mido, PrettyMIDI). Potentially slower for intensive processing.
    * **C++:** For performance-critical sections if Python proves too slow. Can be wrapped for use with Python. Libraries like Essentia are C++ based.
* **User Interface (UI):**
    * **Python GUI Frameworks:** PyQt, Kivy, Tkinter (for simpler UIs).
    * **Electron (JavaScript/HTML/CSS):** For cross-platform desktop applications with web technologies.
    * **Web-based UI (Flask/Django with JavaScript front-end):** If considering a web service model.

### 3.3. Frameworks & Libraries
* **Audio Processing:** Librosa, Madmom, (potentially Spleeter for preprocessing).
* **Machine Learning (if used):** Scikit-learn (for SVM, Random Forest), TensorFlow/PyTorch (for Deep Learning).
* **MIDI Handling:** PrettyMIDI or Mido (Python).
* **GUI:** PyQt (Python) for a robust desktop application.
* **Numerical Computation:** NumPy (Python).

**Initial Choice:** Python for core logic and UI (PyQt) for ease of development and library availability. C++ components can be integrated later if performance bottlenecks are identified.

---

## 4. Outline the Process for Audio Analysis

### 4.1. Audio Loading and Preprocessing
1.  **Load Audio File:** Use a library (e.g., Librosa) to load the audio file into a numerical array (waveform).
2.  **Resample:** Convert the audio to a consistent sample rate (e.g., 44.1 kHz) if necessary.
3.  **Mono Conversion:** Convert stereo audio to mono, as drum information is usually consistent across channels (or sum them, or take an average).
4.  **Normalization:** Normalize amplitude to a standard range (e.g., -1 to 1).
5.  **(Optional) Source Separation:**
    * If aiming for higher accuracy, use a tool like Spleeter to isolate the drum stem from the audio. This can significantly simplify subsequent steps but adds processing time and dependency.

### 4.2. Beat and Onset Detection
1.  **Tempo Estimation:** Estimate the global tempo of the song (e.g., using Librosa's `beat.tempo`). This helps in quantization later.
2.  **Onset Detection:**
    * Calculate an onset strength envelope (e.g., using `librosa.onset.onset_strength`).
    * Identify peaks in the onset strength envelope to find onset times (e.g., using `librosa.onset.onset_detect`).
    * Store the timestamps of these detected onsets.

### 4.3. Drum Sound Identification (for each detected onset)
This is the most challenging part.
1.  **Windowing:** For each detected onset, extract a small audio segment (window) around it (e.g., 50-200ms).
2.  **Feature Extraction:** From each windowed segment, extract relevant audio features:
    * Spectral Centroid
    * Spectral Bandwidth
    * Spectral Rolloff
    * MFCCs (typically first 13-20 coefficients)
    * Zero-Crossing Rate
    * Root Mean Square (RMS) Energy (for potential velocity mapping)
3.  **Classification:**
    * **Approach A (Rule-based/Template Matching - Simpler):**
        * Develop heuristics based on typical spectral characteristics of drum sounds (e.g., kicks have low spectral centroids, hi-hats have high).
        * Create average spectral templates for each target drum sound from a dataset.
        * Compare extracted features/spectra against these templates using a similarity measure (e.g., cosine similarity, Euclidean distance).
    * **Approach B (Machine Learning - More Robust):**
        * **Training (Offline):**
            * Acquire or create a dataset of labeled drum sounds (audio segments + corresponding drum type).
            * Extract features from this dataset.
            * Train a classifier (e.g., SVM, Random Forest, or a CNN if using spectrograms directly as input) to distinguish between drum types (Kick, Snare, Hi-Hat, etc.).
        * **Prediction (Online):**
            * Feed the extracted features from the input audio's onsets into the trained classifier.
            * The classifier outputs a predicted drum type for each onset.
    * **Target Drum Sounds (MVP):** Kick, Snare, Hi-Hat (Closed).
    * **Target Drum Sounds (Future):** Open Hi-Hat, Toms (High, Mid, Low), Crash Cymbal, Ride Cymbal.

---

## 5. Develop Algorithms for Mapping Detected Drum Beats to MIDI Events

1.  **MIDI Note Assignment:**
    * Create a mapping from classified drum sound labels to GM MIDI note numbers.
        * Kick: 35 or 36
        * Snare: 38 (Acoustic Snare), 40 (Electric Snare)
        * Hi-Hat Closed: 42
        * Hi-Hat Open: 46
        * Crash Cymbal: 49
        * Ride Cymbal: 51
        * Low Tom: 45
        * Mid Tom: 47
        * High Tom: 50
2.  **Timestamp to MIDI Tick Conversion:**
    * MIDI timing is based on "ticks per beat" (also known as Pulses Per Quarter Note - PPQN).
    * Define a PPQN for the output MIDI file (e.g., 480).
    * Convert the absolute timestamps (in seconds) of detected drum events into MIDI ticks relative to the song's tempo:
        `midi_tick = (timestamp_seconds / 60.0) * tempo_bpm * ppqn`
3.  **Velocity Mapping:**
    * Estimate the intensity/loudness of each drum hit. This can be derived from:
        * RMS energy of the audio segment around the onset.
        * Peak amplitude of the segment.
    * Normalize this intensity to the MIDI velocity range (0-127).
    * Apply a scaling factor or curve if needed to get musically appropriate velocities.
4.  **Note Duration:**
    * For most drum sounds, MIDI note duration is not critical as playback devices often use the "Note On" to trigger a sample that plays for its natural length.
    * A short, fixed duration can be used (e.g., equivalent to a 16th or 32nd note).
    * For cymbals or open hi-hats, if sustain is important, one might try to estimate duration, but this is complex. For MVP, fixed short duration is fine.
5.  **MIDI File Construction:**
    * Use a MIDI library (e.g., PrettyMIDI, Mido).
    * Create a new MIDI object.
    * Create an Instrument object for drums (set `is_drum=True` in PrettyMIDI, or use channel 10).
    * For each detected and classified drum event:
        * Create a MIDI `Note On` event with the mapped note number, calculated velocity, and start time (in ticks).
        * Create a corresponding MIDI `Note Off` event (or use note duration if the library handles it that way) shortly after the `Note On`.
    * Set the tempo in the MIDI file.
    * Write the MIDI object to a .mid file.
6.  **(Optional) Quantization:**
    * If desired, quantize the MIDI event start times to the nearest beat subdivision (e.g., 8th, 16th note) based on the estimated tempo. This can make the output sound "tighter" but might lose some original groove.
    * Implement selectable quantization levels.

---

## 6. Design a User Interface (UI)

### 6.1. UI Elements and Flow
* **Main Window:**
    * **File Input:**
        * "Load Audio File" button.
        * File dialog to select WAV, MP3.
        * Display area for the loaded filename.
    * **Processing Controls:**
        * "Convert to MIDI" button.
        * Progress bar and status messages (e.g., "Loading audio...", "Detecting onsets...", "Classifying drums...", "Generating MIDI...").
    * **Parameters/Settings (collapsible or separate settings dialog):**
        * **Global Sensitivity/Threshold:** Slider or input field (0-100%).
        * **(Advanced) Per-Instrument Thresholds:** Sliders for Kick, Snare, Hi-Hat (if implemented).
        * **(Advanced) Quantization Options:** Dropdown (None, 8th note, 16th note).
        * **(Advanced) Source Separation:** Checkbox ("Isolate drums before processing?").
    * **Output:**
        * "Save MIDI File" button (enabled after processing).
        * File dialog to specify output .mid filename and location.
    * **Help/About:**
        * Basic instructions, version info.

### 6.2. UI Technology
* **PyQt (Python):** Recommended for a native desktop look and feel, good integration with Python.
* **Layout:** Keep it simple and intuitive. A single main window with clear sections.

### 6.3. User Experience Considerations
* **Feedback:** Provide constant feedback during long operations.
* **Responsiveness:** Ensure the UI doesn't freeze during processing (use threading for background tasks).
* **Error Handling:** Display clear error messages (e.g., "Invalid file format," "Could not detect drums").
* **Defaults:** Provide sensible default values for all parameters.

---

## 7. Plan for Testing and Quality Assurance

### 7.1. Unit Testing
* Test individual modules and functions:
    * Audio loading for various formats and edge cases (corrupt files, silent files).
    * Onset detection algorithm against known audio with marked onsets.
    * Feature extraction functions (verify output types and ranges).
    * MIDI event creation (correct note numbers, velocities, timing).
    * Classifier accuracy (if using ML) on a hold-out test set.

### 7.2. Integration Testing
* Test the entire pipeline:
    * Load an audio file, process it, and check the output MIDI file.
    * Verify interactions between modules (e.g., onsets passed correctly to classification).

### 7.3. Accuracy Evaluation
* **Objective Metrics (using a ground truth dataset of audio + corresponding MIDI):**
    * **Precision:** (True Positives) / (True Positives + False Positives) - How many detected notes are correct?
    * **Recall:** (True Positives) / (True Positives + False Negatives) - How many actual notes were detected?
    * **F1-Score:** Harmonic mean of precision and recall.
    * Metrics can be calculated per drum instrument.
    * Timing accuracy: average deviation of detected note times from ground truth.
* **Subjective Evaluation:**
    * Listen to the original audio and the generated MIDI drum track (played with a standard GM drum soundfont).
    * Does it sound rhythmically correct?
    * Are the main drum components present?
    * Are there many false detections or missed hits?

### 7.4. Performance Testing
* Measure processing time for various audio file lengths and complexities.
* Monitor memory usage.
* Identify bottlenecks for potential optimization.

### 7.5. Usability Testing
* Have target users (musicians, producers) test the UI.
* Gather feedback on ease of use, clarity of options, and overall experience.

### 7.6. Test Data Strategy
* Create/collect a diverse dataset of audio files:
    * Different genres (rock, pop, electronic, jazz).
    * Varying production quality (clean studio recordings, noisy live recordings).
    * Simple and complex drum patterns.
    * Some files with corresponding professionally-made MIDI drum tracks for ground truth comparison.

---

## 8. Consider User Feedback and Potential Improvements for Future Updates

### 8.1. Feedback Channels
* In-app feedback form (if feasible).
* Email address for support and feedback.
* Online forum or issue tracker (e.g., GitHub Issues if open source).

### 8.2. Iterative Development
* Release an MVP and gather user feedback.
* Prioritize bug fixes and improvements based on feedback.

### 8.3. Potential Future Improvements (Post V1.0)
* **Improved Drum Sound Classification:**
    * More sophisticated ML models (e.g., CNNs, RNNs).
    * Ability to train/retrain models with user-provided samples.
    * Detection of more nuanced articulations (ghost notes, flams, cymbal chokes).
* **Advanced MIDI Editing/Refinement Features:**
    * A simple piano roll editor to manually add/remove/edit notes.
    * More granular velocity adjustment tools.
    * Humanization options (slight timing/velocity variations).
* **Real-time Processing:**
    * Audio input from a microphone or line-in, generating MIDI in real-time. This is significantly more complex.
* **Support for More Formats:**
    * Other audio formats (OGG, AAC).
    * Importing existing MIDI files for comparison or as a guide.
* **Stem Separation Integration:** Tighter integration or options for different source separation models.
* **Tempo Mapping:** Handling songs with varying tempos more effectively.
* **Parameter Presets:** Save and load settings for different types of audio or desired output styles.
* **Plugin Version:** Develop VST/AU plugin versions for DAWs.

---

This plan provides a comprehensive roadmap. The complexity, especially in the drum sound identification stage, means that development will be iterative, and achieving high accuracy will require significant effort and refinement.