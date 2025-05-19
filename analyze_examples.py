#!/usr/bin/env python
"""Analyze MIDI and MP3 files from examples directory to understand their structure."""

import mido
import os
import sys
import soundfile as sf
import numpy as np
from collections import Counter

def analyze_midi_file(midi_path):
    """Analyze a MIDI file and print information about its structure."""
    print(f"\nAnalyzing MIDI file: {os.path.basename(midi_path)}")
    
    try:
        mid = mido.MidiFile(midi_path)
        print(f"MIDI format: {mid.type}, tracks: {len(mid.tracks)}")
        
        # Analyze tracks
        for i, track in enumerate(mid.tracks):
            print(f"  Track {i}: {len(track)} events")
            
            # Count message types
            msg_types = Counter(msg.type for msg in track)
            print(f"  Message types: {dict(msg_types)}")
            
            # Find drum notes
            notes = Counter()
            for msg in track:
                # Count note_on messages with velocity > 0
                if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                    notes[msg.note] += 1
            
            if notes:
                print(f"  Notes used: {dict(notes)}")
                # Map common drum notes to names
                drum_map = {
                    35: "Acoustic Bass Drum",
                    36: "Bass Drum (Kick)",
                    38: "Acoustic Snare",
                    40: "Electric Snare",
                    42: "Closed Hi-hat",
                    46: "Open Hi-hat",
                    49: "Crash Cymbal",
                    51: "Ride Cymbal"
                }
                print("  Drum types:")
                for note, count in notes.items():
                    name = drum_map.get(note, f"Unknown ({note})")
                    print(f"    {name}: {count} hits")
                    
    except Exception as e:
        print(f"Error analyzing {midi_path}: {e}")

def analyze_mp3_file(mp3_path):
    """Load MP3 file and print basic information using soundfile."""
    print(f"\nAnalyzing MP3 file: {os.path.basename(mp3_path)}")
    
    try:
        # Load audio file with soundfile
        data, sr = sf.read(mp3_path)
        
        # Basic info
        duration = len(data) / sr
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Samples: {len(data)}")
        
        # Simple energy-based onset detection (for demonstration)
        # Calculate frame energy
        frame_size = int(0.02 * sr)  # 20ms frames
        hop_size = int(0.01 * sr)    # 10ms hop
        
        # Calculate energy for each frame
        energy = []
        for i in range(0, len(data) - frame_size, hop_size):
            frame = data[i:i + frame_size]
            if len(frame.shape) > 1:  # If stereo, convert to mono
                frame = frame.mean(axis=1)
            energy.append(np.sum(frame ** 2))
        energy = np.array(energy)
        
        # Calculate energy difference (for onset detection)
        energy_diff = np.diff(energy)
        energy_diff = np.append(0, energy_diff)  # Pad with 0 to match length
        
        # Normalize energy difference
        energy_diff = (energy_diff - np.mean(energy_diff)) / np.std(energy_diff)
        
        # Find peaks in energy difference (simple threshold-based detection)
        threshold = np.mean(energy_diff) + 2.0 * np.std(energy_diff)  # Adjust sensitivity as needed
        min_distance = int(0.05 * sr / hop_size)  # Minimum 50ms between onsets
        
        # Find peaks
        peaks = []
        for i in range(1, len(energy_diff) - 1):
            if (energy_diff[i] > threshold and 
                energy_diff[i] > energy_diff[i-1] and 
                energy_diff[i] > energy_diff[i+1]):
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        
        # Convert peak indices to time
        onset_times = np.array(peaks) * hop_size / sr
        
        print(f"  Detected {len(onset_times)} onsets (potential drum hits)")
        if len(onset_times) > 0:
            print(f"  First onset at {onset_times[0]:.3f}s, last at {onset_times[-1]:.3f}s")
            
    except Exception as e:
        print(f"Error analyzing {mp3_path}: {e}")

def main():
    """Main function to analyze example files."""
    examples_dir = "examples"
    midi_dir = os.path.join(examples_dir, "midi")
    mp3_dir = os.path.join(examples_dir, "mp3")
    
    # Get a list of example files with both MP3 and MIDI versions
    midi_files = {f.replace(".mid", ""): os.path.join(midi_dir, f) 
                 for f in os.listdir(midi_dir) if f.endswith(".mid")}
    mp3_files = {f.replace(".mp3", ""): os.path.join(mp3_dir, f) 
                for f in os.listdir(mp3_dir) if f.endswith(".mp3") and not f.endswith(".mp3:Zone.Identifier")}
    
    # Find files that have both MP3 and MIDI versions
    common_files = set(midi_files.keys()).intersection(mp3_files.keys())
    
    print(f"Found {len(common_files)} files with both MP3 and MIDI versions")
    
    # Analyze a few examples
    samples = list(common_files)[:3]  # Just analyze first 3 for brevity
    
    for basename in samples:
        print(f"\n{'='*50}\nAnalyzing {basename}")
        midi_path = midi_files[basename]
        mp3_path = mp3_files[basename]
        
        analyze_midi_file(midi_path)
        analyze_mp3_file(mp3_path)

if __name__ == "__main__":
    main() 