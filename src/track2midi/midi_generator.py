"""MIDI generation module for creating MIDI files from detected drum events."""

import mido
import os
from datetime import datetime
from typing import List

from .config import DEFAULT_PPQN, GM_DRUM_MAP
from .analysis_engine import DetectedDrumEvent


def ensure_results_dir() -> str:
    """Ensure the results directory exists and return its path.
    
    Returns:
        Path to the results directory
    """
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def create_midi_file(
    drum_events: List[DetectedDrumEvent],
    tempo_bpm: float,
    output_path: str,
    ppqn: int = DEFAULT_PPQN,
) -> str:
    """Create a MIDI file from detected drum events.
    
    Args:
        drum_events: List of detected drum events
        tempo_bpm: Tempo in beats per minute
        output_path: Path to save the MIDI file
        ppqn: Pulses Per Quarter Note (MIDI resolution)
        
    Returns:
        Path to the saved MIDI file
        
    Raises:
        ValueError: If no drum events are provided
        IOError: If the MIDI file cannot be written
    """
    if not drum_events:
        raise ValueError("No drum events to convert to MIDI")
    
    # If the estimated tempo is unrealistic, use a default tempo
    if tempo_bpm < 60 or tempo_bpm > 200:
        tempo_bpm = 120.0
    
    # Create a new MIDI file
    mid = mido.MidiFile(ticks_per_beat=ppqn, type=1)  # Type 1 for multiple tracks
    
    # Create a tempo track
    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    
    # Add tempo meta message
    tempo = mido.bpm2tempo(tempo_bpm)
    tempo_track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    
    # Create a drum track
    drum_track = mido.MidiTrack()
    mid.tracks.append(drum_track)
    
    # Set the drum channel (channel 9 is drums in 0-based indexing)
    drum_track.append(mido.MetaMessage('track_name', name='Drums', time=0))
    drum_track.append(mido.Message('program_change', program=0, channel=9, time=0))
    
    # Sort events by time
    drum_events = sorted(drum_events, key=lambda x: x.time_seconds)
    
    # Convert drum events to MIDI messages using a simpler approach with relative timing
    last_tick = 0  # Track the last tick position
    
    for event in drum_events:
        # Get MIDI note number for the drum type
        note = GM_DRUM_MAP.get(event.label)
        if note is None:
            print(f"Warning: Unknown drum type '{event.label}', skipping event")
            continue  # Skip unknown drum types
        
        # Convert amplitude to MIDI velocity (60-127 for better dynamics)
        velocity = int(min(127, max(60, event.amplitude * 127)))
        
        # Calculate absolute time in ticks
        current_tick = int(event.time_seconds * tempo_bpm * ppqn / 60)
        
        # Calculate delta time (always positive because events are sorted)
        delta_ticks = max(0, current_tick - last_tick)
        
        # Add note on message with delta time
        drum_track.append(mido.Message('note_on', note=note, velocity=velocity, 
                                      time=delta_ticks, channel=9))
        
        # Update the last tick position
        last_tick = current_tick
        
        # Add note off message (immediately after note on, with fixed duration)
        note_duration = ppqn // 8  # 1/32 note duration
        drum_track.append(mido.Message('note_off', note=note, velocity=0, 
                                      time=note_duration, channel=9))
        
        # Update tick position for the note off message
        last_tick += note_duration
    
    # Add end of track message
    drum_track.append(mido.MetaMessage('end_of_track', time=0))
    tempo_track.append(mido.MetaMessage('end_of_track', time=0))
    
    # Ensure results directory exists
    results_dir = ensure_results_dir()
    
    # Get filename without path and extension
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    # Create new filename with date prepended
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{date_str}_{base_name}.mid"
    
    # Create full path in results directory
    final_path = os.path.join(results_dir, new_filename)
    
    # Save the MIDI file
    try:
        mid.save(final_path)
        return final_path
    except IOError as e:
        raise IOError(f"Error saving MIDI file: {str(e)}") 