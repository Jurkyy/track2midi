"""MIDI generation module for creating MIDI files from detected drum events."""

import mido
import os
from datetime import datetime
from typing import List, Optional
import logging

from .config import DEFAULT_PPQN, GM_DRUM_MAP
from .analysis_engine import DetectedDrumEvent

logger = logging.getLogger(__name__)


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
    tempo_bpm: Optional[float],
    output_path: str,
    ppqn: int = DEFAULT_PPQN,
    default_tempo: float = 120.0
) -> str:
    """Create a MIDI file from detected drum events.
    
    Args:
        drum_events: List of detected drum events
        tempo_bpm: Tempo in beats per minute. If None or outside a reasonable range,
                   `default_tempo` will be used.
        output_path: Original intended path to save the MIDI file (filename will be modified)
        ppqn: Pulses Per Quarter Note (MIDI resolution)
        default_tempo: Default tempo to use if `tempo_bpm` is not provided or is unrealistic.
        
    Returns:
        Path to the saved MIDI file
        
    Raises:
        ValueError: If no drum events are provided
        IOError: If the MIDI file cannot be written
    """
    if not drum_events:
        raise ValueError("No drum events to convert to MIDI")
    
    current_tempo_bpm = tempo_bpm
    if current_tempo_bpm is None or current_tempo_bpm < 40 or current_tempo_bpm > 240: # Wider reasonable range
        logger.info(f"Provided tempo {tempo_bpm} BPM is None or unrealistic. Using default tempo: {default_tempo} BPM.")
        current_tempo_bpm = default_tempo
    else:
        logger.info(f"Using provided tempo: {current_tempo_bpm:.1f} BPM.")
    
    # Create a new MIDI file
    mid = mido.MidiFile(ticks_per_beat=ppqn, type=1)  # Type 1 for multiple tracks
    
    # Create a tempo track
    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    
    # Add tempo meta message
    tempo_val = mido.bpm2tempo(current_tempo_bpm)
    tempo_track.append(mido.MetaMessage('set_tempo', tempo=tempo_val))
    
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
            # Use a logger for warnings
            logger.warning(f"Unknown drum type '{event.label}' for event at {event.time_seconds:.2f}s, skipping event")
            continue  # Skip unknown drum types
        
        # Convert amplitude to MIDI velocity (60-127 for better dynamics)
        velocity = int(min(127, max(60, event.amplitude * 127)))
        
        # Calculate absolute time in ticks
        current_tick = int(event.time_seconds * current_tempo_bpm * ppqn / 60)
        
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