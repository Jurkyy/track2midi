"""MIDI file comparison module for analyzing differences between MIDI files."""

import mido
import os
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MidiEvent:
    """Represents a MIDI event with its properties."""
    time: int  # Time in ticks
    note: int  # MIDI note number
    velocity: int  # MIDI velocity
    event_type: str  # 'note_on' or 'note_off'


def load_midi_events(file_path: str) -> List[MidiEvent]:
    """Load MIDI events from a file.
    
    Args:
        file_path: Path to the MIDI file
        
    Returns:
        List of MIDI events
    """
    events = []
    mid = mido.MidiFile(file_path)
    
    current_time = 0
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type in ['note_on', 'note_off']:
                events.append(MidiEvent(
                    time=current_time,
                    note=msg.note,
                    velocity=msg.velocity,
                    event_type=msg.type
                ))
    
    return events


def compare_midi_files(generated_path: str, example_path: str) -> Dict[str, any]:
    """Compare two MIDI files and return differences.
    
    Args:
        generated_path: Path to the generated MIDI file
        example_path: Path to the example MIDI file
        
    Returns:
        Dictionary containing comparison results
    """
    try:
        generated_events = load_midi_events(generated_path)
        example_events = load_midi_events(example_path)
        
        # Get unique notes in each file
        generated_notes = {event.note for event in generated_events if event.event_type == 'note_on'}
        example_notes = {event.note for event in example_events if event.event_type == 'note_on'}
        
        # Count events by type
        generated_counts = defaultdict(int)
        example_counts = defaultdict(int)
        
        for event in generated_events:
            if event.event_type == 'note_on':
                generated_counts[event.note] += 1
        
        for event in example_events:
            if event.event_type == 'note_on':
                example_counts[event.note] += 1
        
        # Calculate differences
        missing_notes = example_notes - generated_notes
        extra_notes = generated_notes - example_notes
        common_notes = generated_notes & example_notes
        
        # Compare event counts for common notes
        count_differences = {}
        for note in common_notes:
            gen_count = generated_counts[note]
            ex_count = example_counts[note]
            if gen_count != ex_count:
                count_differences[note] = {
                    'generated': gen_count,
                    'example': ex_count
                }
        
        return {
            'generated_event_count': len(generated_events),
            'example_event_count': len(example_events),
            'missing_notes': sorted(list(missing_notes)),
            'extra_notes': sorted(list(extra_notes)),
            'common_notes': sorted(list(common_notes)),
            'count_differences': count_differences,
            'generated_notes': sorted(list(generated_notes)),
            'example_notes': sorted(list(example_notes))
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'generated_path': generated_path,
            'example_path': example_path
        }


def print_comparison_results(results: Dict[str, any]) -> None:
    """Print MIDI comparison results in a readable format.
    
    Args:
        results: Dictionary containing comparison results
    """
    if 'error' in results:
        print(f"Error comparing files: {results['error']}")
        print(f"Generated file: {results['generated_path']}")
        print(f"Example file: {results['example_path']}")
        return
    
    print("\nMIDI File Comparison Results:")
    print("-" * 50)
    print(f"Generated file events: {results['generated_event_count']}")
    print(f"Example file events: {results['example_event_count']}")
    
    print("\nNotes in generated file:", results['generated_notes'])
    print("Notes in example file:", results['example_notes'])
    
    if results['missing_notes']:
        print("\nMissing notes (in example but not in generated):", results['missing_notes'])
    
    if results['extra_notes']:
        print("\nExtra notes (in generated but not in example):", results['extra_notes'])
    
    if results['count_differences']:
        print("\nNote count differences:")
        for note, counts in results['count_differences'].items():
            print(f"  Note {note}: generated={counts['generated']}, example={counts['example']}")


def main() -> None:
    """Command-line interface for MIDI file comparison."""
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m track2midi.midi_compare <generated_midi> <example_midi>")
        sys.exit(1)
    
    generated_path = sys.argv[1]
    example_path = sys.argv[2]
    
    results = compare_midi_files(generated_path, example_path)
    print_comparison_results(results)


if __name__ == "__main__":
    main() 