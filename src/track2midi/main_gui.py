"""
Main GUI module for the Audio to MIDI Drum Track Converter.

This module provides the graphical user interface for the application,
handling user interactions and coordinating the conversion process.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from typing import Optional, Dict, Any
import logging
import soundfile as sf
import numpy as np
import time

from .audio_processor import load_audio, normalize_audio
from .analysis_engine import process_audio
from .midi_generator import create_midi_file
from .midi_compare import compare_midi_files, print_comparison_results
from .config import DEFAULT_SAMPLE_RATE, DEFAULT_SENSITIVITY
from .train_model import train_model
from .ml_classifier import get_default_model_path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrumConverterApp(tk.Tk):
    """Main application window for the Audio to MIDI Drum Track Converter."""
    
    def __init__(self) -> None:
        """Initialize the main application window and its components."""
        super().__init__()

        # Window setup
        self.title("Track2MIDI - Audio to MIDI Drum Converter")
        self.geometry("600x500")
        self.resizable(True, True)

        # Instance variables
        self.audio_file_path: Optional[str] = None
        self.output_midi_path: Optional[str] = None
        self.sensitivity_var = tk.DoubleVar(value=DEFAULT_SENSITIVITY)
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False

        # Create main frame
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Create widgets
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create and arrange all GUI widgets."""
        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(3, weight=1)  # Make status frame expandable
        
        # File input section
        file_frame = ttk.LabelFrame(self.main_frame, text="Audio Input", padding="5")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)

        ttk.Button(file_frame, text="Load Audio", command=self._load_audio_file).grid(row=0, column=0, padx=5)
        self.audio_file_label = ttk.Label(file_frame, text="No file selected")
        self.audio_file_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        # Parameters section
        param_frame = ttk.LabelFrame(self.main_frame, text="Parameters", padding="5")
        param_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        param_frame.columnconfigure(1, weight=1)

        ttk.Label(param_frame, text="Sensitivity:").grid(row=0, column=0, padx=5)
        sensitivity_scale = ttk.Scale(param_frame, from_=0.1, to=5.0, 
                                    variable=self.sensitivity_var, orient=tk.HORIZONTAL)
        sensitivity_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Label(param_frame, textvariable=self.sensitivity_var).grid(row=0, column=2, padx=5, pady=5)

        # Processing section
        process_frame = ttk.LabelFrame(self.main_frame, text="Conversion", padding="5")
        process_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        process_frame.columnconfigure(0, weight=1)

        self.convert_button = ttk.Button(process_frame, text="Convert to MIDI", 
                                       command=self._start_conversion, state=tk.DISABLED)
        self.convert_button.grid(row=0, column=0, pady=5)

        self.progress_bar = ttk.Progressbar(process_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        self.status_label = ttk.Label(process_frame, text="Ready")
        self.status_label.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

        # Output section
        output_frame = ttk.LabelFrame(self.main_frame, text="Output", padding="5")
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        self.output_label = ttk.Label(output_frame, text="No MIDI file generated")
        self.output_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.save_button = ttk.Button(output_frame, text="Save MIDI As...", 
                                    command=self._save_midi_file, state=tk.DISABLED)
        self.save_button.grid(row=0, column=1, padx=5, sticky=(tk.N, tk.S))

        # Add ML model training section
        ml_frame = ttk.LabelFrame(self.main_frame, text="Machine Learning", padding="5")
        ml_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        ml_frame.columnconfigure(0, weight=1)
        ml_frame.rowconfigure(0, weight=1)

        ttk.Label(ml_frame, text="Train drum classification model using synthetic data").grid(row=0, column=0, padx=5, pady=5)

        ml_buttons_frame = ttk.Frame(ml_frame)
        ml_buttons_frame.grid(row=1, column=0, padx=5, sticky=(tk.W, tk.E))

        # Number of samples for training
        samples_frame = ttk.Frame(ml_buttons_frame)
        samples_frame.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))

        ttk.Label(samples_frame, text="Samples:").grid(row=0, column=0, padx=5)
        self.num_samples_var = tk.StringVar(value="2000")
        samples_entry = ttk.Entry(samples_frame, textvariable=self.num_samples_var, width=8)
        samples_entry.grid(row=0, column=1, padx=5)

        # Train button
        self.train_button = ttk.Button(ml_buttons_frame, text="Train Model", command=self._train_model)
        self.train_button.grid(row=0, column=2, padx=5)

        # Model status
        # Status section
        status_frame = ttk.LabelFrame(self.main_frame, text="Status", padding="5")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        # Create a frame for the text widget and buttons
        status_content = ttk.Frame(status_frame)
        status_content.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        status_content.columnconfigure(0, weight=1)
        status_content.rowconfigure(0, weight=1)

        # Create a text widget for status messages with scrollbar
        text_frame = ttk.Frame(status_content)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.status_text = tk.Text(text_frame, height=4, wrap=tk.WORD)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.status_text.insert('1.0', "Ready")
        self.status_text.config(state='disabled')

        # Add scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)

        # Add copy and compare buttons
        button_frame = ttk.Frame(status_content)
        button_frame.grid(row=0, column=1, padx=5, sticky=(tk.N, tk.S))
        
        copy_button = ttk.Button(button_frame, text="Copy Status", command=self._copy_status)
        copy_button.grid(row=0, column=0, pady=2)
        
        self.compare_button = ttk.Button(button_frame, text="Compare with Example", 
                                       command=self._compare_with_example, state='disabled')
        self.compare_button.grid(row=1, column=0, pady=2)

    def _copy_status(self) -> None:
        """Copy the current status text to clipboard."""
        self.status_text.config(state='normal')
        status_text = self.status_text.get('1.0', tk.END).strip()
        self.status_text.config(state='disabled')
        self.clipboard_clear()
        self.clipboard_append(status_text)

    def _compare_with_example(self) -> None:
        """Compare the last generated MIDI file with an example file."""
        if not self.last_generated_midi:
            messagebox.showerror("Error", "No MIDI file has been generated yet.")
            return

        # Get the base name of the generated file
        base_name = os.path.splitext(os.path.basename(self.last_generated_midi))[0]
        # Remove the date prefix if it exists (format: YYYYMMDD_HHMMSS_filename)
        if '_' in base_name:
            base_name = '_'.join(base_name.split('_')[2:])  # Take everything after the second underscore
        
        # Look for example file
        example_path = os.path.join("examples", "midi", f"{base_name}.mid")
        if not os.path.exists(example_path):
            messagebox.showerror("Error", f"No example file found at {example_path}")
            return

        # Compare the files
        results = compare_midi_files(self.last_generated_midi, example_path)
        
        # Create a new window to show comparison results
        comparison_window = tk.Toplevel(self)
        comparison_window.title("MIDI Comparison Results")
        comparison_window.geometry("600x400")
        
        # Create text widget for results
        result_text = tk.Text(comparison_window, wrap=tk.WORD)
        result_text.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Print results to the text widget
        if 'error' in results:
            result_text.insert('1.0', f"Error comparing files: {results['error']}\n")
            result_text.insert('end', f"Generated file: {results['generated_path']}\n")
            result_text.insert('end', f"Example file: {results['example_path']}\n")
        else:
            result_text.insert('1.0', "MIDI File Comparison Results:\n")
            result_text.insert('end', "-" * 50 + "\n")
            result_text.insert('end', f"Generated file events: {results['generated_event_count']}\n")
            result_text.insert('end', f"Example file events: {results['example_event_count']}\n\n")
            
            result_text.insert('end', f"Notes in generated file: {results['generated_notes']}\n")
            result_text.insert('end', f"Notes in example file: {results['example_notes']}\n\n")
            
            if results['missing_notes']:
                result_text.insert('end', f"Missing notes (in example but not in generated): {results['missing_notes']}\n\n")
            
            if results['extra_notes']:
                result_text.insert('end', f"Extra notes (in generated but not in example): {results['extra_notes']}\n\n")
            
            if results['count_differences']:
                result_text.insert('end', "Note count differences:\n")
                for note, counts in results['count_differences'].items():
                    result_text.insert('end', f"  Note {note}: generated={counts['generated']}, example={counts['example']}\n")
        
        result_text.config(state='disabled')  # Make read-only but selectable

    def _load_audio_file(self) -> None:
        """Open file dialog to select an audio file."""
        filetypes = [
            ("Audio files", "*.wav *.mp3"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if filename:
            self.audio_file_path = filename
            self.audio_file_label.config(text=os.path.basename(filename))
            self._update_status("Audio file loaded. Ready to convert.")

    def _start_conversion(self) -> None:
        """Start the conversion process in a separate thread."""
        if not self.audio_file_path:
            messagebox.showerror("Error", "Please load an audio file first.")
            return

        if self.is_processing:
            messagebox.showwarning("Warning", "Conversion already in progress.")
            return

        self.is_processing = True
        self.convert_button.config(state='disabled')
        self.compare_button.config(state='disabled')
        self.progress_bar.start()
        self._update_status("Converting...")

        # Start conversion in a separate thread
        self.processing_thread = threading.Thread(target=self._conversion_thread_target)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _conversion_thread_target(self) -> None:
        """Target function for the conversion thread."""
        try:
            # Load and normalize audio
            audio_data, sr = load_audio(self.audio_file_path)
            audio_data = normalize_audio(audio_data)
            
            # Process audio to detect drum hits
            drum_events, tempo = process_audio(
                audio_data, 
                sr, 
                sensitivity_factor=self.sensitivity_var.get()
            )
            
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(self.audio_file_path))[0]
            output_path = f"{base_name}.mid"
            
            # Create MIDI file
            saved_path = create_midi_file(
                drum_events,
                tempo,
                output_path
            )
            
            # Store the path of the generated MIDI file
            self.last_generated_midi = saved_path

            # Update UI on success
            self.after(0, self._conversion_complete, True, saved_path)
        except Exception as e:
            # Update UI on error
            self.after(0, self._conversion_complete, False, str(e))

    def _conversion_complete(self, success: bool, result: str) -> None:
        """Handle completion of the conversion process."""
        self.is_processing = False
        self.convert_button.config(state='normal')
        self.compare_button.config(state='normal' if success else 'disabled')
        self.progress_bar.stop()

        if success:
            self._update_status(f"Conversion complete. MIDI saved to: {os.path.basename(result)}")
            messagebox.showinfo("Success", f"MIDI file created successfully!\nSaved to: {result}")
        else:
            self._update_status(f"Conversion failed: {result}")
            messagebox.showerror("Error", f"Conversion failed: {result}")

    def _update_status(self, message: str) -> None:
        """Update the status text widget with a new message."""
        self.status_text.config(state='normal')
        self.status_text.delete('1.0', tk.END)
        self.status_text.insert('1.0', message)
        self.status_text.config(state='disabled')

def main() -> None:
    """Entry point for the application."""
    app = DrumConverterApp()
    app.mainloop()

if __name__ == "__main__":
    main() 