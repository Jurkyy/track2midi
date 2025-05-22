"""
Main GUI module for the Audio to MIDI Drum Track Converter.

This module provides the graphical user interface for the application,
handling user interactions and coordinating the conversion process.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from typing import Optional, Dict, Any, List
import logging
import soundfile as sf
import numpy as np
import time

from .audio_processor import load_audio, normalize_audio
from .analysis_engine import process_audio, is_using_ml_classification, force_load_ml_classifier
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
        self.last_generated_midi: Optional[str] = None
        self.sensitivity_var = tk.DoubleVar(value=DEFAULT_SENSITIVITY)
        self.processing_thread: Optional[threading.Thread] = None
        self.is_processing = False
        self.focus_on_snare_var = tk.BooleanVar(value=True)

        # Create main frame
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Create widgets
        self._create_widgets()
        
        # Check ML status
        self.after(100, self._check_ml_classification_status)

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

        # Output section
        output_frame = ttk.LabelFrame(self.main_frame, text="Output", padding="5")
        output_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(0, weight=1)

        self.output_label = ttk.Label(output_frame, text="No MIDI file generated")
        self.output_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Add save button
        self.save_button = ttk.Button(output_frame, text="Save MIDI As...",
                                     command=self._save_midi_file, state=tk.DISABLED)
        self.save_button.grid(row=0, column=1, padx=5)

        # Add ML model training section
        ml_frame = ttk.LabelFrame(self.main_frame, text="Machine Learning", padding="5")
        ml_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
        ml_frame.columnconfigure(0, weight=1)

        # Title with explanation
        ttk.Label(ml_frame, text="Train and use the ML-based drum classification model", 
                 font=("", 10, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        # Controls frame
        ml_buttons_frame = ttk.Frame(ml_frame)
        ml_buttons_frame.grid(row=1, column=0, padx=5, sticky=(tk.W, tk.E))
        ml_buttons_frame.columnconfigure(3, weight=1)  # Give extra space to the right

        # Create a frame for checkboxes
        checkbox_frame = ttk.Frame(ml_buttons_frame)
        checkbox_frame.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))

        # Focus on snare checkbox
        focus_on_snare_cb = ttk.Checkbutton(
            checkbox_frame,
            text="Focus on snare",
            variable=self.focus_on_snare_var
        )
        focus_on_snare_cb.grid(row=0, column=0, padx=5, sticky=tk.W)

        # Train button
        self.train_button = ttk.Button(ml_buttons_frame, text="Train Model", command=self._train_model)
        self.train_button.grid(row=0, column=2, padx=5)
        
        # Load ML Model button
        self.load_ml_button = ttk.Button(ml_buttons_frame, text="Load ML Model", command=self._load_ml_model)
        self.load_ml_button.grid(row=0, column=3, padx=5)

        # Status indicators frame
        ml_status_frame = ttk.Frame(ml_frame)
        ml_status_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        ml_status_frame.columnconfigure(1, weight=1)  # Status labels take full width

        # Model status indicators
        model_path = get_default_model_path()
        model_exists = os.path.exists(model_path)
        
        # Status label frame
        status_label_frame = ttk.Frame(ml_status_frame)
        status_label_frame.grid(row=0, column=0, sticky=tk.W, pady=3)
        
        # Model status with icon
        self.model_status_icon = ttk.Label(status_label_frame, text="●", foreground='green' if model_exists else 'red')
        self.model_status_icon.grid(row=0, column=0, padx=(0, 5))
        
        self.model_status_label = ttk.Label(status_label_frame, 
                                          text=f"Model: {'Trained' if model_exists else 'Not trained'}")
        self.model_status_label.grid(row=0, column=1, sticky=tk.W)

        # Classification method with icon
        class_status_frame = ttk.Frame(ml_status_frame)
        class_status_frame.grid(row=1, column=0, sticky=tk.W, pady=3)
        
        self.class_status_icon = ttk.Label(class_status_frame, text="●", 
                                         foreground='blue' if model_exists else 'gray')
        self.class_status_icon.grid(row=0, column=0, padx=(0, 5))
        
        self.classification_method_label = ttk.Label(class_status_frame, 
                                                   text=f"Classification: {'ML' if model_exists else 'Rule-based'}")
        self.classification_method_label.grid(row=0, column=1, sticky=tk.W)

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

        # Add buttons at the bottom
        buttons_frame = ttk.Frame(status_frame)
        buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Add copy button
        copy_button = ttk.Button(buttons_frame, text="Copy Status", command=self._copy_status)
        copy_button.pack(side=tk.LEFT, padx=5)
        
        # Add compare button
        self.compare_button = ttk.Button(buttons_frame, text="Compare with Example", 
                                       command=self._compare_with_example, state='disabled')
        self.compare_button.pack(side=tk.LEFT, padx=5)

    def _save_midi_file(self) -> None:
        """Open save dialog to choose where to save the MIDI file."""
        if not self.last_generated_midi:
            messagebox.showerror("Error", "No MIDI file generated yet")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save MIDI File",
            defaultextension=".mid",
            filetypes=[
                ("MIDI files", "*.mid"),
                ("All files", "*.*")
            ],
            initialfile=os.path.basename(self.last_generated_midi)
        )
        
        if file_path:
            # Copy the file
            try:
                with open(self.last_generated_midi, 'rb') as src, open(file_path, 'wb') as dst:
                    dst.write(src.read())
                messagebox.showinfo("Success", f"MIDI file saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")

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
            self.convert_button.config(state='normal')

    def _train_model(self) -> None:
        """Start the ML model training process in a separate thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Training Busy", "A training process is already running.")
            return

        self._update_status("Starting model training...")
        self.train_button.config(state=tk.DISABLED)
        self.load_ml_button.config(state=tk.DISABLED) # Disable load button during training

        self.processing_thread = threading.Thread(
            target=self._training_thread_target,
            # args=() # No arguments needed now
            daemon=True
        )
        self.processing_thread.start()
        self.after(100, self._check_training_progress)

    def _check_training_progress(self) -> None:
        """Check the progress of the training process."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.progress_bar.start()
            self.progress_bar.step(1)
            self.after(100, self._check_training_progress)
        else:
            self.progress_bar.stop()
            self.progress_bar.step(0)
            self.after(100, self._check_ml_classification_status)

    def _training_thread_target(self) -> None:
        """Handles the actual model training in a background thread."""
        try:
            focus_on_snare = self.focus_on_snare_var.get()

            logger.info("Starting training process in background thread...")
            
            start_time = time.time()
            
            focus_classes_value = ["snare"] if focus_on_snare else None

            accuracy, report, model_path = train_model(
                focus_on_classes=focus_classes_value,
            )
            
            end_time = time.time()
            training_duration = end_time - start_time # Added for clarity in message
            
            if accuracy is not None and report is not None:
                status_message = (
                    f"Training complete in {training_duration:.2f}s. Model saved to {model_path}\\n"
                    f"Accuracy: {accuracy:.4f}\\nClassification Report:\\n{report}"
                )
                logger.info(status_message)
                # Show success messagebox in main thread
                success_popup_message = f"Model trained successfully in {training_duration:.2f}s.\\nAccuracy: {accuracy:.4f}"
                self.after(0, lambda msg=success_popup_message: messagebox.showinfo("Training Success", msg))
                self.after(0, lambda: self._update_model_status(True)) # Update status in main thread
            else:
                status_message = f"Training failed after {training_duration:.2f}s. Check logs for details."
                logger.error(status_message)
                self.after(0, lambda: messagebox.showerror("Training Failed", status_message))
                self.after(0, lambda: self._update_model_status(False)) # Update status in main thread
            
            self._update_status(status_message)
            
        except Exception as training_exception:
            logger.error(f"Training error: {str(training_exception)}")
            # Capture exception for the lambda
            self.after(0, lambda err=training_exception: self._update_status(f"Training error: {str(err)}"))
            self.after(0, lambda err=training_exception: messagebox.showerror("Training Error", str(err)))
        
        finally:
            # Re-enable buttons and stop progress bar
            self.after(0, lambda: self.train_button.config(state='normal'))
            self.after(0, lambda: self.load_ml_button.config(state='normal'))
            self.after(0, lambda: self.progress_bar.stop())

    def _load_ml_model(self) -> None:
        """Load the ML model and activate ML classification."""
        # Show progress
        self.progress_bar.start()
        self._update_status("Loading ML model...")
        self.load_ml_button.config(state='disabled')
        
        # Start loading in a separate thread
        thread = threading.Thread(target=self._load_ml_model_thread)
        thread.daemon = True
        thread.start()
        
    def _load_ml_model_thread(self) -> None:
        """Target function for the ML model loading thread."""
        try:
            # Force load the ML model
            success = force_load_ml_classifier()
            
            if success:
                # Update UI with success
                self.after(0, lambda: self._update_status("ML model loaded successfully!"))
                self.after(0, lambda: self._check_ml_classification_status())
                self.after(0, lambda: messagebox.showinfo("Success", "ML model loaded and activated successfully"))
            else:
                # Update UI with failure
                self.after(0, lambda: self._update_status("Failed to load ML model. Check if model is trained."))
                self.after(0, lambda: messagebox.showerror("Error", "Failed to load ML model. Try training the model first."))
        
        except Exception as e:
            # Handle any exceptions
            logger.error(f"Error loading ML model: {str(e)}")
            self.after(0, lambda: self._update_status(f"Error loading ML model: {str(e)}"))
            self.after(0, lambda: messagebox.showerror("Error", f"Error loading ML model: {str(e)}"))
        
        finally:
            # Re-enable buttons and stop progress
            self.after(0, lambda: self.load_ml_button.config(state='normal'))
            self.after(0, lambda: self.progress_bar.stop())
            
    def _update_model_status(self, is_trained: bool) -> None:
        """Update the model status label."""
        status_text = f"Model: {'Trained' if is_trained else 'Not trained'}"
        self.model_status_label.config(text=status_text)
        
        # Update classification method indicator
        classification_text = f"Classification: {'ML' if is_trained else 'Rule-based'}"
        self.classification_method_label.config(
            text=classification_text
        )
        self.class_status_icon.config(
            foreground='blue' if is_trained else 'gray'
        )

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
            
            # Show statistics if ML classification was used
            if is_using_ml_classification():
                self.after(100, lambda: self._show_classification_stats(drum_events))
                
        except Exception as e:
            # Update UI on error
            self.after(0, self._conversion_complete, False, str(e))

    def _show_classification_stats(self, drum_events: List) -> None:
        """Show statistics about the classification results."""
        if not drum_events:
            return
            
        # Count occurrences of each drum type
        counts = {}
        for event in drum_events:
            counts[event.label] = counts.get(event.label, 0) + 1
            
        # Create a new window to show statistics
        stats_window = tk.Toplevel(self)
        stats_window.title("Classification Results")
        stats_window.geometry("400x300")
        
        # Create a frame with a scrollbar
        frame = ttk.Frame(stats_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        ttk.Label(frame, text="Drum Classification Results", font=("", 12, "bold")).pack(pady=5)
        ttk.Label(frame, text=f"Total drum hits detected: {len(drum_events)}").pack(pady=5)
        
        # Create a canvas for the bar graph
        canvas = tk.Canvas(frame, bg="white", height=150)
        canvas.pack(fill=tk.X, pady=10)
        
        # Sort counts by number
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Draw bar graph
        if sorted_counts:
            # Get max count for scaling
            max_count = sorted_counts[0][1]
            bar_width = 380 / len(sorted_counts)
            
            # Color map for different drum types
            colors = {
                "kick": "#3366cc",
                "snare": "#dc3912",
                "snare_electric": "#ff9900",
                "hihat_closed": "#109618", 
                "hihat_open": "#990099",
                "tom_floor": "#0099c6",
                "tom_low": "#dd4477",
                "crash": "#66aa00",
                "ride": "#b82e2e",
            }
            
            # Draw bars
            for i, (label, count) in enumerate(sorted_counts):
                # Calculate bar height
                bar_height = (count / max_count) * 120
                
                # Draw rectangle
                x0 = 10 + (i * bar_width)
                y0 = 140 - bar_height
                x1 = x0 + bar_width - 5
                y1 = 140
                
                color = colors.get(label, "#aaaaaa")
                canvas.create_rectangle(x0, y0, x1, y1, fill=color)
                
                # Draw label (vertical)
                canvas.create_text(x0 + (bar_width/2), 145, text=label, angle=90, anchor=tk.W)
                
                # Draw count
                canvas.create_text(x0 + (bar_width/2), y0 - 5, text=str(count), anchor=tk.S)
        
        # Add a table with counts
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Table headers
        ttk.Label(table_frame, text="Drum Type", font=("", 10, "bold"), width=15).grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(table_frame, text="Count", font=("", 10, "bold"), width=10).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(table_frame, text="Percentage", font=("", 10, "bold"), width=15).grid(row=0, column=2, padx=5, pady=2)
        
        # Add rows
        total = len(drum_events)
        for i, (label, count) in enumerate(sorted_counts):
            ttk.Label(table_frame, text=label).grid(row=i+1, column=0, padx=5, pady=2)
            ttk.Label(table_frame, text=str(count)).grid(row=i+1, column=1, padx=5, pady=2)
            ttk.Label(table_frame, text=f"{count/total*100:.1f}%").grid(row=i+1, column=2, padx=5, pady=2)

    def _conversion_complete(self, success: bool, result: str) -> None:
        """Handle completion of the conversion process."""
        self.is_processing = False
        self.convert_button.config(state='normal')
        self.compare_button.config(state='normal' if success else 'disabled')
        self.save_button.config(state='normal' if success else 'disabled')
        self.progress_bar.stop()
        
        # Update ML classification status
        self._check_ml_classification_status()

        if success:
            self.output_label.config(text=os.path.basename(result))
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

    def _check_ml_classification_status(self) -> None:
        """Check if ML classification is active and update UI accordingly."""
        # Update the classification method indicator
        is_ml_active = is_using_ml_classification()
        classification_text = f"Classification: {'ML' if is_ml_active else 'Rule-based'}"
        self.classification_method_label.config(
            text=classification_text
        )
        self.class_status_icon.config(
            foreground='blue' if is_ml_active else 'gray'
        )
        
        # Also update the model status
        model_path = get_default_model_path()
        model_exists = os.path.exists(model_path)
        self.model_status_label.config(
            text=f"Model: {'Trained' if model_exists else 'Not trained'}"
        )
        self.model_status_icon.config(
            foreground='green' if model_exists else 'red'
        )

def main() -> None:
    """Entry point for the application."""
    app = DrumConverterApp()
    app.mainloop()

if __name__ == "__main__":
    main() 