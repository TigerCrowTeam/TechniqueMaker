import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from numpy.typing import NDArray
import json
import hashlib
import BaseWaveforms  # Import the new waveforms.py file

# --- Your three file-writing functions ---
# These functions remain in the main GUI file as they are specific to the application's output.

def numpy_complex_to_binary_file(complex_array: NDArray, filename: str):
    """Saves a complex NumPy array to a raw binary file (cf32)."""
    np.asarray(complex_array, dtype=np.complex64).tofile(filename)
    print(f"Saved binary file: {filename}")

def numpy_complex_to_binary_file_int(complex_array: NDArray, filename: str):
    """Saves a complex NumPy array to an interleaved 16-bit signed integer binary file."""
    complex_array = np.asarray(complex_array, dtype=np.complex64)
    interleaved_data = np.stack((complex_array.real, complex_array.imag), axis=-1).flatten()
    max_val = np.max(np.abs(interleaved_data))
    if max_val == 0:
        interleaved_data = interleaved_data.astype(np.int16)
    else:
        interleaved_data = np.round(interleaved_data / max_val * 32000).astype(np.int16)
    interleaved_data.astype('>i2').tofile(filename)
    print(f"Saved integer binary file: {filename}")

def numpy_complex_to_sigmf_file(
    complex_array: NDArray,
    filename_prefix: str,
    sample_rate_hz: float,
    center_frequency_hz: float = 0.0,
    description: str = "Generated signal"
):
    """Writes a NumPy array to a SigMF compliant .sigmf-data and .sigmf-meta file pair."""
    data_filename = f"{filename_prefix}.sigmf-data"
    meta_filename = f"{filename_prefix}.sigmf-meta"
    np.asarray(complex_array, dtype=np.complex64).tofile(data_filename)
    sha512_hash = hashlib.sha512()
    with open(data_filename, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha512_hash.update(byte_block)
    data_sha512 = sha512_hash.hexdigest()
    metadata = {
        "global": {
            "core:datatype": "cf32",
            "core:sample_rate": sample_rate_hz,
            "core:version": "1.0.1",
            "core:sha512": data_sha512,
            "core:description": description
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": center_frequency_hz
            }
        ],
        "annotations": []
    }
    with open(meta_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"SigMF data written to '{data_filename}' and metadata to '{meta_filename}'.")

#--- string inferer
def infer_type_from_string(s: str):

    s = s.strip()  # Clean up any leading/trailing whitespace
    if not s:
        return s

    try:
        # Attempt to convert to an integer first.
        # This handles cases like "5"
        val = int(s)
        return val
    except ValueError:
        # If integer conversion fails, try to convert to a float.
        # This handles cases like "10.0"
        try:
            val = float(s)
            return val
        except ValueError:
            # If both fail, it's a string.
            return s

# --- The GUI Application Class ---
class WaveformApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Waveform Generator")
        self.geometry("600x400")
        self.resizable(False, False)

        # Import the waveform definitions from the separate module
        self.waveform_definitions = BaseWaveforms.waveform_definitions

        # A dictionary to map file format names to their functions and extensions
        self.file_format_definitions = {
            "Raw Complex Binary (cf32)": {
                "func": numpy_complex_to_binary_file,
                "ext": ".bin"
            },
            "Scaled Integer Binary (i16)": {
                "func": numpy_complex_to_binary_file_int,
                "ext": ".WAVEFORM"
            },
            "SigMF (.sigmf-data)": {
                "func": numpy_complex_to_sigmf_file,
                "ext": ".sigmf-data"
            }
        }

        # UI elements to hold dynamic widgets
        self.param_widgets = []
        self.param_vars = {}

        self.create_widgets()

    def create_widgets(self):
        """Builds the main GUI layout."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        # 1. Waveform Selection Section
        waveform_frame = ttk.LabelFrame(main_frame, text="1. Select Waveform", padding="10")
        waveform_frame.pack(fill="x", pady=5)
        self.waveform_combobox = ttk.Combobox(
            waveform_frame,
            values=list(self.waveform_definitions.keys()),
            state="readonly"
        )
        self.waveform_combobox.pack(fill="x")
        self.waveform_combobox.set(list(self.waveform_definitions.keys())[0])
        self.waveform_combobox.bind("<<ComboboxSelected>>", self.update_parameters)

        # 2. Dynamic Parameters Section
        self.param_frame = ttk.LabelFrame(main_frame, text="2. Waveform Parameters", padding="10")
        self.param_frame.pack(fill="x", pady=5)
        self.update_parameters()

        # 3. File Format Selection Section
        format_frame = ttk.LabelFrame(main_frame, text="3. Select Output Format", padding="10")
        format_frame.pack(fill="x", pady=5)
        self.format_combobox = ttk.Combobox(
            format_frame,
            values=list(self.file_format_definitions.keys()),
            state="readonly"
        )
        self.format_combobox.pack(fill="x")
        self.format_combobox.set(list(self.file_format_definitions.keys())[0])

        # 4. Generate Button and Status Bar
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill="x", pady=10)
        generate_button = ttk.Button(action_frame, text="Generate and Save File", command=self.generate_and_save)
        generate_button.pack(side="left", expand=True, fill="x", padx=2)

        self.status_label = ttk.Label(main_frame, text="", relief="sunken", anchor="w")
        self.status_label.pack(side="bottom", fill="x", ipady=2)

    def update_parameters(self, event=None):
        """Clears old parameters and creates new ones based on waveform selection."""
        # Clear existing widgets
        for widget in self.param_widgets:
            widget.destroy()
        self.param_widgets.clear()
        self.param_vars.clear()

        selected_waveform = self.waveform_combobox.get()
        params_list = self.waveform_definitions[selected_waveform]["params"]
        params_list2 = self.waveform_definitions[selected_waveform]["params2"]

        for param_name, param_title in zip(params_list,params_list2):
            frame = ttk.Frame(self.param_frame)
            frame.pack(fill="x", pady=2)
            self.param_widgets.append(frame)

            label = ttk.Label(frame, text=f"{param_title}:")
            label.pack(side="left", padx=5)
            self.param_widgets.append(label)

            var = tk.StringVar()
            entry = ttk.Entry(frame, textvariable=var)
            entry.pack(side="right", fill="x", expand=True)
            self.param_widgets.append(entry)

            self.param_vars[param_name] = var

    def generate_and_save(self):
        """Handles the core logic of creating the waveform and saving the file."""
        self.status_label.config(text="Generating...")
        try:
            # 1. Get selected waveform and file format
            waveform_name = self.waveform_combobox.get()
            format_name = self.format_combobox.get()

            waveform_def = self.waveform_definitions[waveform_name]
            format_def = self.file_format_definitions[format_name]

            # 2. Gather and validate parameters, handling string inputs correctly
            params = {}
            for name, var in self.param_vars.items():
                value_str = var.get()
                
                params[name] = infer_type_from_string(value_str)
                

            # 3. Ask user for a filename
            file_extension = format_def["ext"]
            filename = filedialog.asksaveasfilename(
                defaultextension=file_extension,
                filetypes=[(f"{format_name} File", file_extension)]
            )
            if not filename:
                self.status_label.config(text="Generation cancelled.")
                return

            # 4. Create the waveform
            waveform_func = waveform_def["func"]
            # The '*' operator unpacks the dictionary into keyword arguments
            complex_array = waveform_func(**params)

            # 5. Save the file
            save_func = format_def["func"]
            if "SigMF" in format_name:
                print("**************************************")
                print(params['sample_rate_hz'])
                print("**************************************")
                
                # SigMF needs a prefix, not a full filename with extension
                save_func(complex_array, filename_prefix=filename.rsplit('.', 1)[0], sample_rate_hz=params['sample_rate_hz'])
            else:
                save_func(complex_array, filename=filename)

            self.status_label.config(text=f"Success! File(s) saved to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            self.status_label.config(text="An error occurred.")
            print(f"Error: {e}")

if __name__ == "__main__":
    app = WaveformApp()
    app.mainloop()