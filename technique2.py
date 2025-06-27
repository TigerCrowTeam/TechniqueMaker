import math
import struct
import numpy as np
from numpy.typing import NDArray # For specific NumPy array type hints
import tkinter as tk
from tkinter import messagebox
import inspect
import json
import hashlib
import os # For checking file existence and path manipulation

# --- Provided Functions ---

def numpy_complex_to_binary_file(complex_array, filename="complex_output.raw"):
    """
    Writes a NumPy array of complex numbers to a binary file
    as interleaved 32-bit floats (real, imaginary, real, imaginary...).
    """
    complex_flat = np.asarray(complex_array).flatten()
    output_data = b''

    counter = 0
    for num in complex_flat:
        if np.iscomplexobj(num):
            real_part = num.real
            imag_part = num.imag
        else:
            real_part = float(num)
            imag_part = 0.0
        # Pack as little-endian floats
        output_data += struct.pack('<ff', real_part, imag_part)
        counter = counter + 1
        if counter % 10000 == 0:
            print(f"Processed {counter} samples for binary float output.")

    with open(filename, 'wb') as f:
        f.write(output_data)

    print(f"Complex data written to '{filename}' in binary format (interleaved floats).")

def numpy_complex_to_binary_file_int(data_array, filename):
    """
    Converts a NumPy array of complex or floating-point numbers to interleaved
    16-bit signed integers and writes them to a binary file.
    Scales the data to fit within the int16 range.
    """
    if not isinstance(data_array, np.ndarray) or (data_array.dtype != np.complex_ and not np.issubdtype(data_array.dtype, np.floating)):
        raise ValueError("Input must be a numpy array of complex or floating-point numbers.")

    # Ensure the data is treated as complex for consistent processing
    data_array_complex = np.asarray(data_array, dtype=np.complex64)

    # Scale the data to fit within int16 range (-32768 to 32767)
    # Find the maximum absolute value across real and imaginary parts
    max_val = np.max(np.concatenate((np.abs(data_array_complex.real), np.abs(data_array_complex.imag))))

    # Avoid division by zero if max_val is 0
    if max_val == 0:
        scaled_data = data_array_complex * 0
    else:
        scaled_data = data_array_complex / max_val * 32000 # Scale to leave some headroom

    with open(filename, 'wb') as f:
        counter = 0
        for value in scaled_data:
            real_rounded = int(np.round(value.real))
            imag_rounded = int(np.round(value.imag))

            # Ensure values are within int16 bounds
            real_rounded = max(-32768, min(real_rounded, 32767))
            imag_rounded = max(-32768, min(imag_rounded, 32767))

            counter = counter + 1
            if counter % 10000 == 0:
                print(f"Processed {counter} samples for binary int output.")

            # Pack the rounded real and imaginary parts as 16-bit signed big-endian integers
            f.write(struct.pack('>h', real_rounded))
            f.write(struct.pack('>h', imag_rounded))

    print(f"Data has been processed and saved to '{filename}'.")

def numpy_complex_to_sigmf_file(
    complex_array: NDArray,
    filename_prefix: str,
    sample_rate_hz: float,
    center_frequency_hz: float = 0.0,
    description: str = "Generated signal"
):
    """
    Writes a NumPy array of complex numbers to a SigMF compliant .sigmf-data and .sigmf-meta file pair.
    The data is saved as interleaved 32-bit complex floats (cf32).
    """
    data_filename = f"{filename_prefix}.sigmf-data"
    meta_filename = f"{filename_prefix}.sigmf-meta"

    # Ensure the array is complex and flatten it, explicitly converting to complex64 for cf32
    complex_flat = np.asarray(complex_array, dtype=np.complex64).flatten()

    # Write data file as interleaved float32, little-endian
    with open(data_filename, 'wb') as f:
        # For complex64, real and imag parts are float32. Use '<ff' for little-endian float float.
        for i, num in enumerate(complex_flat):
            f.write(struct.pack('<ff', num.real, num.imag))
            if (i + 1) % 10000 == 0:
                print(f"Written {i + 1} samples to SigMF data file.")

    # Calculate SHA512 hash of the data file
    sha512_hash = hashlib.sha512()
    with open(data_filename, 'rb') as f:
        # Read file in chunks to handle potentially large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha512_hash.update(byte_block)
    data_sha512 = sha512_hash.hexdigest()

    # Create metadata dictionary
    metadata = {
        "global": {
            "core:datatype": "cf32", # Complex float 32-bit
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

    # Write metadata file
    with open(meta_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"SigMF data written to '{data_filename}' and metadata to '{meta_filename}'.")


def narrowband_noise_creator(
    bandwidth_hz: float,
    sample_rate_hz: float,
    technique_length_seconds: float,
    interference_type: str = "complex"
) -> NDArray:
    """
    Generates narrowband noise.

    Args:
        bandwidth_hz: The bandwidth of the noise in Hz.
        sample_rate_hz: The sample rate in Hz.
        technique_length_seconds: The duration of the noise in seconds.
        interference_type: Type of noise ('complex', 'real', or 'sinc').

    Returns:
        An NDArray containing the generated narrowband noise.
    """
    if sample_rate_hz / 2 <= bandwidth_hz:
        raise ValueError("sample_rate_hz needs to be more than 2 times greater than bandwidth_hz")

    if technique_length_seconds * sample_rate_hz < 16:
        raise ValueError("Product of sample_rate_hz and technique_length_seconds needs to be at least 16")

    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    num_interference_phasors = math.floor(bandwidth_hz * num_samples / sample_rate_hz)

    if num_interference_phasors == 0 and bandwidth_hz > 0:
        num_interference_phasors = 2 # Ensure at least two phasors for non-zero bandwidth
    elif num_interference_phasors % 2 == 1:
        num_interference_phasors = num_interference_phasors + 1

    # Create a zero array in the frequency domain
    freq_domain = np.zeros(num_samples, dtype=complex)

    # Populate the positive and negative frequency bins
    half_phasors = num_interference_phasors // 2
    freq_domain[0:half_phasors + 1] = 1 # Positive frequencies including DC
    if half_phasors > 0: # Ensure we don't try to index -0, which is valid but less clear
        freq_domain[-half_phasors:] = 1 # Negative frequencies

    #The phases are used for a vector of unit length phasors that are
    #multiplied by the noise spectrum.

    # Generate phases based on the interference type
    if interference_type == "complex":
        # For complex noise, phases are uniformly random between 0 and 2*pi.
        phases = 2 * np.pi * np.random.random(num_samples)
    elif interference_type == "real":
        # For real noise, phases must be complex conjugate symmetric.
        # Generating random real noise and taking its FFT's phase provides the required symmetry.
        phases = np.angle(np.fft.fft(np.random.randn(num_samples)))
    elif interference_type == "sinc":
        # For a sinc pulse, all phases are zero, resulting in a real inverse FFT.
        phases = np.zeros(num_samples)
    else:
        raise ValueError("Invalid 'interference_type'. Choose 'complex', 'real', or 'sinc'.")


    #The phases are multiplied by magnitude of the frequency domain waveform.
    freq_domain = freq_domain * np.exp(1j * phases)
    # Calculate the power scaling factor. This normalizes the output power.
    # The '+1' in num_interference_phasors + 1 accounts for the DC component's contribution to power.
    power_scaler = np.sqrt(num_interference_phasors + 1) / num_samples
    # Perform the inverse FFT and apply the power scaling.
    output_signal = np.fft.ifft(freq_domain) / power_scaler
    # Round the values to mitigate floating-point errors from the IFFT,
    # ensuring cleaner real and imaginary parts.
    output_signal = np.round(np.real(output_signal), 8) + np.round(np.imag(output_signal), 8) * 1j
    return output_signal

def swept_noise_creator(
    sweep_hz: float,
    bandwidth_hz: float,
    sample_rate_hz: float,
    technique_length_seconds: float,
    interference_type: str = "complex"
) -> NDArray:
    """
    Generates swept noise.
    """
    #Calls the noise function to generate narrowband noise
    noise = narrowband_noise_creator(bandwidth_hz, sample_rate_hz, technique_length_seconds, interference_type)
    #Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / len(noise), len(noise), dtype=np.float64)
    #Function of the desired sweep
    #y=mx+b
    freq_sweep_func = sweep_hz / technique_length_seconds * time - sweep_hz / 2
    #Numerical integration of the sweep
    #Done with a cumulative sum divided by sample rate
    cum_freq_sweep_func = np.cumsum(freq_sweep_func) / sample_rate_hz
    #Creates a phasor that sweeps over the sweep within the technique time length
    shifter = np.exp(1j * 2 * np.pi * cum_freq_sweep_func)
    #Multiplies the noise by the sweeping phasor to make the noise sweep
    swept_noise = noise * shifter

    return swept_noise

def chunk_noise_creator(
    technique_width_hz: float,
    chunks: int,
    sample_rate_hz: float,
    technique_length_seconds: float,
    interference_type: str = "complex"
) -> NDArray:
    """
    Generates chunked noise.
    """
    bandwidth_hz = technique_width_hz / chunks
    #Calls the noise function to generate narrowband noise
    noise = narrowband_noise_creator(bandwidth_hz, sample_rate_hz, technique_length_seconds, interference_type)

    #Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / len(noise), len(noise), dtype=np.float64)

    freq_chunk_centers = np.linspace(-1 * (technique_width_hz / 2 - technique_width_hz / chunks / 2),(technique_width_hz / 2 - technique_width_hz / chunks / 2), chunks)

    timed_chunks = np.floor(time / technique_length_seconds * chunks).astype(int)

    # Create a NumPy array containing integers from 0 to n-1
    arr = np.arange(chunks)

    # Shuffle the array in-place
    np.random.shuffle(arr)

    timed_chunks_randomized = arr[timed_chunks]

    shifter = np.exp(1j * 2 * np.pi * freq_chunk_centers[timed_chunks_randomized] * time)

    #Multiplies the noise by the sweeping phasor to make the noise sweep
    chunked_noise = noise * shifter

    return chunked_noise

def noise_tones(
    frequencies_str: str,
    bandwidth_hz: float,
    sample_rate_hz: float,
    technique_length_seconds: float,
    interference_type: str = "complex"
) -> NDArray:
    """
    Produces a sum of phasor arrays for a given space-separated string of frequencies.
    """
    # Parse the string of frequencies into a list of floats
    try:
        frequencies = [float(freq) for freq in frequencies_str.split()]
    except ValueError:
        raise ValueError("Invalid frequency string format. Frequencies must be space-separated numbers.")

    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    
    noise = narrowband_noise_creator(bandwidth_hz, sample_rate_hz, technique_length_seconds, interference_type)
    
    # Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / num_samples, num_samples, dtype=np.float64)
    
    # Initialize an array to store the sum of phasor waves
    noise_waves = np.zeros(num_samples, dtype=complex) # Phasors are complex
    
    # Generate and sum phasor waves for each frequency
    for frequency in frequencies:
        noise_waves += noise*np.exp(1j*2 * np.pi * time * frequency)
        
    return noise_waves

def cosine_tones(
    frequencies_str: str,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    """
    Produces a sum of cosine arrays for a given space-separated string of frequencies.
    """
    # Parse the string of frequencies into a list of floats
    try:
        frequencies = [float(freq) for freq in frequencies_str.split()]
    except ValueError:
        raise ValueError("Invalid frequency string format. Frequencies must be space-separated numbers.")

    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    
    # Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / num_samples, num_samples, dtype=np.float64)
    
    # Initialize an array to store the sum of cosine waves
    cosine_waves = np.zeros(num_samples, dtype=np.float64)
    
    # Generate and sum cosine waves for each frequency
    for frequency in frequencies:
        cosine_waves += np.cos(2 * np.pi * time * frequency)
        
    return cosine_waves

def phasor_tones(
    frequencies_str: str,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    """
    Produces a sum of phasor arrays for a given space-separated string of frequencies.
    """
    # Parse the string of frequencies into a list of floats
    try:
        frequencies = [float(freq) for freq in frequencies_str.split()]
    except ValueError:
        raise ValueError("Invalid frequency string format. Frequencies must be space-separated numbers.")

    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    
    # Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / num_samples, num_samples, dtype=np.float64)
    
    # Initialize an array to store the sum of phasor waves
    phasor_waves = np.zeros(num_samples, dtype=complex) # Phasors are complex
    
    # Generate and sum phasor waves for each frequency
    for frequency in frequencies:
        phasor_waves += np.exp(1j*2 * np.pi * time * frequency)
        
    return phasor_waves


def swept_phasors(
    sweep_hz: float,
    tones: int,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    """
    Generates a sum of swept phasor tones.
    Each tone sweeps a mini-bandwidth within the total sweep_hz.
    """
    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    #Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / num_samples, num_samples, dtype=np.float64)
    swept_tones = np.zeros(num_samples, dtype=complex) # Initialized as complex
    
    # Calculate individual tone start frequencies
    # If tones is 0, this will result in issues, adding a check to prevent division by zero
    if tones <= 0:
        raise ValueError("Number of tones must be a positive integer for swept_phasors.")

    # tone_freqs define the starting frequency of each mini-sweep.
    # We want 'tones' individual sweeps covering the full 'sweep_hz' range.
    # The linspace generates 'tones' points across the sweep range.
    tone_freqs = np.linspace(-sweep_hz / 2, sweep_hz / 2, tones, endpoint=False) # Exclude endpoint to make 'tones' distinct start points
    mini_sweep_hz = sweep_hz / tones

    for k in range(tones):
        # Linear frequency sweep for each individual tone
        # Starts at tone_freqs[k] and sweeps up by mini_sweep_hz over the technique length
        freq_sweep_func = (mini_sweep_hz / technique_length_seconds) * time + tone_freqs[k]
        
        # Numerical integration of the sweep to get phase
        # Done with a cumulative sum divided by sample rate
        cum_freq_sweep_func = np.cumsum(freq_sweep_func) / sample_rate_hz
        
        # Creates a phasor that sweeps over the sweep within the technique time length
        swept_tones += np.exp(1j * 2 * np.pi * cum_freq_sweep_func)

    return swept_tones

def swept_cosines(
    sweep_hz: float,
    tones: int,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    """
    Generates a sum of swept cosine tones.
    Each tone sweeps a mini-bandwidth within the total sweep_hz.
    The output is real (cosine).
    """
    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    #Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / num_samples, num_samples, dtype=np.float64)
    swept_tones = np.zeros(num_samples, dtype=np.float64) # Initialized as float (real output)

    # Calculate individual tone start frequencies
    if tones <= 0:
        raise ValueError("Number of tones must be a positive integer for swept_cosines.")

    tone_freqs = np.linspace(-sweep_hz / 2, sweep_hz / 2, tones, endpoint=False)
    mini_sweep_hz = sweep_hz / tones

    for k in range(tones):
        # Linear frequency sweep for each individual tone
        freq_sweep_func = (mini_sweep_hz / technique_length_seconds) * time + tone_freqs[k]
        
        # Numerical integration of the sweep to get phase
        cum_freq_sweep_func = np.cumsum(freq_sweep_func) / sample_rate_hz
        
        # Creates a cosine wave that sweeps over the sweep within the technique time length
        swept_tones += np.cos(2 * np.pi * cum_freq_sweep_func) # Adding a real value to a real array

    return swept_tones

def FM_cosine(
    sweep_range_hz: float,
    modulated_frequency: float,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    """
    Generates an FM-modulated cosine wave.
    """
    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    #Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / num_samples, num_samples, dtype=np.float64)

    # Instantaneous frequency deviation based on a cosine modulator
    freq_deviation_func = .5 * sweep_range_hz * np.cos(2 * np.pi * modulated_frequency * time)
    
    # Numerical integration of the frequency deviation to get the instantaneous phase
    # This represents the phase argument of the complex exponential or cosine
    cum_phase_func = np.cumsum(freq_deviation_func) / sample_rate_hz
    
    # Creates the FM-modulated complex exponential (phasor)
    FM_modulated_cosine = np.exp(1j * 2 * np.pi * cum_phase_func)

    return FM_modulated_cosine

# --- GUI Application ---

class SignalGeneratorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Signal Generator - Select Options")
        master.geometry("500x700")
        master.option_add('*Font', 'Arial 10') # Set a default font

        self.data_format_functions = {
            "Binary Float (complex_output.raw)": numpy_complex_to_binary_file,
            "Binary Int (16-bit, big-endian)": numpy_complex_to_binary_file_int,
            "SigMF (.sigmf-data/.sigmf-meta)": numpy_complex_to_sigmf_file,
        }

        self.technique_functions = {
            "Narrowband Noise Creator": narrowband_noise_creator,
            "Swept Noise Creator": swept_noise_creator,
            "Chunk Noise Creator": chunk_noise_creator,
            "Noise Tones": noise_tones,
            "Cosine Tones": cosine_tones,
            "Phasor Tones": phasor_tones,
            "Swept Phasors": swept_phasors,
            "Swept Cosines": swept_cosines,
            "FM Cosine": FM_cosine,
        }

        self.selected_data_format = tk.StringVar(master)
        self.selected_technique = tk.StringVar(master)

        self.create_widgets()

    def create_widgets(self):
        # Data Format Selection
        data_format_frame = tk.LabelFrame(self.master, text="Select Data Format", padx=10, pady=10)
        data_format_frame.pack(padx=20, pady=10, fill="x")

        for text, func in self.data_format_functions.items():
            tk.Radiobutton(data_format_frame, text=text, variable=self.selected_data_format, value=text).pack(anchor="w")

        # Technique Selection
        technique_frame = tk.LabelFrame(self.master, text="Select Technique", padx=10, pady=10)
        technique_frame.pack(padx=20, pady=10, fill="x")

        for text, func in self.technique_functions.items():
            tk.Radiobutton(technique_frame, text=text, variable=self.selected_technique, value=text).pack(anchor="w")

        # Next Button
        tk.Button(self.master, text="Next", command=self.open_parameter_gui).pack(pady=20)

    def open_parameter_gui(self):
        data_format_name = self.selected_data_format.get()
        technique_name = self.selected_technique.get()

        if not data_format_name or not technique_name:
            messagebox.showerror("Selection Error", "Please select both a data format and a technique.")
            return

        # Hide the main window
        self.master.withdraw()

        # Open new Toplevel window for parameters
        self.param_window = tk.Toplevel(self.master)
        self.param_window.title(f"Parameters for {technique_name}")
        self.param_window.geometry("600x700") # Adjust size as needed
        self.param_window.option_add('*Font', 'Arial 10')
        self.param_window.protocol("WM_DELETE_WINDOW", self.on_param_window_close) # Handle window close button

        self.param_inputs = {} # Store Entry widgets
        self.param_vars = {} # Store StringVar/DoubleVar for parameters

        # Get the actual function object
        selected_technique_func = self.technique_functions[technique_name]
        selected_data_format_func = self.data_format_functions[data_format_name]

        # Use inspect to get function signature and parameters
        signature = inspect.signature(selected_technique_func)

        row_num = 0

        # Input fields for technique parameters
        for param_name, param in signature.parameters.items():
            # Skip 'self' if it's a method (not applicable here, but good practice)
            if param_name == 'self':
                continue

            # Special handling for interference_type for noise functions
            if param_name == "interference_type":
                tk.Label(self.param_window, text=f"{param_name.replace('_', ' ').title()}:").grid(row=row_num, column=0, padx=5, pady=5, sticky="w")
                options = ["complex", "real", "sinc"]
                var = tk.StringVar(self.param_window)
                var.set(param.default if param.default != inspect.Parameter.empty else options[0]) # Set default or first option
                option_menu = tk.OptionMenu(self.param_window, var, *options)
                option_menu.grid(row=row_num, column=1, padx=5, pady=5, sticky="ew")
                self.param_vars[param_name] = var
            # Special handling for frequencies_str
            elif param_name == "frequencies_str":
                tk.Label(self.param_window, text=f"{param_name.replace('_', ' ').title()} (space-separated):").grid(row=row_num, column=0, padx=5, pady=5, sticky="w")
                entry = tk.Entry(self.param_window, width=40)
                entry.grid(row=row_num, column=1, padx=5, pady=5, sticky="ew")
                self.param_inputs[param_name] = entry
            else:
                tk.Label(self.param_window, text=f"{param_name.replace('_', ' ').title()}:").grid(row=row_num, column=0, padx=5, pady=5, sticky="w")
                entry = tk.Entry(self.param_window, width=40)
                if param.default != inspect.Parameter.empty:
                    entry.insert(0, str(param.default)) # Set default value
                entry.grid(row=row_num, column=1, padx=5, pady=5, sticky="ew")
                self.param_inputs[param_name] = entry
            row_num += 1

        # Output Filename Input
        tk.Label(self.param_window, text="Output Filename (prefix for SigMF):").grid(row=row_num, column=0, padx=5, pady=5, sticky="w")
        self.filename_entry = tk.Entry(self.param_window, width=40)
        # Suggest a default filename based on the selected technique
        default_filename = technique_name.replace(" ", "_").lower()
        
        # Modified logic for default extensions
        if data_format_name == "SigMF (.sigmf-data/.sigmf-meta)":
            self.filename_entry.insert(0, f"{default_filename}")
        elif data_format_name == "Binary Int (16-bit, big-endian)":
            self.filename_entry.insert(0, f"{default_filename}.WAVEFORM")
        else: # Default for "Binary Float (complex_output.raw)"
            self.filename_entry.insert(0, f"{default_filename}.raw")

        self.filename_entry.grid(row=row_num, column=1, padx=5, pady=5, sticky="ew")
        row_num += 1

        # Buttons
        button_frame = tk.Frame(self.param_window)
        button_frame.grid(row=row_num, column=0, columnspan=2, pady=20)

        tk.Button(button_frame, text="Generate File", command=lambda: self.generate_file(
            selected_technique_func, selected_data_format_func, data_format_name
        )).pack(side="left", padx=10)
        tk.Button(button_frame, text="Back", command=self.go_back_to_main).pack(side="left", padx=10)

        # Configure column weights for resizing
        self.param_window.grid_columnconfigure(1, weight=1)

    def on_param_window_close(self):
        # This function is called when the user clicks the 'X' button on the parameter window
        self.param_window.destroy()
        self.master.deiconify() # Show the main window again

    def go_back_to_main(self):
        self.param_window.destroy()
        self.master.deiconify() # Show the main window again

    def generate_file(self, technique_func, data_format_func, data_format_name):
        params = {}
        for param_name, entry_widget in self.param_inputs.items():
            params[param_name] = entry_widget.get()
        
        # Add values from OptionMenu/StringVar parameters
        for param_name, var_obj in self.param_vars.items():
            params[param_name] = var_obj.get()

        filename = self.filename_entry.get()

        if not filename:
            messagebox.showerror("Input Error", "Please provide an output filename.")
            return

        # Type conversion and validation
        converted_params = {}
        signature = inspect.signature(technique_func)
        try:
            for param_name, value_str in params.items():
                param_type = signature.parameters[param_name].annotation
                if param_name == "frequencies_str":
                    converted_params[param_name] = value_str # Keep as string
                elif param_type == float:
                    converted_params[param_name] = float(value_str)
                elif param_type == int:
                    converted_params[param_name] = int(value_str)
                elif param_type == str:
                    converted_params[param_name] = value_str
                # No specific conversion needed for NDArray if it's the return type
                # For `interference_type`, it's handled by OptionMenu, already string.
                elif value_str == "":
                    # If an empty string is provided for a numeric type with a default,
                    # use the default. Otherwise, it's an error for non-optional params.
                    if signature.parameters[param_name].default != inspect.Parameter.empty:
                        converted_params[param_name] = signature.parameters[param_name].default
                    else:
                        raise ValueError(f"Parameter '{param_name}' cannot be empty.")
                else:
                    # Attempt a generic conversion for other types if not explicitly handled
                    converted_params[param_name] = value_str

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input for a parameter: {e}. Please check your values.")
            return

        try:
            # Generate the signal
            generated_signal = technique_func(**converted_params)

            # Pass `sample_rate_hz` and `center_frequency_hz` to SigMF if needed
            if data_format_name == "SigMF (.sigmf-data/.sigmf-meta)":
                sample_rate = converted_params.get('sample_rate_hz')
                if sample_rate is None:
                    messagebox.showerror("Error", "SigMF format requires 'sample_rate_hz' parameter from the technique. Please ensure it's provided.")
                    return
                # Optional: get center_frequency_hz if the technique has it or a default
                center_freq = converted_params.get('center_frequency_hz', 0.0) # Default to 0 if not present in technique
                data_format_func(generated_signal, filename, sample_rate, center_frequency_hz=center_freq)
            else:
                data_format_func(generated_signal, filename)
            
            messagebox.showinfo("Success", f"File '{filename}' generated successfully!")

        except Exception as e:
            messagebox.showerror("Generation Error", f"An error occurred during file generation: {e}")


def main():
    root = tk.Tk()
    app = SignalGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
