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

def numpy_complex_to_binary_file(complex_array, filename="complex_output.raw"):
    """
    Converts a NumPy array (real or complex) to a binary file suitable for
    GNU Radio's File Source block with complex float output. This version
    saves files with a '.raw' extension.

    Args:
        complex_array (numpy.ndarray): A NumPy array of real or complex numbers.
        filename (str, optional): The name of the output binary file.
                                  Defaults to "complex_output.raw".
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
        output_data += struct.pack('<ff', real_part, imag_part)
        counter = counter + 1
        if counter % 10000 == 0:
            print(f"Processed {counter} samples for binary float output.")

    with open(filename, 'wb') as f:
        f.write(output_data)

    print(f"Complex data written to '{filename}' in binary format (interleaved floats).")

def numpy_complex_to_binary_file_int(data_array, filename):
    """
    Converts a numpy array to a binary file, handling both complex and float arrays.
    This version saves files with a '.WAVEFORM' extension and stores data as int16.

    If the input array contains complex numbers, it rounds both real and
    imaginary components. If the input array contains floating-point numbers,
    it treats them as complex numbers with a zero imaginary component and
    rounds the real part. The rounded real and (zero) imaginary components
    are then stored as 16-bit signed big-endian integers in the specified
    binary file.

    Args:
        data_array (numpy.ndarray): A numpy array of complex or floating-point numbers.
        filename (str, optional): The name of the binary file to create.
            Defaults to "complex_data.WAVEFORM".
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
    Converts a NumPy array of complex floats to a SigMF file (data and metadata).

    Args:
        complex_array (NDArray): A NumPy array of complex numbers.
        filename_prefix (str): The prefix for the output SigMF files (e.g., "my_signal").
                               This will create "my_signal.sigmf-data" and "my_signal.sigmf-meta".
        sample_rate_hz (float): The sample rate of the signal in Hertz.
        center_frequency_hz (float, optional): The center frequency of the signal in Hertz. Defaults to 0.0.
        description (str, optional): A description of the generated signal.
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
    Generates a NumPy array of narrowband noise.

    The noise can be either complex white Gaussian, real white Gaussian,
    or a real sinc pulse. The bandwidth is adjusted slightly to ensure
    an even number of active frequency bins for proper symmetric
    representation in the frequency domain. When complex or real noise
    is created, the bandwidth is shifted down to ensure the math works out.

    Args:
        bandwidth_hz (float): The desired bandwidth of the noise in Hertz.
        sample_rate_hz (float): The sampling rate in Hertz.
        technique_length_seconds (float): The desired length of the noise in seconds.
        interference_type (str, optional): The type of noise to generate.
                                           "complex": Generates complex white Gaussian noise.
                                           "real": Generates real white Gaussian noise.
                                           "sinc": Generates a sinc pulse (all zero phases).
                                           Defaults to "complex".

    Returns:
        np.ndarray: A NumPy array containing the generated narrowband noise.
                    The data type will be complex if 'complex' noise is selected,
                    and complex (with zero imaginary parts) if 'real' or 'sinc'
                    noise is selected, due to the nature of IFFT.
    """

    if sample_rate_hz / 2 <= bandwidth_hz:
        raise ValueError("sample_rate_hz needs to be more than 2 times greater than bandwidth_hz")

    if technique_length_seconds * sample_rate_hz < 16:
        raise ValueError("Product of sample_rate_hz and technique_length_seconds needs to be at least 16")

    """
    Calculation of the number of samples in the waveform based on sample rate
    and technique length.  Number of samples is rounded down if product
    of sample rate and technique length is not an integer.
    """
    num_samples = math.floor(sample_rate_hz * technique_length_seconds)

    """
    The number of noise phasors is equal to the bandwidth of the noise
    divided by the frequency increment which is equal to the sample rate
    divided by the number of samples.  This value is rounded down.
    """
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


def swept_phasors(
    sweep_hz: float,
    tones: int,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    #Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / num_samples, num_samples, dtype=np.float64)
    swept_tones = np.zeros(num_samples, dtype=complex) # Initialized as complex
    tone_freqs = np.linspace(-sweep_hz / 2, sweep_hz / 2, tones + 1)
    mini_sweep_hz = sweep_hz / tones

    for k in range(tones):
        freq_sweep_func = mini_sweep_hz / technique_length_seconds * time + tone_freqs[k]
        #Numerical integration of the sweep
        #Done with a cumulative sum divided by sample rate
        cum_freq_sweep_func = np.cumsum(freq_sweep_func) / sample_rate_hz
        #Creates a phasor that sweeps over the sweep within the technique time length
        swept_tones = swept_tones + np.exp(1j * 2 * np.pi * cum_freq_sweep_func)

    return swept_tones

def swept_cosines(
    sweep_hz: float,
    tones: int,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    #Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / num_samples, num_samples, dtype=np.float64)
    swept_tones = np.zeros(num_samples, dtype=complex) # Initialized as complex (imaginary part will be zero)
    tone_freqs = np.linspace(-sweep_hz / 2, sweep_hz / 2, tones + 1)
    mini_sweep_hz = sweep_hz / tones

    for k in range(tones):
        freq_sweep_func = mini_sweep_hz / technique_length_seconds * time + tone_freqs[k]
        #Numerical integration of the sweep
        #Done with a cumulative sum divided by sample rate
        cum_freq_sweep_func = np.cumsum(freq_sweep_func) / sample_rate_hz
        #Creates a phasor that sweeps over the sweep within the technique time length
        swept_tones = swept_tones + np.cos(2 * np.pi * cum_freq_sweep_func) # Adding a real value to a complex array

    return swept_tones

def FM_cosine(
    sweep_range_hz: float,
    modulated_frequency: float,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    #Creates a time array for the technique length
    time = np.linspace(0, technique_length_seconds - technique_length_seconds / num_samples, num_samples, dtype=np.float64)

    freq_sweep_func = .5 * sweep_range_hz * np.cos(2 * np.pi * modulated_frequency * time)
    #Numerical integration of the sweep
    #Done with a cumulative sum divided by sample rate
    cum_freq_sweep_func = np.cumsum(freq_sweep_func) / sample_rate_hz
    #Creates a phasor that sweeps over the sweep within the technique time length
    FM_modulated_cosine = np.exp(1j * 2 * np.pi * cum_freq_sweep_func)

    return FM_modulated_cosine


# --- GUI Application ---

class NumPyFileGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("NumPy Array Generator")
        master.geometry("600x600") # Set initial window size

        # Configure columns to expand
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(0, weight=1)
        master.grid_rowconfigure(1, weight=1)
        master.grid_rowconfigure(2, weight=1) # Added for more rows if needed

        self.formatter_func_name = tk.StringVar(master)
        self.generator_func_name = tk.StringVar(master)
        self.parameter_entries = {} # To store Tkinter Entry widgets for parameters
        self.filename_var = tk.StringVar(master)
        self.filename_var.set("output_signal") # Default filename prefix, without extension

        self.all_frames = [] # To keep track of frames for easy destruction

        self.create_first_gui()

    def clear_frames(self):
        """Clears all dynamic frames from the window."""
        for frame in self.all_frames:
            frame.destroy()
        self.all_frames.clear()

    def create_first_gui(self):
        """Creates the initial GUI for selecting formatter and generator functions."""
        self.clear_frames()

        # Frame for formatter selection
        formatter_frame = tk.LabelFrame(self.master, text="Select Array Formatting Function", padx=10, pady=10)
        formatter_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        formatter_frame.grid_columnconfigure(0, weight=1)
        self.all_frames.append(formatter_frame)

        tk.Radiobutton(formatter_frame, text="Binary File (complex float, .raw)",
                        variable=self.formatter_func_name,
                        value="numpy_complex_to_binary_file").pack(anchor="w", pady=2)
        tk.Radiobutton(formatter_frame, text="MXG file (complex int16 BigEndian, .WAVEFORM)",
                        variable=self.formatter_func_name,
                        value="numpy_complex_to_binary_file_int").pack(anchor="w", pady=2)
        tk.Radiobutton(formatter_frame, text="SigMF File (cf32, .sigmf-data/.sigmf-meta)",
                        variable=self.formatter_func_name,
                        value="numpy_complex_to_sigmf_file").pack(anchor="w", pady=2)


        # Set a default selection
        self.formatter_func_name.set("numpy_complex_to_binary_file")

        # Frame for generator selection
        generator_frame = tk.LabelFrame(self.master, text="Select Array Generation Function", padx=10, pady=10)
        generator_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        generator_frame.grid_columnconfigure(0, weight=1)
        self.all_frames.append(generator_frame)

        tk.Radiobutton(generator_frame, text="narrowband_noise_creator",
                        variable=self.generator_func_name,
                        value="narrowband_noise_creator").pack(anchor="w", pady=2)
        tk.Radiobutton(generator_frame, text="swept_noise_creator",
                        variable=self.generator_func_name,
                        value="swept_noise_creator").pack(anchor="w", pady=2)
        tk.Radiobutton(generator_frame, text="chunk_noise_creator",
                        variable=self.generator_func_name,
                        value="chunk_noise_creator").pack(anchor="w", pady=2)
        tk.Radiobutton(generator_frame, text="swept_phasors",
                        variable=self.generator_func_name,
                        value="swept_phasors").pack(anchor="w", pady=2)
        tk.Radiobutton(generator_frame, text="swept_cosines",
                        variable=self.generator_func_name,
                        value="swept_cosines").pack(anchor="w", pady=2)
        tk.Radiobutton(generator_frame, text="FM_cosine",
                        variable=self.generator_func_name,
                        value="FM_cosine").pack(anchor="w", pady=2)

        # Set a default selection
        self.generator_func_name.set("narrowband_noise_creator")

        # Next button
        next_button = tk.Button(self.master, text="Next", command=self.show_parameters_gui,
                                 bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), relief="raised", bd=3)
        next_button.grid(row=1, column=0, columnspan=2, pady=20, ipadx=20, ipady=10)
        self.all_frames.append(next_button) # Add button to all_frames for clearing

    def show_parameters_gui(self):
        """Creates the second GUI for entering function parameters and filename."""
        self.clear_frames()

        selected_generator_name = self.generator_func_name.get()
        generator_func = GENERATOR_FUNCTIONS[selected_generator_name]
        sig = inspect.signature(generator_func)

        param_frame = tk.LabelFrame(self.master, text=f"Parameters for {selected_generator_name}", padx=10, pady=10)
        param_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        param_frame.grid_columnconfigure(1, weight=1)
        self.all_frames.append(param_frame)

        self.parameter_entries.clear() # Clear previous entries

        row_num = 0
        for name, param in sig.parameters.items():
            if name == 'return': # Skip return type hint
                continue

            # Skip the 'interference_type' parameter for functions that don't use it
            if name == 'interference_type' and selected_generator_name not in ["narrowband_noise_creator", "swept_noise_creator", "chunk_noise_creator"]:
                continue

            tk.Label(param_frame, text=f"{name}:", anchor="w").grid(row=row_num, column=0, padx=5, pady=2, sticky="w")
            entry_var = tk.StringVar(param_frame)

            # Set default values from function signature
            if param.default is not inspect.Parameter.empty:
                entry_var.set(str(param.default))

            # Special handling for 'interference_type' as a dropdown
            if name == "interference_type":
                option_menu = tk.OptionMenu(param_frame, entry_var, "complex", "real", "sinc")
                option_menu.grid(row=row_num, column=1, padx=5, pady=2, sticky="ew")
            else:
                entry = tk.Entry(param_frame, textvariable=entry_var)
                entry.grid(row=row_num, column=1, padx=5, pady=2, sticky="ew")
            self.parameter_entries[name] = entry_var # Store StringVar for retrieval
            row_num += 1

        # Filename input
        filename_frame = tk.Frame(self.master, padx=10, pady=10)
        filename_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        filename_frame.grid_columnconfigure(1, weight=1)
        self.all_frames.append(filename_frame)

        tk.Label(filename_frame, text="Output Filename (no extension needed):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        tk.Entry(filename_frame, textvariable=self.filename_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        # Buttons for this stage
        button_frame = tk.Frame(self.master, padx=10, pady=10)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        self.all_frames.append(button_frame)

        back_button = tk.Button(button_frame, text="Back", command=self.create_first_gui,
                                 bg="#FFA500", fg="white", font=("Arial", 12, "bold"), relief="raised", bd=3)
        back_button.pack(side="left", padx=10, ipadx=10, ipady=5)

        generate_button = tk.Button(button_frame, text="Generate File", command=self.execute_generation,
                                     bg="#007BFF", fg="white", font=("Arial", 12, "bold"), relief="raised", bd=3)
        generate_button.pack(side="right", padx=10, ipadx=10, ipady=5)


    def execute_generation(self):
        """
        Retrieves parameters, calls selected functions, and handles file output.
        Displays messages using custom message box.
        """
        selected_formatter_name = self.formatter_func_name.get()
        selected_generator_name = self.generator_func_name.get()

        formatter_func = FORMATTER_FUNCTIONS[selected_formatter_name]
        generator_func = GENERATOR_FUNCTIONS[selected_generator_name]

        params = {}
        try:
            sig = inspect.signature(generator_func)
            for name, param in sig.parameters.items():
                if name == 'return':
                    continue

                # Skip parameters not used by the current function (e.g., interference_type for non-noise)
                if name == 'interference_type' and selected_generator_name not in ["narrowband_noise_creator", "swept_noise_creator", "chunk_noise_creator"]:
                    continue

                value_str = self.parameter_entries[name].get()
                # Attempt to convert to the expected type
                if param.annotation is float:
                    params[name] = float(value_str)
                elif param.annotation is int:
                    params[name] = int(value_str)
                elif param.annotation is str:
                    params[name] = value_str
                else: # Default to string if type hint not found or unknown
                    params[name] = value_str

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input for a parameter: {e}. Please ensure all numerical inputs are valid numbers.")
            return
        except KeyError as e:
            messagebox.showerror("Input Error", f"Missing parameter: {e}. This should not happen. Please report this issue.")
            return

        output_filename_base = self.filename_var.get()
        if not output_filename_base:
            messagebox.showerror("Input Error", "Output filename cannot be empty.")
            return

        # Remove any existing extension to ensure correct new one is applied
        output_filename_base = os.path.splitext(output_filename_base)[0]

        final_output_filename = ""

        try:
            # Generate the NumPy array
            generated_array = generator_func(**params)

            # Determine arguments for the formatter function based on selected type
            if selected_formatter_name == "numpy_complex_to_sigmf_file":
                sample_rate_for_sigmf = params.get('sample_rate_hz')
                if sample_rate_for_sigmf is None:
                    messagebox.showerror("Error", "Sample rate (sample_rate_hz) is required for SigMF output but not found in generator parameters. Please choose a generator that provides sample_rate_hz.")
                    return
                # Use a default center frequency or get it from generator params if available
                center_frequency_for_sigmf = params.get('center_frequency_hz', 0.0)
                formatter_func(generated_array, output_filename_base, sample_rate_for_sigmf, center_frequency_for_sigmf)
                final_output_filename = f"{output_filename_base}.sigmf-data and {output_filename_base}.sigmf-meta"
                messagebox.showinfo("Success", f"Successfully generated '{final_output_filename}' using {selected_generator_name}.")
            else:
                if selected_formatter_name == "numpy_complex_to_binary_file":
                    final_output_filename = f"{output_filename_base}.raw"
                elif selected_formatter_name == "numpy_complex_to_binary_file_int":
                    final_output_filename = f"{output_filename_base}.WAVEFORM"
                else:
                    final_output_filename = output_filename_base # Fallback, should not happen

                formatter_func(generated_array, final_output_filename)
                messagebox.showinfo("Success", f"Successfully generated '{final_output_filename}' using {selected_generator_name} and formatted with {selected_formatter_name}.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during generation or file writing: {e}")

# Mapping of function names to actual functions (global for easy access)
FORMATTER_FUNCTIONS = {
    "numpy_complex_to_binary_file": numpy_complex_to_binary_file,
    "numpy_complex_to_binary_file_int": numpy_complex_to_binary_file_int,
    "numpy_complex_to_sigmf_file": numpy_complex_to_sigmf_file,
}

GENERATOR_FUNCTIONS = {
    "narrowband_noise_creator": narrowband_noise_creator,
    "swept_noise_creator": swept_noise_creator,
    "chunk_noise_creator": chunk_noise_creator,
    "swept_phasors": swept_phasors,
    "swept_cosines": swept_cosines,
    "FM_cosine": FM_cosine,
}

# Main execution block
if __name__ == "__main__":
    root = tk.Tk()
    app = NumPyFileGeneratorApp(root)
    root.mainloop()