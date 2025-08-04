import numpy as np
from scipy import signal
import math
from numpy.typing import NDArray # For specific NumPy array type hints

def _root_raised_cosine_filter(
    symbol_rate_hz: float,
    sample_rate_hz: float,
    rolloff: float,
    num_taps: int
) -> NDArray[np.float64]:
    """
    Generates the coefficients for a Root Raised Cosine (RRC) filter.

    Args:
        symbol_rate_hz: The symbol rate in Hz.
        sample_rate_hz: The sample rate in Hz.
        rolloff: The rolloff factor (alpha), between 0 and 1.
        num_taps: The number of filter taps. Must be an odd integer.

    Returns:
        An NDArray containing the RRC filter coefficients.

    Raises:
        ValueError: If num_taps is an even number.
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd number for a symmetric RRC filter.")
    if not (0 <= rolloff <= 1):
        raise ValueError("Rolloff factor must be between 0 and 1.")

    Ts = 1.0 / symbol_rate_hz  # Symbol period in seconds
    # Time vector for the filter, centered around 0.
    # The indices go from -(num_taps - 1) / 2 to (num_taps - 1) / 2.
    # We then divide by sample_rate_hz to get actual time values (t_i).
    t = np.arange(-(num_taps // 2), num_taps // 2 + 1) / sample_rate_hz

    h = np.zeros(num_taps, dtype=np.float64)

    for i, ti in enumerate(t):
        ti_norm = ti / Ts  # Calculate normalized time (ti / Ts)

        if np.isclose(ti, 0):
            # Special case for t = 0
            h[i] = (1 / Ts) * (1 - rolloff + (4 * rolloff / np.pi))
        elif np.isclose(abs(ti_norm), 1.0 / (4 * rolloff)):
            # Special case for t = +/- Ts / (4 * rolloff)
            # This corresponds to abs(ti_norm) = 1 / (4 * rolloff)
            h[i] = (rolloff / (np.sqrt(2) * Ts)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            )
        else:
            # General case for RRC filter impulse response
            numerator = np.sin(np.pi * ti_norm * (1 - rolloff)) + \
                        4 * rolloff * ti_norm * np.cos(np.pi * ti_norm * (1 + rolloff))
            denominator = np.pi * ti_norm * (1 - (4 * rolloff * ti_norm)**2)
            h[i] = (1 / Ts) * (numerator / denominator)

    # Normalize the filter to have unit energy (sum of squares = 1)
    # This ensures that the filter does not change the overall energy of the signal
    # when filtering white noise.
    # Avoid division by zero if h is all zeros (unlikely for valid parameters)
    if np.sum(h**2) > 1e-9: # Check for non-zero energy
        h = h / np.sqrt(np.sum(h**2))

    return h

def _create_time_array(sample_rate_hz: float, technique_length_seconds: float) -> NDArray[np.float64]:
    """Helper function to create a time array."""
    num_samples = math.floor(sample_rate_hz * technique_length_seconds)
    if num_samples <= 0:
        raise ValueError("Calculated number of samples is zero or negative. Ensure sample_rate_hz and technique_length_seconds are positive.")
    return np.linspace(0, technique_length_seconds - technique_length_seconds / num_samples, num_samples, dtype=np.float64)

# --- Waveform Generation Functions ---
# You would replace these with your actual 10 functions.
# Ensure that each function returns a NumPy array of complex numbers.

def narrowband_noise_creator(
    bandwidth_hz: float,
    sample_rate_hz: float,
    technique_length_seconds: float,
    interference_type: str = "complex"
) -> NDArray[np.complex128]:
    """
    Generates narrowband noise.
    Args:
        bandwidth_hz: The bandwidth of the noise in Hz.
        sample_rate_hz: The sample rate in Hz.
        technique_length_seconds: The duration of the noise in seconds.
        interference_type: Type of noise ('complex', 'real', or 'sinc').
    Returns:
        An NDArray containing the generated narrowband noise (complex).
    """
    if sample_rate_hz <= 2 * bandwidth_hz:
        raise ValueError(
            "sample_rate_hz must be more than 2 times greater than bandwidth_hz "
            "(Nyquist criterion for real signals, or to properly represent complex baseband)."
        )
    if bandwidth_hz < 0:
        raise ValueError("Bandwidth cannot be negative.")
    time = _create_time_array(sample_rate_hz, technique_length_seconds)
    num_samples = len(time)
    num_interference_phasors = math.floor(bandwidth_hz * num_samples / sample_rate_hz)
    if bandwidth_hz > 0 and num_interference_phasors == 0:
        num_interference_phasors = 2
    elif num_interference_phasors % 2 == 1:
        num_interference_phasors += 1
    freq_domain = np.zeros(num_samples, dtype=np.complex128)
    half_phasors = num_interference_phasors // 2
    if half_phasors > 0: # Ensure there are phasors to set
        freq_domain[0:half_phasors + 1] = 1 # Positive frequencies including DC
        freq_domain[-half_phasors:] = 1 # Negative frequencies (symmetric part)
    if interference_type == "complex":
        phases = 2 * np.pi * np.random.random(num_samples)
    elif interference_type == "real":
        random_real_noise = np.random.randn(num_samples)
        fft_of_real_noise = np.fft.fft(random_real_noise)
        phases = np.angle(fft_of_real_noise)
    elif interference_type == "sinc":
        phases = np.zeros(num_samples)
    else:
        raise ValueError("Invalid 'interference_type'. Choose 'complex', 'real', or 'sinc'.")
    freq_domain = freq_domain * np.exp(1j * phases)
    if num_interference_phasors == 0:
        power_scaler = 1.0 / num_samples
    else:
        power_scaler = np.sqrt(num_interference_phasors + 1) / num_samples
    output_signal = np.fft.ifft(freq_domain) / power_scaler
    if interference_type == "real" or interference_type == "sinc":
        output_signal = np.real(output_signal)
    else:
        output_signal = np.round(np.real(output_signal), 8) + np.round(np.imag(output_signal), 8) * 1j
    return output_signal

def rrc_modulated_noise(
    symbol_rate_hz: float,
    sample_rate_hz: float,
    rolloff: float,
    technique_length_seconds: float
) -> NDArray[np.float64]:
    """
    Generates white Gaussian noise modulated (filtered) with a Root Raised Cosine (RRC) filter.
    Args:
        symbol_rate_hz: The symbol rate in Hz.
        sample_rate_hz: The sample rate in Hz.
        rolloff: The rolloff factor (alpha), between 0 and 1.
        technique_length_seconds: The length of time in seconds for the noise signal.
    Returns:
        The RRC-filtered noise signal (real).
    """
    if sample_rate_hz < symbol_rate_hz * (1 + rolloff):
        raise ValueError(
            "sample_rate_hz must be sufficiently high relative to symbol_rate_hz and rolloff "
            "to avoid aliasing in the RRC filter."
        )
    if not (0 <= rolloff <= 1):
        raise ValueError("Rolloff factor must be between 0 and 1.")
    if symbol_rate_hz <= 0:
        raise ValueError("Symbol rate must be positive.")
    num_samples = math.floor(technique_length_seconds * sample_rate_hz)
    if num_samples <= 0:
        raise ValueError("Calculated number of samples is zero or negative. Ensure sample_rate_hz and technique_length_seconds are positive.")
    noise = np.random.randn(num_samples)
    samples_per_symbol = sample_rate_hz / symbol_rate_hz
    num_taps_raw = int(12 * samples_per_symbol) + 1
    num_taps = num_taps_raw if num_taps_raw % 2 != 0 else num_taps_raw + 1
    if num_taps < 3:
        num_taps = 3
    rrc_filter_coeffs = _root_raised_cosine_filter(symbol_rate_hz, sample_rate_hz, rolloff, num_taps)
    filtered_noise = signal.lfilter(rrc_filter_coeffs, 1.0, noise)
    return filtered_noise

def swept_noise_creator(
    sweep_hz: float,
    bandwidth_hz: float,
    sample_rate_hz: float,
    technique_length_seconds: float,
    interference_type: str = "complex"
) -> NDArray[np.complex128]:
    """
    Generates swept narrowband noise.
    Args:
        sweep_hz: The total frequency range of the sweep in Hz.
        bandwidth_hz: The bandwidth of the underlying narrowband noise in Hz.
        sample_rate_hz: The sample rate in Hz.
        technique_length_seconds: The duration of the noise in seconds.
        interference_type: Type of underlying narrowband noise ('complex', 'real', or 'sinc').
    Returns:
        An NDArray containing the generated swept noise (complex).
    """
    noise = narrowband_noise_creator(bandwidth_hz, sample_rate_hz, technique_length_seconds, interference_type)
    time = _create_time_array(sample_rate_hz, technique_length_seconds)
    freq_sweep_func = (sweep_hz / technique_length_seconds) * time - (sweep_hz / 2)
    cum_freq_sweep_func = np.cumsum(freq_sweep_func) / sample_rate_hz
    shifter = np.exp(1j * 2 * np.pi * cum_freq_sweep_func)
    swept_noise = noise * shifter
    return swept_noise

def chunk_noise_creator(
    technique_width_hz: float,
    chunks: int,
    sample_rate_hz: float,
    technique_length_seconds: float,
    interference_type: str = "complex"
) -> NDArray[np.complex128]:
    """
    Generates chunked noise, where a narrowband noise signal jumps between
    randomly ordered frequency chunks within a specified total width.
    Args:
        technique_width_hz: The total frequency span over which chunks are distributed.
        chunks: The number of distinct frequency chunks.
        sample_rate_hz: The sample rate in Hz.
        technique_length_seconds: The duration of the noise in seconds.
        interference_type: Type of underlying narrowband noise ('complex', 'real', or 'sinc').
    Returns:
        An NDArray containing the generated chunked noise (complex).
    """
    if chunks <= 0:
        raise ValueError("Number of chunks must be a positive integer.")
    if technique_width_hz <= 0:
        raise ValueError("Technique width must be positive.")
    if technique_width_hz / chunks > sample_rate_hz / 2:
        raise ValueError("Bandwidth per chunk exceeds Nyquist limit for sample rate.")
    bandwidth_hz = technique_width_hz / chunks
    noise = narrowband_noise_creator(bandwidth_hz, sample_rate_hz, technique_length_seconds, interference_type)
    time = _create_time_array(sample_rate_hz, technique_length_seconds)
    freq_chunk_centers = np.linspace(
        -1 * (technique_width_hz / 2 - bandwidth_hz / 2),
        (technique_width_hz / 2 - bandwidth_hz / 2),
        chunks
    )
    timed_chunks_raw_indices = np.floor(time / technique_length_seconds * chunks).astype(int)
    timed_chunks_raw_indices = np.clip(timed_chunks_raw_indices, 0, chunks - 1)
    chunk_order_indices = np.arange(chunks)
    np.random.shuffle(chunk_order_indices)
    timed_chunks_randomized_freq_indices = chunk_order_indices[timed_chunks_raw_indices]
    shifter = np.exp(1j * 2 * np.pi * freq_chunk_centers[timed_chunks_randomized_freq_indices] * time)
    chunked_noise = noise * shifter
    return chunked_noise

def noise_tones(
    frequencies_str: str,
    bandwidth_hz: float,
    sample_rate_hz: float,
    technique_length_seconds: float,
    interference_type: str = "complex"
) -> NDArray[np.complex128]:
    """
    Produces a sum of narrowband noise signals centered at specified frequencies.
    Args:
        frequencies_str: A space-separated string of frequencies (e.g., "100 200 300").
        bandwidth_hz: The bandwidth of each individual noise tone in Hz.
        sample_rate_hz: The sample rate in Hz.
        technique_length_seconds: The duration of the noise signal in seconds.
        interference_type: Type of underlying narrowband noise ('complex', 'real', or 'sinc').
    Returns:
        An NDArray containing the sum of noise tones (complex).
    """
    try:
        frequencies = [float(freq) for freq in frequencies_str.split()]
    except ValueError:
        raise ValueError("Invalid frequency string format. Frequencies must be space-separated numbers.")
    if not frequencies:
        raise ValueError("No frequencies provided in frequencies_str.")
    time = _create_time_array(sample_rate_hz, technique_length_seconds)
    num_samples = len(time)
    if bandwidth_hz > sample_rate_hz / 2:
        raise ValueError("Individual tone bandwidth exceeds Nyquist limit for sample rate.")
    base_noise = narrowband_noise_creator(bandwidth_hz, sample_rate_hz, technique_length_seconds, interference_type)
    noise_waves = np.zeros(num_samples, dtype=np.complex128)
    for frequency in frequencies:
        shifter = np.exp(1j * 2 * np.pi * time * frequency)
        noise_waves += base_noise * shifter
    return noise_waves

def cosine_tones(
    frequencies_str: str,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray[np.float64]:
    """
    Produces a sum of cosine arrays for a given space-separated string of frequencies.
    Args:
        frequencies_str: A space-separated string of frequencies (e.g., "100 200 300").
        sample_rate_hz: The sample rate in Hz.
        technique_length_seconds: The duration of the signal in seconds.
    Returns:
        An NDArray containing the sum of cosine waves (real).
    """
    try:
        frequencies = [float(freq) for freq in frequencies_str.split()]
    except ValueError:
        raise ValueError("Invalid frequency string format. Frequencies must be space-separated numbers.")
    if not frequencies:
        raise ValueError("No frequencies provided in frequencies_str.")
    time = _create_time_array(sample_rate_hz, technique_length_seconds)
    num_samples = len(time)
    cosine_waves = np.zeros(num_samples, dtype=np.float64)
    for frequency in frequencies:
        cosine_waves += np.cos(2 * np.pi * time * frequency)
    return cosine_waves

def phasor_tones(
    frequencies_str: str,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray[np.complex128]:
    """
    Produces a sum of phasor arrays (complex exponentials) for a given
    space-separated string of frequencies.
    Args:
        frequencies_str: A space-separated string of frequencies (e.g., "100 200 300").
        sample_rate_hz: The sample rate in Hz.
        technique_length_seconds: The duration of the signal in seconds.
    Returns:
        An NDArray containing the sum of phasor waves (complex).
    """
    try:
        frequencies = [float(freq) for freq in frequencies_str.split()]
    except ValueError:
        raise ValueError("Invalid frequency string format. Frequencies must be space-separated numbers.")
    if not frequencies:
        raise ValueError("No frequencies provided in frequencies_str.")
    time = _create_time_array(sample_rate_hz, technique_length_seconds)
    num_samples = len(time)
    phasor_waves = np.zeros(num_samples, dtype=np.complex128) # Phasors are complex
    for frequency in frequencies:
        phasor_waves += np.exp(1j * 2 * np.pi * time * frequency)
    return phasor_waves

def swept_phasors(
    sweep_hz: float,
    tones: int,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray[np.complex128]:
    """
    Generates a sum of swept phasor tones.
    Each tone sweeps a mini-bandwidth within the total sweep_hz.
    Args:
        sweep_hz: The total frequency range of the sweep in Hz.
        tones: The number of individual swept tones to sum.
        sample_rate_hz: The sample rate in Hz.
        technique_length_seconds: The duration of the signal in seconds.
    Returns:
        An NDArray containing the sum of swept phasor tones (complex).
    """
    if tones <= 0:
        raise ValueError("Number of tones must be a positive integer for swept_phasors.")
    if sweep_hz < 0:
        raise ValueError("Sweep range (sweep_hz) cannot be negative.")
    time = _create_time_array(sample_rate_hz, technique_length_seconds)
    num_samples = len(time)
    swept_tones = np.zeros(num_samples, dtype=np.complex128) # Initialized as complex
    tone_freqs = np.linspace(-sweep_hz / 2, sweep_hz / 2, tones, endpoint=False)
    mini_sweep_hz = sweep_hz / tones
    for k in range(tones):
        freq_sweep_func = (mini_sweep_hz / technique_length_seconds) * time + tone_freqs[k]
        cum_freq_sweep_func = np.cumsum(freq_sweep_func) / sample_rate_hz
        swept_tones += np.exp(1j * 2 * np.pi * cum_freq_sweep_func)
    return swept_tones

def swept_cosines(
    sweep_hz: float,
    tones: int,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray[np.float64]:
    """
    Generates a sum of swept cosine tones.
    Each tone sweeps a mini-bandwidth within the total sweep_hz.
    The output is real (cosine).
    Args:
        sweep_hz: The total frequency range of the sweep in Hz.
        tones: The number of individual swept tones to sum.
        sample_rate_hz: The sample rate in Hz.
        technique_length_seconds: The duration of the signal in seconds.
    Returns:
        An NDArray containing the sum of swept cosine tones (real).
    """
    if tones <= 0:
        raise ValueError("Number of tones must be a positive integer for swept_cosines.")
    if sweep_hz < 0:
        raise ValueError("Sweep range (sweep_hz) cannot be negative.")
    time = _create_time_array(sample_rate_hz, technique_length_seconds)
    num_samples = len(time)
    swept_tones = np.zeros(num_samples, dtype=np.float64) # Initialized as float (real output)
    tone_freqs = np.linspace(-sweep_hz / 2, sweep_hz / 2, tones, endpoint=False)
    mini_sweep_hz = sweep_hz / tones
    for k in range(tones):
        freq_sweep_func = (mini_sweep_hz / technique_length_seconds) * time + tone_freqs[k]
        cum_freq_sweep_func = np.cumsum(freq_sweep_func) / sample_rate_hz
        swept_tones += np.cos(2 * np.pi * cum_freq_sweep_func) # Adding a real value to a real array
    return swept_tones

def FM_cosine(
    sweep_range_hz: float,
    modulated_frequency: float,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray[np.complex128]:
    """
    Generates an FM-modulated complex exponential (phasor) wave.
    The instantaneous frequency of the carrier is modulated by a cosine wave.
    Args:
        sweep_range_hz: The peak frequency deviation from the carrier frequency in Hz.
        modulated_frequency: The frequency of the modulating cosine wave in Hz.
        sample_rate_hz: The sample rate in Hz.
        technique_length_seconds: The duration of the signal in seconds.
    Returns:
        An NDArray containing the FM-modulated complex exponential (complex).
    """
    if sweep_range_hz < 0:
        raise ValueError("Sweep range (peak frequency deviation) cannot be negative.")
    if modulated_frequency < 0:
        raise ValueError("Modulated frequency cannot be negative.")
    time = _create_time_array(sample_rate_hz, technique_length_seconds)
    freq_deviation_func = .5 * sweep_range_hz * np.cos(2 * np.pi * modulated_frequency * time)
    cum_phase_func = np.cumsum(freq_deviation_func) / sample_rate_hz
    FM_modulated_phasor = np.exp(1j * 2 * np.pi * cum_phase_func)
    return FM_modulated_phasor


# A dictionary to map waveform names to their functions and parameter names
# This is now stored with the functions themselves for better organization.
waveform_definitions = {
    "Narrowband Noise": {
        "func": narrowband_noise_creator,
        "params": ["bandwidth_hz", "sample_rate_hz", "technique_length_seconds", "interference_type"],
        "params2": ["Bandwidth (Hz)", "Sample Rate (Hz)", "Technique Length (seconds)", "Interference Type (real or complex)"]
    },
    "RRC Modulated Noise": {
        "func": rrc_modulated_noise,
        "params": ["symbol_rate_hz", "sample_rate_hz", "rolloff", "technique_length_seconds"],
        "params2": ["Symbol Rate (Hz)", "Sample Rate (Hz)", "rolloff (0 < r < 1)", "Technique Length (seconds)"]
    },
    "Swept Noise": {
        "func": swept_noise_creator,
        "params": ["sweep_hz", "bandwidth_hz", "sample_rate_hz", "technique_length_seconds", "interference_type"],
        "params2": ["Sweep (Hz)", "Bandwidth (Hz)", "Sample Rate (Hz)", "Technique Length (seconds)", "Interference Type (real or complex)"]
    },
    "Chunked Noise": {
        "func": chunk_noise_creator,
        "params": ["technique_width_hz", "chunks", "sample_rate_hz", "technique_length_seconds", "interference_type"],
        "params2": ["Technique Width (Hz)", "Chunks (Integer)", "Sample Rate (Hz)", "Technique Length (seconds)", "Interference Type (real or complex)"]
    },
    "Noise Tones": {
        "func": noise_tones,
        "params": ["frequencies_str", "bandwidth_hz", "sample_rate_hz", "technique_length_seconds", "interference_type"],
        "params2": ["Space Delimited Frequencies (Hz)", "Bandwidth (Hz)", "Sample Rate (Hz)", "Technique Length seconds", "Interference Type (real or complex)"]
    },
    "Cosine Tones": {
        "func": cosine_tones,
        "params": ["frequencies_str", "sample_rate_hz", "technique_length_seconds"],
        "params2": ["Space Delimited Frequencies (Hz)", "Sample Rate (Hz)", "Technique Length (seconds)"]
    },
    "Phasor Tones": {
        "func": phasor_tones,
        "params": ["frequencies_str", "sample_rate_hz", "technique_length_seconds"],
        "params2": ["Space Delimited Frequencies (Hz)", "Sample Rate (Hz)", "Technique Length (seconds)"]
    },
    "Swept Phasors": {
        "func": swept_phasors,
        "params": ["sweep_hz", "tones", "sample_rate_hz", "technique_length_seconds"],
        "params2": ["Sweep (Hz)", "Tones (Integer)", "Sample Rate (Hz)", "Technique Length (seconds)"]
    },
    "Swept Cosines": {
        "func": swept_cosines,
        "params": ["sweep_hz", "tones", "sample_rate_hz", "technique_length_seconds"],
        "params2": ["Sweep (Hz)", "Tones (Integer)", "Sample Rate (Hz)", "Technique Length (seconds)"]
    },
    "FM Cosine": {
        "func": FM_cosine,
        "params": ["sweep_range_hz", "modulated_frequency", "sample_rate_hz", "technique_length_seconds"],
        "params2": ["Sweep Range (Hz)", "Modulated Frequency (Hz)", "Sample Rate (Hz)", "Technique Length (seconds)"]
    },
}