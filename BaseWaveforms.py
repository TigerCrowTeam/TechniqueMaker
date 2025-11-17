import numpy as np
from scipy import signal
import math
from numpy.typing import NDArray # For specific NumPy array type hints
from typing import Literal

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
    interference_type: Literal["complex", "real","sinc"] = "complex"
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



def songMaker(
    songName: Literal["Air Force Song", "Anchors Away","Marine Hymn","Baby Shark"],
    bandwidth_hz: float,
    sample_rate_hz: float,
) -> NDArray[np.complex128]:

    def noteMaker(noteLengthInBeats,noteNumber,bpm,sampsPerSec,bandwidth):

        #BaseFreq = D sharp
        BaseFreq=155.563

        #Note Number 0 = was originaly D sharp but is now a rest
        #Note Number 1 = E or the open 6th string of a guitar
        X=np.arange(0,7,1/12)
        B=2**X*BaseFreq

        secPerBeats=60/bpm
        sampsPerPeriod=int(1/B[noteNumber]*sampsPerSec)
        sampsPerNote=sampsPerSec*secPerBeats*noteLengthInBeats
        cyclesPerNote=round(sampsPerNote/sampsPerPeriod*.9)
        sampsWithSilence=round(sampsPerNote-cyclesPerNote*sampsPerPeriod)

        E=round(bandwidth/(sampsPerSec/sampsPerPeriod)/2)
        #Number of frequency components has to be 2 or greater otherwise
        #the function won't make a note when the frequency is too high
        E=np.max([E,2])
        
        F=np.zeros(sampsPerPeriod)
        F[0:E]=np.ones(E)
        G=np.real(np.fft.fft(F))
        G=G-np.mean(G)
        G=G/np.max(G)

        J=np.roll(G,round(sampsPerPeriod/2))
        G=G-J
        G=G/np.max(np.absolute(G))
        
        I=np.array([])

        for k in range(cyclesPerNote):
            I=np.concatenate([I,G])

        E=round(bandwidth/(sampsPerSec/sampsWithSilence)/2)
        F=np.zeros(sampsWithSilence)
        F[0:E]=np.ones(E)
        G=np.real(np.fft.fft(F))
        G=G-np.mean(G)
        G=G/np.max(G)

        I=np.concatenate([I,G])

        if noteNumber==0:
            I=I*0

        return(I)

    if songName=="Air Force Song":
        BPMval=300
        A=[2,1,1,1,1,2,1,2,1,2,1,6,1,1,1,3,3,3,3,2,1,6,1,1,1,12,2,1,6,1,1,1,3,3,3,3,2,1,6,1,1,1,12,2,1,6,1,1,1,3,3,3,3,2,1,6,1,1,1,5,1,3,5,1,4,1,1,5,1,3,3,1,1,1,2,1,2,1,3,8,10]
        B=[13,13,13,13,13,13,0,13,0,10,13,13,11,10,8,10,11,12,13,15,18,18,20,18,15,13,10,13,13,11,10,8,10,11,12,13,17,20,20,18,17,15,13,10,13,13,11,10,8,10,11,12,13,15,18,18,15,17,18,17,0,10,18,18,19,19,19,20,20,21,21,22,20,18,22,18,22,18,20,18,0]
    elif songName=="Anchors Away":
        BPMval=150
        A=[2,1,1,1.5,.5,2,2,1,1,4,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1.5,.5,2,2,1,1,4,2,1,1,1,1,1,1,.75,.25,.5,.5,.75,.25,.5,.5,4,4]
        B=[16,20,23,25,20,25,28,30,23,28,25,28,25,23,25,27,28,22,25,30,28,27,23,21,18,16,20,23,25,20,25,28,30,23,28,25,28,25,23,25,27,28,32,23,22,23,30,23,22,23,28,0]
    elif songName=="Marine Hymn":
        BPMval=240
        A=[1,1,2,2,2,2,3,1,2,1,1,2,2,1,3,5,1,1,1,2,2,2,2,3,1,2,1,1,2,2,1,3,5,1,1.5,.5,2,2,2,2,3,1,2,1.5,.5,2,2,1,3,5,1,1,1,2,2,2,2,3,1,2,1,1,2,2,2,2,5,3]
        B=[17,18,20,20,20,20,20,25,20,17,18,20,20,18,15,13,0,17,18,20,20,20,20,20,25,20,17,18,20,20,18,15,13,0,25,24,22,18,22,25,20,17,20,25,24,22,18,22,25,20,0,13,17,20,20,20,20,20,25,20,17,18,20,20,18,15,13,0]
    elif songName=="Army Song":
        BPMval=240
        A=[2,1,1,2,1,1,2,1,1,1.5,.5,1,1,2,1,1,1,2,1,1,2,1,4,2,1,1,2,1,1,2,1,1,1.5,.5,1,1,2,1,1,1,2,1,1,2,1,4,2,1,1,2,2,2,1,1,1,1,1,1,4,1,2,1,1,1,1,1,4,2,1,1,2,2,4,1.5,.5,1,1,2,1,1,1,2,1,1,2,1,4,2,1,1,1,2,1,1,2,1,4]
        B=[0,14,11,14,14,11,14,14,11,14,16,14,11,14,11,12,14,12,9,14,12,9,7,0,14,11,14,14,11,14,14,11,14,16,14,11,14,11,12,14,12,9,14,12,9,7,0,14,14,19,19,14,14,14,16,18,19,16,14,19,19,18,16,18,19,16,21,0,14,14,19,19,18,16,18,19,16,14,11,12,14,12,9,14,12,9,7,0,11,12,14,12,9,14,16,18,19]    
    elif songName=="Baby Shark":
        BPMval=110
        A=[1,1,.5,.5,.5,.25,.5,.25,.5,.5,.5,.5,.5,.5,.25,.5,.25,.5,.5,.5,.5,.5,.5,.25,.5,.25,.5,.5,.5,1,1]
        B=[9,11,14,14,14,14,14,14,14,9,11,14,14,14,14,14,14,14,9,11,14,14,14,14,14,14,14,14,14,13,0]
    elif songName=="Star Wars":
        BPMval=160
        A=[1.5,0.25,0.25,1.5,0.25,0.25,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.66,0.17,0.17,0.17,0.33,0.33,0.33,1,1,0.33,0.33,0.33,4,4,0.66,0.66,0.66,4,2,0.66,0.66,0.66,4,2,0.66,0.66,0.66,4,1,0.33,0.33,0.33,4,4,0.66,0.66,0.66,4,2,0.66,0.66,0.66,4,2,0.66,0.66,0.66,3,1,1.5,0.5,3,1,1,1,1,1,0.66,0.66,0.66,1.5,0.5,2,1.5,0.5,3,1,1,1,1,0.5,0.5,1.5,0.5,5,1,3,1,1,1,1,1,0.66,0.66,0.66,1.5,0.5,2,1.5,0.5,1.33,0.66,1.33,0.66,1.33,0.66,1.33,0.66,6,0.66,0.66,0.66,3,1,2,2,4,4,0.66,0.66,0.66,4,2,0.66,0.66,0.66,4,2,0.66,0.66,0.66,3,1]
        B=[7,7,7,7,7,7,7,2,12,14,12,7,7,2,12,14,7,7,7,12,7,14,17,0,2,2,2,7,14,12,11,9,19,14,12,11,9,19,14,12,11,12,9,0,2,2,2,7,14,12,11,9,19,14,12,11,9,19,14,12,11,12,9,0,2,2,4,4,12,11,9,7,7,9,11,9,4,6,2,2,4,4,12,11,9,7,14,14,9,9,2,4,6,12,11,9,7,7,9,11,9,4,6,14,14,19,17,15,14,12,10,9,7,14,14,14,14,4,7,12,9,7,14,12,11,9,19,14,12,11,9,19,14,12,11,12,9,0]
    elif songName=="Pink Panther":
        BPMval=320
        A=[1,1,4,1,1,4,1,1,1,1,1,1,1,1,1,1,1,1,1,7,1,1,1,1,7,5,1,1,4,1,1,4,1,1,1,1,1,1,1,1,1,1,1,1,1,18,5,1,1,4,1,1,4,1,1,1,1,1,1,1,1,1,1,1,1,1,7,1,1,1,1,7,9,2,1,2,1,2,1,1,2,1,2,1,2,1,2,1,1,1,2,13,5]
        B=[12,13,0,15,16,0,12,13,0,15,16,0,21,20,0,13,16,0,20,19,18,16,13,11,13,0,12,13,0,15,16,0,12,13,0,15,16,0,21,20,0,16,20,0,25,24,0,12,13,0,15,16,0,12,13,0,15,16,0,21,20,0,13,16,0,20,19,18,16,13,11,13,0,25,23,20,18,16,13,19,18,19,18,19,18,19,18,16,13,11,13,13,0]
    elif songName=="Mission Impossible":
        BPMval=90
        A=[0.5,0.25,0.5,0.25,0.75,0.5,0.25,0.5,0.5,0.5,0.25,0.5,0.25,0.75,0.5,0.25,0.5,0.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,3.5,0.25,0.25,1.75,0.75,0.5,0.5,0.75,3.25,2.25,0.25,0.25,0.25,2.25,0.25,0.25,0.25,2.25,0.25,0.25,0.25,0.25,2.75,1]
        B=[6,6,6,6,6,6,6,9,11,6,6,6,6,6,6,6,4,5,21,18,13,21,18,13,21,18,11,9,11,0,9,6,17,9,6,16,9,6,15,14,13,0,26,23,18,26,23,17,26,23,16,14,16,0,9,6,17,9,6,16,9,6,15,14,13,0,21,18,13,21,18,12,21,18,11,9,11,0,18,21,23,25,0,11,6,9,11,10,6,11,10,9,6,9,8,7,6,0]
    elif songName=="Annoying Tone":
        BPMval=100
        A=[1]
        B=[60]
    else:
        print("error")

    bandwidthHz=bandwidth_hz
    sampleRateHz=sample_rate_hz

    Q=np.array([])
    for j in range(len(A)):
        Q=np.concatenate([Q,noteMaker(A[j],B[j],BPMval,sampleRateHz,bandwidthHz)])

    Q=Q/np.std(Q)

    return Q


# A dictionary to map waveform names to their functions and parameter names
# This is now stored with the functions themselves for better organization.
waveform_definitions = {
    "Narrowband Noise": {
        "func": narrowband_noise_creator,
        "params": [
            {"name": "bandwidth_hz", "title": "Bandwidth (Hz)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"},
            {"name": "technique_length_seconds", "title": "Technique Length (seconds)", "type": "entry"},
            {"name": "interference_type", "title": "Interference Type", "type": "options", "choices": ["complex", "real", "sinc"]}
        ]
    },
    "RRC Modulated Noise": {
        "func": rrc_modulated_noise,
        "params": [
            {"name": "symbol_rate_hz", "title": "Symbol Rate (Hz)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"},
            {"name": "rolloff", "title": "Rolloff (0 < r < 1)", "type": "entry"},
            {"name": "technique_length_seconds", "title": "Technique Length (seconds)", "type": "entry"}
        ]
    },
    "Swept Noise": {
        "func": swept_noise_creator,
        "params": [
            {"name": "sweep_hz", "title": "Sweep (Hz)", "type": "entry"},
            {"name": "bandwidth_hz", "title": "Bandwidth (Hz)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"},
            {"name": "technique_length_seconds", "title": "Technique Length (seconds)", "type": "entry"},
            {"name": "interference_type", "title": "Interference Type", "type": "options", "choices": ["complex", "real", "sinc"]}
        ]
    },
    "Chunked Noise": {
        "func": chunk_noise_creator,
        "params": [
            {"name": "technique_width_hz", "title": "Technique Width (Hz)", "type": "entry"},
            {"name": "chunks", "title": "Chunks (Integer)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"},
            {"name": "technique_length_seconds", "title": "Technique Length (seconds)", "type": "entry"},
            {"name": "interference_type", "title": "Interference Type", "type": "options", "choices": ["complex", "real", "sinc"]}
        ]
    },
    "Noise Tones": {
        "func": noise_tones,
        "params": [
            {"name": "frequencies_str", "title": "Space Delimited Frequencies (Hz)", "type": "entry"},
            {"name": "bandwidth_hz", "title": "Bandwidth (Hz)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"},
            {"name": "technique_length_seconds", "title": "Technique Length seconds", "type": "entry"},
            {"name": "interference_type", "title": "Interference Type", "type": "options", "choices": ["complex", "real", "sinc"]}
        ]
    },
    "Cosine Tones": {
        "func": cosine_tones,
        "params": [
            {"name": "frequencies_str", "title": "Space Delimited Frequencies (Hz)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"},
            {"name": "technique_length_seconds", "title": "Technique Length (seconds)", "type": "entry"}
        ]
    },
    "Phasor Tones": {
        "func": phasor_tones,
        "params": [
            {"name": "frequencies_str", "title": "Space Delimited Frequencies (Hz)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"},
            {"name": "technique_length_seconds", "title": "Technique Length (seconds)", "type": "entry"}
        ]
    },
    "Swept Phasors": {
        "func": swept_phasors,
        "params": [
            {"name": "sweep_hz", "title": "Sweep (Hz)", "type": "entry"},
            {"name": "tones", "title": "Tones (Integer)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"},
            {"name": "technique_length_seconds", "title": "Technique Length (seconds)", "type": "entry"}
        ]
    },
    "Swept Cosines": {
        "func": swept_cosines,
        "params": [
            {"name": "sweep_hz", "title": "Sweep (Hz)", "type": "entry"},
            {"name": "tones", "title": "Tones (Integer)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"},
            {"name": "technique_length_seconds", "title": "Technique Length (seconds)", "type": "entry"}
        ]
    },
    "FM Cosine": {
        "func": FM_cosine,
        "params": [
            {"name": "sweep_range_hz", "title": "Sweep Range (Hz)", "type": "entry"},
            {"name": "modulated_frequency", "title": "Modulated Frequency (Hz)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"},
            {"name": "technique_length_seconds", "title": "Technique Length (seconds)", "type": "entry"}
        ]
    },
    "Song Maker": {
        "func": songMaker,
        "params": [
            {"name": "songName", "title": "Song Name", "type": "options", "choices": ["Air Force Song", "Anchors Away", "Marine Hymn","Baby Shark","Pink Panther","Star Wars","Mission Impossible","Annoying Tone"]},
            {"name": "bandwidth_hz", "title": "Bandwidth (Hz)", "type": "entry"},
            {"name": "sample_rate_hz", "title": "Sample Rate (Hz)", "type": "entry"}
        ]
    }
}

