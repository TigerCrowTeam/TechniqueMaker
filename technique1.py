import math
import struct
import numpy as np
from numpy.typing import NDArray # For specific NumPy array type hints

def numpy_complex_to_binary_file(complex_array, filename="complex_output.bin"):
    """
    Converts a NumPy array (real or complex) to a binary file suitable for
    GNU Radio's File Source block with complex float output.

    Args:
        complex_array (numpy.ndarray): A NumPy array of real or complex numbers.
        filename (str, optional): The name of the output binary file.
                                   Defaults to "complex_output.bin".
    """
    complex_flat = np.asarray(complex_array).flatten()
    output_data = b''

    counter=0
    for num in complex_flat:
        if np.iscomplexobj(num):
            real_part = num.real
            imag_part = num.imag
        else:
            real_part = float(num)
            imag_part = 0.0
        output_data += struct.pack('<ff', real_part, imag_part)
        counter=counter+1
        if counter%10000==0:
            print(counter)

    with open(filename, 'wb') as f:
        f.write(output_data)

    print(f"Complex data written to '{filename}' in binary format (interleaved floats).")



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
    
    if sample_rate_hz/2<=bandwidth_hz:
        raise ValueError("sample_rate_hz needs to be more than 2 times greater than bandwidth_hz")
        
    if technique_length_seconds*sample_rate_hz<16:
        raise ValueError("Product of sample_rate_hz and technique_length_seconds needs to be at least 16")
    
    """
    Calculation of the number of samples in the waveform based on sample rate
    and technique length.  Number of samples is rounded down if product
    of sample rate and technique length is not an integer.
    """
    num_samples=math.floor(sample_rate_hz*technique_length_seconds)
    
    """
    The number of noise phasors is equal to the bandwidth of the noise 
    divided by the frequency increment which is equal to the sample rate
    divided by the number of samples.  This value is rounded down.
    """
    num_interference_phasors=math.floor(bandwidth_hz*num_samples/sample_rate_hz)
    
    if num_interference_phasors == 0 and bandwidth_hz > 0:
        num_interference_phasors= 2 # Ensure at least two phasors for non-zero bandwidth
    elif num_interference_phasors%2==1:
        num_interference_phasors=num_interference_phasors+1
    
    
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
    freq_domain=freq_domain*np.exp(1j*phases)
    # Calculate the power scaling factor. This normalizes the output power.
    # The '+1' in num_interference_phasors + 1 accounts for the DC component's contribution to power.
    power_scaler=np.sqrt(num_interference_phasors+1)/num_samples   
    # Perform the inverse FFT and apply the power scaling.
    output_signal=np.fft.ifft(freq_domain)/power_scaler
    # Round the values to mitigate floating-point errors from the IFFT,
    # ensuring cleaner real and imaginary parts.
    output_signal=np.round(np.real(output_signal),8)+np.round(np.imag(output_signal),8)*1j
    return output_signal

def swept_noise_creator(
    sweep_hz: float,
    bandwidth_hz: float,
    sample_rate_hz: float,
    technique_length_seconds: float,
    interference_type: str = "complex"
) -> NDArray:
    
    #Calls the noise function to generate narrowband noise
    noise=narrowband_noise_creator(bandwidth_hz, sample_rate_hz, technique_length_seconds, interference_type)
    #Creates a time array for the technique length
    time=np.linspace(0, technique_length_seconds-technique_length_seconds/len(noise), len(noise), dtype=np.float64)
    #Function of the desired sweep
    #y=mx+b
    freq_sweep_func=sweep_hz/technique_length_seconds*time-sweep_hz/2
    #Numerical integration of the sweep
    #Done with a cumulative sum divided by sample rate
    cum_freq_sweep_func=np.cumsum(freq_sweep_func)/sample_rate_hz
    #Creates a phasor that sweeps over the sweep within the technique time length
    shifter=np.exp(1j*2*np.pi*cum_freq_sweep_func)
    #Multiplies the noise by the sweeping phasor to make the noise sweep
    swept_noise=noise*shifter
    
    return swept_noise

def chunk_noise_creator(
    technique_width_hz: float,
    chunks: int,
    sample_rate_hz: float,
    technique_length_seconds: float,
    interference_type: str = "complex"
) -> NDArray:
    bandwidth_hz=technique_width_hz/chunks
    #Calls the noise function to generate narrowband noise
    noise=narrowband_noise_creator(bandwidth_hz, sample_rate_hz, technique_length_seconds, interference_type)
    
    #Creates a time array for the technique length
    time=np.linspace(0, technique_length_seconds-technique_length_seconds/len(noise), len(noise), dtype=np.float64)
    
    freq_chunk_centers=np.linspace(-1*(technique_width_hz/2-technique_width_hz/chunks/2),(technique_width_hz/2-technique_width_hz/chunks/2),chunks)
    
    timed_chunks=np.floor(time/technique_length_seconds*chunks).astype(int)

    # Create a NumPy array containing integers from 0 to n-1
    arr = np.arange(chunks)

    # Shuffle the array in-place
    np.random.shuffle(arr)
    
    timed_chunks_randomized=arr[timed_chunks]
    
    shifter=np.exp(1j*2*np.pi*freq_chunk_centers[timed_chunks_randomized]*time)

    #Multiplies the noise by the sweeping phasor to make the noise sweep
    chunked_noise=noise*shifter
    
    return chunked_noise

def swept_tone(
    sweep_hz: float,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    num_samples=math.floor(sample_rate_hz*technique_length_seconds)
    #Creates a time array for the technique length
    time=np.linspace(0, technique_length_seconds-technique_length_seconds/num_samples, num_samples, dtype=np.float64)
    #Function of the desired sweep
    #y=mx+b
    freq_sweep_func=sweep_hz/technique_length_seconds*time-sweep_hz/2
    #Numerical integration of the sweep
    #Done with a cumulative sum divided by sample rate
    cum_freq_sweep_func=np.cumsum(freq_sweep_func)/sample_rate_hz
    #Creates a phasor that sweeps over the sweep within the technique time length
    shifter=np.exp(1j*2*np.pi*cum_freq_sweep_func)
    
    return shifter

def swept_phasors(
    sweep_hz: float,
    tones: int,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    num_samples=math.floor(sample_rate_hz*technique_length_seconds)
    #Creates a time array for the technique length
    time=np.linspace(0, technique_length_seconds-technique_length_seconds/num_samples, num_samples, dtype=np.float64)
    swept_tones = np.zeros(num_samples, dtype=float)
    tone_freqs=np.linspace(-sweep_hz/2,sweep_hz/2,tones+1)
    mini_sweep_hz=sweep_hz/tones
    
    for k in range(tones):
        freq_sweep_func=mini_sweep_hz/technique_length_seconds*time+tone_freqs[k]
        #Numerical integration of the sweep
        #Done with a cumulative sum divided by sample rate
        cum_freq_sweep_func=np.cumsum(freq_sweep_func)/sample_rate_hz
        #Creates a phasor that sweeps over the sweep within the technique time length
        swept_tones=swept_tones+np.exp(1j*2*np.pi*cum_freq_sweep_func)
        
    return swept_tones

def swept_cosines(
    sweep_hz: float,
    tones: int,
    sample_rate_hz: float,
    technique_length_seconds: float
) -> NDArray:
    num_samples=math.floor(sample_rate_hz*technique_length_seconds)
    #Creates a time array for the technique length
    time=np.linspace(0, technique_length_seconds-technique_length_seconds/num_samples, num_samples, dtype=np.float64)
    swept_tones = np.zeros(num_samples, dtype=float)
    tone_freqs=np.linspace(-sweep_hz/2,sweep_hz/2,tones+1)
    mini_sweep_hz=sweep_hz/tones
    
    for k in range(tones):
        freq_sweep_func=mini_sweep_hz/technique_length_seconds*time+tone_freqs[k]
        #Numerical integration of the sweep
        #Done with a cumulative sum divided by sample rate
        cum_freq_sweep_func=np.cumsum(freq_sweep_func)/sample_rate_hz
        #Creates a phasor that sweeps over the sweep within the technique time length
        swept_tones=swept_tones+np.cos(2*np.pi*cum_freq_sweep_func)
        
    return swept_tones
        
    

X=swept_tone(100000,240000,2)
numpy_complex_to_binary_file(X, "sweptNoise01.bin")