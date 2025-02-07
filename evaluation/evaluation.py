import numpy as np
import parselmouth
import librosa
import librosa.display
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt
import soundfile as sf
import re
from scipy.io import wavfile
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw 
from pesq import pesq

def compute_spectrogram(signal, sr):
    f, t, Sxx = scipy.signal.spectrogram(signal, sr)
    return f, t, Sxx

def hz_to_mel(f):
    """Convert frequency (Hz) to mel scale."""
    return 2595 * np.log10(1 + f / 700)

def snr_fft(signal, sr):
    N = len(signal)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/sr)

    #power spectrum (magnitude squared of FFT)
    power_spectrum = np.abs(X[:N//2])**2
    freqs = freqs[:N//2]  #positive freqs

    #fundamental frequency
    fundamental_idx = np.argmax(power_spectrum)
    fundamental_power = power_spectrum[fundamental_idx]

    #total noise power (excluding fundamental)
    noise_power = np.sum(power_spectrum) - fundamental_power

    #snr
    snr = 10 * np.log10(fundamental_power / noise_power)

    return snr

def snr_scf(signal, sr, n_harm=4):
    #more accurate power estimation with power spectral density (PSD) instead of manual fft
    freqs, psd = scipy.signal.periodogram(signal, sr, scaling='density')
    
    #fundamental frequency (strongest peak)
    fundamental_idx = np.argmax(psd)
    fundamental_power = psd[fundamental_idx]
    fundamental_freq = freqs[fundamental_idx]
    
    #power of harmonics, set on 4 by default
    harmonic_power = psd[fundamental_idx]
    for h in range(2, n_harm + 1):
        harmonic_freq = h * fundamental_freq
        harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq)) 
        if harmonic_idx < len(psd):
            harmonic_power += psd[harmonic_idx]

    #total power
    total_power = np.sum(psd)

    #noise considering harmonics
    noise_power = total_power - harmonic_power
    
    #snr
    snr = 10 * np.log10(fundamental_power / noise_power)
    #Spectral Crest Factor (SCF), aka the Peak-to-Average Power Ratio (PAPR).
    #measures how dominant the peak power (fundamental frequency) is compared to the average power of the entire spectrum.
    #A high value indicates a signal with a strong fundamental component (e.g., a pure sine wave).
    #A low value suggests a more uniform power distribution, indicating more noise or a spread spectrum.
    scf = np.max(psd) / np.mean(psd)
    
    return snr, scf

def get_frequency_stats(signal, sr):
    fft_spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1/sr)
    
    if np.all(fft_spectrum == 0):
        return 0, 0, 0
    
    min_freq = frequencies[np.where(fft_spectrum > 0)[0][0]]
    max_freq = frequencies[np.where(fft_spectrum > 0)[0][-1]]
    mean_freq = np.average(frequencies, weights=fft_spectrum)

    variance = np.average((frequencies - mean_freq) ** 2, weights=fft_spectrum)
    std_freq = np.sqrt(variance)
    return min_freq, max_freq, mean_freq, std_freq

def compute_phase(signal):
    return np.angle(scipy.fftpack.fft(signal))

def plot_waveform_and_spectrogram(original, generated, sr):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].set_title("Original Waveform")
    axes[0, 0].plot(original, alpha=0.7)
    
    axes[1, 0].set_title("Generated Waveform")
    axes[1, 0].plot(generated, alpha=0.7)
    
    f1, t1, Sxx1 = compute_spectrogram(original, sr)
    axes[0, 1].set_title("Original Spectrogram")
    axes[0, 1].imshow(10 * np.log10(Sxx1), aspect='auto', origin='lower', extent=[t1.min(), t1.max(), f1.min(), f1.max()])
    
    f2, t2, Sxx2 = compute_spectrogram(generated, sr)
    axes[1, 1].set_title("Generated Spectrogram")
    axes[1, 1].imshow(10 * np.log10(Sxx2), aspect='auto', origin='lower', extent=[t2.min(), t2.max(), f2.min(), f2.max()])
    
    plt.tight_layout()
    plt.show()

def pitch_(audio):
    y, sr = librosa.load(audio, sr=None)
    #fundamental frequency (f0) detection with pYIN
    fmin = librosa.note_to_hz('C2')  # ~65 Hz
    fmax = librosa.note_to_hz('C7')  # ~2093 Hz
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
    # Create time axis for plotting
    times = librosa.times_like(f0, sr=sr)
    return times, f0

def pitch_rapt(audio, time_step=0.01, min_pitch=65, max_pitch=2093):
    """Extracts pitch (F0) using RAPT (Praat)."""
    sound = parselmouth.Sound(audio)
    pitch = sound.to_pitch(time_step=time_step, pitch_floor=min_pitch, pitch_ceiling=max_pitch)
    
    times = pitch.xs()  # Time stamps
    f0 = pitch.selected_array['frequency']  # F0 values
    f0[f0 == 0] = np.nan  # Replace unvoiced parts with NaN
    
    return times, f0

def pitch_comp(original, generated, method = "rapt"):

    if method == "pyin":
        extract_f0 = pitch_
    elif method == "rapt":
        extract_f0 = pitch_rapt

    times_orig, f0_orig = extract_f0(original)
    times_gen, f0_gen = extract_f0(generated)

    #both signals needs to have the same length for RMSE computation
    #we use Dynamic Time Warping (DTW) to align the sequences
    #remove nans for proper DTW alignment
    valid_idx_orig = ~np.isnan(f0_orig)
    valid_idx_gen = ~np.isnan(f0_gen)
    
    times_orig, f0_orig = times_orig[valid_idx_orig], f0_orig[valid_idx_orig]
    times_gen, f0_gen = times_gen[valid_idx_gen], f0_gen[valid_idx_gen]

    # Apply DTW to align pitch sequences
    distance, path = fastdtw(f0_orig, f0_gen, dist=euclidean)
    
    # Align F0 values using DTW mapping
    f0_orig_aligned = np.array([f0_orig[i] for i, _ in path])
    f0_gen_aligned = np.array([f0_gen[j] for _, j in path])

    # Compute RMSE in Hz
    rmse_hz = np.sqrt(np.mean((f0_orig_aligned - f0_gen_aligned) ** 2))

    # Compute RMSE in the mel scale
    mel_orig, mel_gen = hz_to_mel(f0_orig_aligned), hz_to_mel(f0_gen_aligned)
    rmse_mel = np.sqrt(np.mean((mel_orig - mel_gen) ** 2))

    # Plot the pitch contours
    plt.figure(figsize=(10, 4))
    plt.plot(times_orig, f0_orig, label="Original Pitch", color="r")
    plt.plot(times_gen, f0_gen, label="Generated Pitch", color="b", linestyle="dashed")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Pitch Comparison using {method.upper()} (DTW-RMSE: {rmse_hz:.2f} Hz, {rmse_mel:.2f} Mel)")
    plt.legend()
    plt.grid()
    plt.show()

    return rmse_hz, rmse_mel


def analyze_audio(original_file, generated_file):
    original, sr = librosa.load(original_file, sr=None)
    generated, _ = librosa.load(generated_file, sr=None)
    
    min_freq_orig, max_freq_orig, mean_freq_orig, std_freq_orig = get_frequency_stats(original, sr)
    min_freq_gen, max_freq_gen, mean_freq_gen, std_freq_gen = get_frequency_stats(generated, sr)
    
    snr_fftor = snr_fft(original, sr)
    snr_original, scf_or = snr_scf(original, sr)
    snr_fftgen = snr_fft(generated, sr)
    snr_generated, scf_gen = snr_scf(generated, sr)
    
    phase_original = compute_phase(original)
    #print(phase_original)
    phase_generated = compute_phase(generated)
    #print(phase_generated)
    #phase_difference = phase_original - phase_generated

    
    print(f"Original SNR (fft): {snr_fftor:.2f} dB, Generated SNR (fft): {snr_fftgen:.2f} dB")
    print(f"Original SNR (harm): {snr_original:.2f} dB, Generated SNR (harm): {snr_generated:.2f} dB")
    print(f"Original SCF: {scf_or:.2f}, Generated SCF: {scf_gen:.2f}")
    print(f"Original Min Frequency: {min_freq_orig:.2f} Hz, Generated Min Frequency: {min_freq_gen:.2f} Hz")
    print(f"Original Max Frequency: {max_freq_orig:.2f} Hz, Generated Max Frequency: {max_freq_gen:.2f} Hz")
    #print(f"Original Mean Frequency: {mean_freq_orig:.2f} Hz, Generated Mean Frequency: {mean_freq_gen:.2f} Hz")
    print(f"Original Mean Frequency: {mean_freq_orig:.2f} ± {std_freq_orig:.2f} Hz")
    print(f"Generated Mean Frequency: {mean_freq_gen:.2f} ± {std_freq_gen:.2f} Hz")
    #print(f"Original Phase: {phase_original:.2f} radians, Generated Phase: {phase_generated:.2f} radians")
    #print(f"Phase Difference: {phase_difference:.2f} radians")

    plot_waveform_and_spectrogram(original, generated, sr)
    pitch_comp(original_file, generated_file, method="rapt")

def pesqs(original, generated):
    rate, ref = wavfile.read(original)
    rate, deg = wavfile.read(generated)

    if rate != 16000:
        ref = scipy.signal.resample(ref, 16000, t=None, axis=0, window=None, domain='time')
        deg = scipy.signal.resample(deg, 16000, t=None, axis=0, window=None, domain='time')

    ref = np.ravel(ref)
    deg = np.ravel(deg)
    wb_pesq = pesq(16000, ref, deg, 'wb')
    nb_pesq = pesq(16000, ref, deg, 'nb')

    print(f"Wide-band pesq: {wb_pesq:.2f}, Narrow-band pesq: {nb_pesq:.2f}")

def plot_train_log(file_path):
    epochs = []
    duration_loss = []
    prior_loss = []
    diffusion_loss = []
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"Epoch (\d+): duration loss = ([\d.]+) \| prior loss = ([\d.]+) \| diffusion loss = ([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                duration_loss.append(float(match.group(2)))
                prior_loss.append(float(match.group(3)))
                diffusion_loss.append(float(match.group(4)))
    
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, duration_loss, label='Duration Loss')
    plt.plot(epochs, prior_loss, label='Prior Loss')
    plt.plot(epochs, diffusion_loss, label='Diffusion Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()
    #plt.savefig("train_log2000.png")


if __name__ == "__main__":
    #print(f"Using GPU: {torch.cuda.is_available()}")
    or_path = "udl_final/evaluation/originalITA.wav"
    gen_path = "udl_final/evaluation/generatedITA_ftAB1500cmu.wav"
    analyze_audio(or_path, gen_path)
    #pesqs(or_path, gen_path)
    #plot_train_log("udl_final/evaluation/train_log2000.txt")
