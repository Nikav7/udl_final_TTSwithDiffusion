import numpy as np
import librosa
import librosa.display
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt
import soundfile as sf
import re

def calculate_snr(signal):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean((signal - np.mean(signal)) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_sdr(signal):
    distortion_power = np.mean((signal - np.mean(signal)) ** 2)
    signal_power = np.mean(signal ** 2)
    sdr = 10 * np.log10(signal_power / distortion_power)
    return sdr

def compute_spectrogram(signal, sr):
    f, t, Sxx = scipy.signal.spectrogram(signal, sr)
    return f, t, Sxx

def get_frequency_stats(signal, sr):
    fft_spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1/sr)
    
    if np.all(fft_spectrum == 0):
        return 0, 0, 0
    
    min_freq = frequencies[np.where(fft_spectrum > 0)[0][0]]
    max_freq = frequencies[np.where(fft_spectrum > 0)[0][-1]]
    mean_freq = np.average(frequencies, weights=fft_spectrum)
    return min_freq, max_freq, mean_freq

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

def analyze_audio(original_file, generated_file):
    original, sr = librosa.load(original_file, sr=None)
    generated, _ = librosa.load(generated_file, sr=None)
    
    min_freq_orig, max_freq_orig, mean_freq_orig = get_frequency_stats(original, sr)
    min_freq_gen, max_freq_gen, mean_freq_gen = get_frequency_stats(generated, sr)
    
    snr_original = calculate_snr(original)
    sdr_original = calculate_sdr(original)
    snr_generated = calculate_snr(generated)
    sdr_generated = calculate_sdr(generated)
    
    phase_original = compute_phase(original)
    phase_generated = compute_phase(generated)
    #phase_difference = phase_original - phase_generated
    
    print(f"Original SNR: {snr_original:.2f} dB, Generated SNR: {snr_generated:.2f} dB")
    print(f"Original SDR: {sdr_original:.2f} dB, Generated SDR: {sdr_generated:.2f} dB")
    print(f"Original Min Frequency: {min_freq_orig:.2f} Hz, Generated Min Frequency: {min_freq_gen:.2f} Hz")
    print(f"Original Max Frequency: {max_freq_orig:.2f} Hz, Generated Max Frequency: {max_freq_gen:.2f} Hz")
    print(f"Original Mean Frequency: {mean_freq_orig:.2f} Hz, Generated Mean Frequency: {mean_freq_gen:.2f} Hz")
    #print(f"Phase Difference: {phase_difference:.2f} radians")
    
    plot_waveform_and_spectrogram(original, generated, sr)


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
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, duration_loss, label='Duration Loss')
    plt.plot(epochs, prior_loss, label='Prior Loss')
    plt.plot(epochs, diffusion_loss, label='Diffusion Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    #print(f"Using GPU: {torch.cuda.is_available()}")  # Check if GPU is available

    #analyze_audio("originalITA.wav", "generatedITA.wav")
    plot_train_log("train_log.txt")
