import sys
import torch
#print(torch.cuda.is_available())
#print(torch.cuda.get_device_name(0))
import librosa
import soundfile as sf
import IPython.display as ipd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import json
import datetime as dt
import os
# Add the directory containing the 'model' module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'SpeechBackbones/GradTTS')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'SpeechBackbones/GradTTS/hifigan')))
# changed original path names to do this. (removed - from names)
from SpeechBackbones.GradTTS import params
from SpeechBackbones.GradTTS.utils import intersperse
from SpeechBackbones.GradTTS.text import text_to_sequence, cmudict
from SpeechBackbones.GradTTS.text.symbols import symbols
from SpeechBackbones.GradTTS.model.tts import GradTTS
from SpeechBackbones.GradTTS.hifigan.env import AttrDict
from SpeechBackbones.GradTTS.hifigan.models import Generator as HiFiGAN

# Paths to checkpoints
# Define the base path for your project
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
GRAD_TTS_CKPT = os.path.join(BASE_PATH, "SpeechBackbones/GradTTS/checkpts/grad_50.pt")
GRAD_TTS_CKPT_LIBRI = os.path.join(BASE_PATH, "SpeechBackbones/GradTTS/checkpts/grad-tts-libri-tts.pt")
HIFIGAN_CKPT = os.path.join(BASE_PATH, "SpeechBackbones/GradTTS/checkpts/hifigan.pt")
HIFIGAN_CONFIG = os.path.join(BASE_PATH, "SpeechBackbones/GradTTS/checkpts/hifigan-config.json")
CMU_DICT = os.path.join(BASE_PATH, "SpeechBackbones/GradTTS/resources/cmu_dictionary")
N_SPKS = 1  # 247 for Libri-TTS model and 1 for single speaker (LJSpeech)

# Initialize Grad-TTS model
def initialize_grad_tts():
    generator = GradTTS(len(symbols)+1, N_SPKS, params.spk_emb_dim,
                    params.n_enc_channels, params.filter_channels,
                    params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                    params.enc_kernel, params.enc_dropout, params.window_size,
                    params.n_feats, params.dec_dim, params.beta_min, params.beta_max,
                    pe_scale=1000)  # pe_scale=1 for `grad-tts-old.pt`
    generator.load_state_dict(torch.load(GRAD_TTS_CKPT, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    return generator

# Initialize hifi gan model
def initialize_hifigan():
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    hifigan = HiFiGAN(h)
    hifigan.load_state_dict(torch.load(HIFIGAN_CKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = hifigan.cuda().eval()
    hifigan.remove_weight_norm()
    return hifigan
    #%matplotlib inline

def preprocess_text(text):
    cmu = cmudict.CMUDict(CMU_DICT)
    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda() [None]
    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

    return x, x_lengths

def plot_results(y_enc, y_dec, attn, filename='results_plot.png'):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.title('Encoder outputs')
    plt.imshow(y_enc.cpu().squeeze(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('Decoder outputs')
    plt.imshow(y_dec.cpu().squeeze(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title('Alignment')
    plt.imshow(attn.cpu().squeeze(), aspect='auto', origin='lower')

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()

def reconstruct_waveform(y_dec, hifigan):
    with torch.no_grad():
        audio = hifigan.forward(y_dec).cpu().squeeze().clamp(-1, 1)
    #ipd.display(ipd.Audio(audio, rate=22050))
    audio_data = audio.numpy()
    sample_rate = 22050
    wavfile.write('generated_audio.wav', sample_rate, audio_data)

def text_to_speech(text = "Random text wow ok."):
    generator = initialize_grad_tts()
    hifigan = initialize_hifigan()
    x, x_lengths = preprocess_text(text)

    t = dt.datetime.now()
    y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=50, temperature=1.3,
                                       stoc=False, spk=None if N_SPKS==1 else torch.LongTensor([15]).cuda(),
                                       length_scale=0.91)
    t = (dt.datetime.now() - t).total_seconds()
    print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

    plot_results(y_enc, y_dec, attn)
    reconstruct_waveform(y_dec, hifigan)



# Example usage
if __name__ == "__main__":
    print(f"Using GPU: {torch.cuda.is_available()}")  # Check if GPU is available

    text = "Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. "
    audio = text_to_speech(text)
    #print(f"Audio saved to output.wav")