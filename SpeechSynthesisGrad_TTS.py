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
GRAD_TTS_CKPT = os.path.join(BASE_PATH, "SpeechBackbones/GradTTS/checkpts/grad_1000cmu.pt")#grad_500Fresh.pt")
GRAD_TTS_CKPT_LIBRI = os.path.join(BASE_PATH, "SpeechBackbones/GradTTS/checkpts/grad-tts-libri-tts.pt")
HIFIGAN_CKPT = os.path.join(BASE_PATH, "SpeechBackbones/GradTTS/checkpts/hifigan.pt")
HIFIGAN_CONFIG = os.path.join(BASE_PATH, "SpeechBackbones/GradTTS/checkpts/hifigan-config.json")
CMU_DICT = os.path.join(BASE_PATH, "SpeechBackbones/GradTTS/resources/cmu_dictionary")
N_SPKS = 1  # 2 for multi_AB_LJ, 247 for Libri-TTS model and 1 for single speaker

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

def reconstruct_waveform(y_dec, hifigan, filename):
    with torch.no_grad():
        audio = hifigan.forward(y_dec).cpu().squeeze().clamp(-1, 1)
    #ipd.display(ipd.Audio(audio, rate=22050))
    audio_data = audio.numpy()
    sample_rate = 22050
    wavfile.write(filename, sample_rate, audio_data)

def text_to_speech(sentences, model_name, speaker_index=0):
    generator = initialize_grad_tts()
    hifigan = initialize_hifigan()

    for idx, text in enumerate(sentences, start=1):
        x, x_lengths = preprocess_text(text)
        spk = torch.LongTensor([speaker_index]).cuda() if params.n_spks > 1 else None

        print(f"Processing sentence {idx}: {text}")
        t_start = dt.datetime.now()

        y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=1000, temperature=1.3,
                                               stoc=False, spk=spk, length_scale=1.0)

        t_elapsed = (dt.datetime.now() - t_start).total_seconds()
        print(f'Grad-TTS RTF: {t_elapsed * 22050 / (y_dec.shape[-1] * 256)}')

        # Generate filenames
        audio_filename = f"generated_audio_{model_name}_sentence{idx}.wav"
        plot_filename = f"results_plot_{model_name}_sentence{idx}.png"

        # Save output files
        plot_results(y_enc, y_dec, attn, plot_filename)
        reconstruct_waveform(y_dec, hifigan, audio_filename)

        print(f"Saved: {audio_filename} and {plot_filename}")



if __name__ == "__main__":
    speaker_index = 1
    model_name = "1000cmuModel"
    # sent1 = "La società dell'impero ottomano non conosce la nobiltà di nascita, è una società dove non esiste nemmeno il concetto di nobiltà, il pregiudizio per cui solo chi ha un grande nome e una famiglia prestigiosa alle spalle ha diritto ad occupare i posti di comando."
    # sent2 = "L'Europa si va dividendo in regni su base geografica e in una certa misura anche nazionale."
    # sent3 = "Tutti i leader dell’impero ottomano erano nati cristiani ed erano figli di povera gente."
    # sent4 = "Ti prego di ascoltare questa voce convertita con i dati provenienti da un video del prof. Alessandro Barbero e il dataset LJ Speech. Steve Jobs era il San Francesco del medioevo."
    # sent5 = "Per gli ordini religiosi del medioevo la riconoscibilità è un elemento fondamentale del successo, perché sono in concorrenza fra loro."    
    # sent6 = "L’idea che i pascià, i visir e gli ammiragli, nei loro sontuosi palazzi di Costantinopoli, da dove governano un impero, siano tutti figli di pastori sconvolge gli osservatori europei."
    # sent7 = "Nella società dell’impero ottomano non c’è niente che temperi la volontà assoluta del sultano, non ci sono forze organizzate o corpi, come si diceva nell’Occidente moderno: non c’è una Chiesa, non c’è l’università, non i comuni urbani, e non c’è una nobiltà. Il risultato è una società al tempo stesso più aperta al talento e più esposta alla tirannia."
    # # Long descriptive sentences
    # sent8 = "Recentemente, i modelli probabilistici di diffusione e la qualità dell'output generato hanno mostrato un alto potenziale nella modellazione di distribuzioni di dati complesse, mentre il calcolo stocastico ha fornito un punto di vista unificato su queste tecniche, consentendo schemi di inferenza flessibili. Questo audio è generato con Grad-TiTieS, un nuovo modello di sintesi vocale con decodificatore che produce spettrogrammi mel trasformando gradualmente il rumore previsto dal codificatore e allineato con l'input di testo, mediante ricerca di allineamento monotonico. Il quadro delle equazioni differenziali stocastiche ci aiuta a generalizzare i modelli probabilistici di diffusione convenzionali per ricostruire dati dal rumore, con parametri diversi. Ciò consente di rendere flessibile questa ricostruzione controllando esplicitamente il compromesso tra qualità del suono e velocità di inferenza. La valutazione umana soggettiva mostra che Grad-TTS è competitivo con gli approcci di sintesi vocale all'avanguardia in termini di Mean Opinion Score."

    # ENG
    sent1 = "The Ottoman Empire society does not know the nobility of birth, it is a society where there is not even the concept of nobility, the prejudice that only those who have a great name and a prestigious family behind them have the right to occupy the positions of power."
    sent2 = "The quick brown fox jumps over the lazy dog?"
    sent3 = "All the leaders of the Ottoman Empire were born Christians and were the children of poor people."
    sent4 = "The idea that the pashas, viziers, and admirals, which are in their sumptuous palaces in Constantinople ruling an empire, are all shepherds’ sons, shocks the European observers."
    sent5 = "Please listen to this converted voice, using data from prof. Alessandro Barbero and the LJ Speech dataset. Steve Jobs was the Saint Francis of the Middle Ages."
    sent6 = "For medieval religious orders, recognizability is a fundamental element of success because they compete with each other."
    sent7 = "In Ottoman imperial society, nothing moderates the absolute will of the sultan; there are no organized forces or bodies, as they were called in the modern West: there is no Church, no university, no urban communes, and no nobility. The result is a society that is at once more open to talent and more vulnerable to tyranny."
    # Long descriptive sentence
    sent8 = "Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. This audio is generated with Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score."

    # REAL AUDIOS
    # ita from AB dataset
    #text = "Per gli ordini religiosi del medioevo la riconoscibilità è un elemento fondamentale del successo, perché sono in concorrenza fra loro."


    # add array instead of single string 
    sentences = [sent1, sent2, sent3, sent4, sent5, sent6, sent7, sent8]

    # text = "Under these circumstances, unnatural as they are, with proper management, the bean will thrust forth its radicle and its plumule."
    audio = text_to_speech(sentences, model_name, speaker_index)
    #print(f"Audio saved to generated_audio.wav")
