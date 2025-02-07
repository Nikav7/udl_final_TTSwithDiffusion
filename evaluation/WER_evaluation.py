import os
import csv
import torch
import whisper # Running this with a separate environment with whisper installed
from jiwer import wer
import matplotlib.pyplot as plt
import librosa

# Set language and model for evaluation
models = {
    "GradTTSModel": "en",
    "500cmuModel": "en",
    "1000cmuModel": "en",
    "1500cmuModel": "en",
    "GradTTSModelita": "it",
    "500itaModel": "it",
    "1000itaModel": "it",
    "1500itaModel": "it"
}
# Load Whisper model for transcription
model = whisper.load_model("medium")

reference_texts = {
    "en": [
        "The Ottoman Empire society does not know the nobility of birth, it is a society where there is not even the concept of nobility, the prejudice that only those who have a great name and a prestigious family behind them have the right to occupy the positions of power.",
        "The quick brown fox jumps over the lazy dog?",
        "All the leaders of the Ottoman Empire were born Christians and were the children of poor people.",
        "The idea that the pashas, viziers, and admirals, which are in their sumptuous palaces in Constantinople ruling an empire, are all shepherds’ sons, shocks the European observers.",
        "Please listen to this converted voice, using data from prof. Alessandro Barbero and the LJ Speech dataset. Steve Jobs was the Saint Francis of the Middle Ages.",
        "For medieval religious orders, recognizability is a fundamental element of success because they compete with each other.",
        "In Ottoman imperial society, nothing moderates the absolute will of the sultan; there are no organized forces or bodies, as they were called in the modern West: there is no Church, no university, no urban communes, and no nobility. The result is a society that is at once more open to talent and more vulnerable to tyranny.",
        "Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. This audio is generated with Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score."
    ],
    "it": [
        "La società dell'impero ottomano non conosce la nobiltà di nascita, è una società dove non esiste nemmeno il concetto di nobiltà, il pregiudizio per cui solo chi ha un grande nome e una famiglia prestigiosa alle spalle ha diritto ad occupare i posti di comando.",
        "L'Europa si va dividendo in regni su base geografica e in una certa misura anche nazionale.",
        "Tutti i leader dell’impero ottomano erano nati cristiani ed erano figli di povera gente.",
        "Ti prego di ascoltare questa voce convertita con i dati provenienti da un video del prof. Alessandro Barbero e il dataset LJ Speech. Steve Jobs era il San Francesco del medioevo.",
        "Per gli ordini religiosi del medioevo la riconoscibilità è un elemento fondamentale del successo, perché sono in concorrenza fra loro."    ,
        "L’idea che i pascià, i visir e gli ammiragli, nei loro sontuosi palazzi di Costantinopoli, da dove governano un impero, siano tutti figli di pastori sconvolge gli osservatori europei.",
        "Nella società dell’impero ottomano non c’è niente che temperi la volontà assoluta del sultano, non ci sono forze organizzate o corpi, come si diceva nell’Occidente moderno: non c’è una Chiesa, non c’è l’università, non i comuni urbani, e non c’è una nobiltà. Il risultato è una società al tempo stesso più aperta al talento e più esposta alla tirannia.",
        "Recentemente, i modelli probabilistici di diffusione e la qualità dell'output generato hanno mostrato un alto potenziale nella modellazione di distribuzioni di dati complesse, mentre il calcolo stocastico ha fornito un punto di vista unificato su queste tecniche, consentendo schemi di inferenza flessibili. Questo audio è generato con Grad-TiTieS, un nuovo modello di sintesi vocale con decodificatore che produce spettrogrammi mel trasformando gradualmente il rumore previsto dal codificatore e allineato con l'input di testo, mediante ricerca di allineamento monotonico. Il quadro delle equazioni differenziali stocastiche ci aiuta a generalizzare i modelli probabilistici di diffusione convenzionali per ricostruire dati dal rumore, con parametri diversi. Ciò consente di rendere flessibile questa ricostruzione controllando esplicitamente il compromesso tra qualità del suono e velocità di inferenza. La valutazione umana soggettiva mostra che Grad-TTS è competitivo con gli approcci di sintesi vocale all'avanguardia in termini di Mean Opinion Score."
    ]
}

def transcribe_audio(file_path, lang):
    """Transcribes an audio file using Whisper ASR."""
    print(f"Transcribing: {file_path} ({lang})")
    try:
        # Load audio
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Decode audio
        options = whisper.DecodingOptions(fp16=torch.cuda.is_available(), language=lang)
        result = whisper.decode(model, mel, options)

        return result.text.strip()
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
        return None

def calculate_wer_for_models():
    """Calculates WER for multiple models and saves results to a CSV file."""
    results = []

    for model_name, lang in models.items():
        print(f"\nEvaluating model: {model_name} ({lang})")
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Outputs', model_name))
        generated_audio_paths = [os.path.join(base_dir, f"generated_audio_{model_name}_sentence{i}.wav") for i in range(1, 9)]

        wer_scores = []
        for i, (ref_text, audio_path) in enumerate(zip(reference_texts[lang], generated_audio_paths), start=1):
            transcription = transcribe_audio(audio_path, lang)
            if transcription:
                sample_wer = wer(ref_text.lower(), transcription.lower())
                wer_scores.append(sample_wer)
                print(f"Sample {i}: WER = {sample_wer:.3f}")
                results.append([model_name, lang, f"sentence{i}", sample_wer])

        avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0
        results.append([model_name, lang, "Average", avg_wer])
        print(f"\n{model_name} - Average WER: {avg_wer:.3f}")

    # Save results to CSV
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "WER_results.csv"))
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerow(["Model", "Language", "Sentence", "WER"])
        writer.writerows(results)

    print(f"Results saved to {csv_file}")


def plot_results():
    """Plots the WER results from the CSV file, including average WER, and saves the plot."""
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "WER_results.csv"))
    data = {}
    avg_wer = {}

    # Read data from CSV file
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="|")
        next(reader)  # Skip header
        for row in reader:
            model, lang, sentence, wer_score = row
            if sentence == "Average":
                avg_wer[model] = float(wer_score)
            else:
                data.setdefault(model, []).append(float(wer_score))

    # Create the plot
    plt.figure(figsize=(10, 5))

    color_map = {}  # Store colors assigned to each model
    for model, scores in data.items():
        line, = plt.plot(range(1, len(scores) + 1), scores, marker="o", label=model)
        color_map[model] = line.get_color()  # Store assigned color

    # Add average WER points with the same color as the model's line
    for model, avg in avg_wer.items():
        plt.scatter(len(data[model]) + 1, avg, marker="*", s=150, label=f"{model} (Avg)", color=color_map[model], edgecolors="black")

    plt.xlabel("Sentence Number")
    plt.ylabel("WER Score")
    plt.legend()
    plt.title("Word Error Rate per Sentence with Average WER")
    plt.grid(True)

    # Save the plot
    plot_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "WER_plot.png"))
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {plot_filename}")

    plt.show()


if __name__ == "__main__":
    calculate_wer_for_models()
    plot_results()

        