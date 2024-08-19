import spaces
import torch
import gradio as gr
import yt_dlp as youtube_dl
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
from transformers import WhisperProcessor
from pyannote.audio import Pipeline as PyannotePipeline
from datetime import datetime
import tempfile
import numpy as np
from itertools import groupby
from datetime import datetime
from gradio.components import State
import os
import re


    
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:307'

try:
    diarization_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ["HF_TOKEN"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diarization_pipeline.to(device)
except Exception as e:
    print(f"Error initializing diarization pipeline: {e}")
    diarization_pipeline = None

# Updated Whisper model
MODEL_NAME = "openai/whisper-medium"
#BATCH_SIZE = 2  # Réduction de la taille du batch
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 3600

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    #chunk_length_s=30,
    device=device,
    model_kwargs={"low_cpu_mem_usage": True},
)




def associate_speakers_with_timestamps(transcription_result, diarization, tolerance=0.01, min_segment_duration=0.05):
    word_segments = transcription_result['chunks']
    diarization_segments = list(diarization.itertracks(yield_label=True))
    speaker_transcription = []
    current_speaker = None
    current_text = []
    unassigned_words = []
    last_segment_index = 0

    def flush_current_segment():
        nonlocal current_speaker, current_text
        if current_speaker and current_text:
            segment_duration = word_segments[-1]['timestamp'][1] - word_segments[0]['timestamp'][0]
            if segment_duration >= min_segment_duration:
                speaker_transcription.append((current_speaker, ' '.join(current_text)))
            else:
                unassigned_words.extend([(word['timestamp'][0], word['text']) for word in word_segments])
            current_text = []

    for word in word_segments:
        word_start, word_end = word['timestamp']
        word_text = word['text']
        assigned = False

        for i in range(last_segment_index, len(diarization_segments)):
            segment, _, speaker = diarization_segments[i]
            if segment.start - tolerance <= word_start < segment.end + tolerance:
                if speaker != current_speaker:
                    flush_current_segment()
                    current_speaker = speaker
                current_text.append(word_text)
                last_segment_index = i
                assigned = True
                break

        if not assigned:
            unassigned_words.append((word_start, word_text))

    flush_current_segment()

    # Traitement des mots non assignés
    unassigned_words.sort(key=lambda x: x[0])  # Trier par timestamp
    for word_start, word_text in unassigned_words:
        closest_segment = min(diarization_segments, key=lambda x: min(abs(x[0].start - word_start), abs(x[0].end - word_start)))
        speaker = closest_segment[2]
        if speaker != current_speaker:
            flush_current_segment()
            current_speaker = speaker
        current_text.append(word_text)
    flush_current_segment()

    # Fusion des segments courts
    merged_transcription = []
    for speaker, text in speaker_transcription:
        if not merged_transcription or merged_transcription[-1][0] != speaker:
            merged_transcription.append((speaker, text))
        else:
            merged_transcription[-1] = (speaker, merged_transcription[-1][1] + " " + text)

    return merged_transcription
    
def simplify_diarization_output(speaker_transcription):
    simplified = []
    for speaker, text in speaker_transcription:
        simplified.append(f"{speaker}: {text}")
    return "\n".join(simplified)

def parse_simplified_diarization(simplified_text):
    pattern = r"(SPEAKER_\d+):\s*(.*)"
    matches = re.findall(pattern, simplified_text, re.MULTILINE)
    return [(speaker, text.strip()) for speaker, text in matches]

def process_transcription(*args):
    generator = transcribe_and_diarize(*args)
    for progress_message, raw_text, speaker_transcription in generator:
        pass  # Consommer le générateur jusqu'à la fin
    simplified_diarization = simplify_diarization_output(speaker_transcription)
    return progress_message, raw_text, simplified_diarization

def process_yt_transcription(*args):
    html_embed, raw_text, speaker_transcription = yt_transcribe(*args)
    simplified_diarization = simplify_diarization_output(speaker_transcription)
    return html_embed, raw_text, simplified_diarization
    

# New functions for progress indicator
def create_progress_indicator():
    return gr.State({"stage": 0, "message": "En attente de démarrage..."})

def update_progress(progress_state, stage, message):
    progress_state["stage"] = stage
    progress_state["message"] = message
    return progress_state

def display_progress(progress_state):
    stages = [
        "Chargement du fichier",
        "Préparation de l'audio",
        "Transcription en cours",
        "Diarisation (identification des locuteurs)",
        "Finalisation des résultats"
    ]
    progress = (progress_state["stage"] / len(stages)) * 100
    return gr.HTML(f"""
        <style>
            @keyframes pulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(66, 133, 244, 0.7); }}
                70% {{ box-shadow: 0 0 0 10px rgba(66, 133, 244, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(66, 133, 244, 0); }}
            }}
        </style>
        <div style="margin-bottom: 20px; background-color: #f0f0f0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="margin-bottom: 10px; color: #333; font-weight: bold;">Progression de la transcription</h4>
            <div style="background-color: #e0e0e0; height: 24px; border-radius: 12px; overflow: hidden; position: relative;">
                <div style="background-color: #4285F4; width: {progress}%; height: 100%; border-radius: 12px; transition: width 0.5s ease-in-out;"></div>
                <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; align-items: center; justify-content: center; color: #333; font-weight: bold;">
                    {progress:.0f}%
                </div>
            </div>
            <p style="margin-top: 10px; color: #666; font-style: italic;">{progress_state["message"]}</p>
            <div style="width: 10px; height: 10px; background-color: #4285F4; border-radius: 50%; margin-top: 10px; animation: pulse 2s infinite;"></div>
        </div>
    """)
    
@spaces.GPU(duration=120)
def transcribe_and_diarize(file_path, task, progress=gr.Progress()):
    progress(0, desc="Initialisation...")
    yield "Chargement du fichier...", None, None

    progress(0.2, desc="Préparation de l'audio...")
    yield "Préparation de l'audio...", None, None

    progress(0.4, desc="Laissez moi quelques minutes pour déchiffrer les voix et rédiger l'audio 🤓 ✍️ ...")
    transcription_result = pipe(file_path, generate_kwargs={"task": task, "language": "fr"}, return_timestamps="word")
    yield "Transcription en cours...", None, None

    progress(0.6, desc=" C'est fait 😮‍💨 ! Je m'active à fusionner tout ça, un instant, J'y suis presque...")
    if diarization_pipeline:
        diarization = diarization_pipeline(file_path)
        speaker_transcription = associate_speakers_with_timestamps(transcription_result, diarization)
    else:
        speaker_transcription = [(None, transcription_result['text'])]
    yield "Diarisation en cours...", None, None

    progress(0.8, desc="Finalisation des résultats...")
    yield "Voilà!", transcription_result['text'], speaker_transcription

    progress(1.0, desc="Terminé!")
    return "Transcription terminée!", transcription_result['text'], speaker_transcription

def format_to_markdown(transcription_text, speaker_transcription, audio_duration=None, location=None, speaker_age=None, context=None):
    metadata = {
        "Date de traitement": datetime.now().strftime('%d/%m/%Y %H:%M'),
        "Durée de l'audio": f"{audio_duration} secondes" if audio_duration else "[à remplir]",
        "Lieu": location if location else "[non spécifié]",
        "Âge de l'intervenant": f"{speaker_age} ans" if speaker_age else "[non spécifié]",
        "Contexte": context if context else "[non spécifié]"
    }
    
    metadata_text = "\n".join([f"- **{key}** : '{value}'" for key, value in metadata.items()])
    
    try:
        if isinstance(speaker_transcription, str):
            speaker_transcription = parse_simplified_diarization(speaker_transcription)
        
        if isinstance(speaker_transcription, list) and all(isinstance(item, tuple) and len(item) == 2 for item in speaker_transcription):
            formatted_transcription = []
            for speaker, text in speaker_transcription:
                formatted_transcription.append(f"**{speaker}**: {text}")
            transcription_text = "\n\n".join(formatted_transcription)
        else:
            raise ValueError("Invalid speaker transcription format")
    except Exception as e:
        print(f"Error formatting speaker transcription: {e}")
        transcription_text = "Error formatting speaker transcription. Using raw transcription instead.\n\n" + transcription_text

    formatted_output = f"""
# Transcription Formatée

## Métadonnées
{metadata_text}

## Transcription
{transcription_text}
"""
    return formatted_output

def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str

def download_yt_audio(yt_url, filename):
    info_loader = youtube_dl.YoutubeDL()
    
    try:
        info = info_loader.extract_info(yt_url, download=False)
    except youtube_dl.utils.DownloadError as err:
        raise gr.Error(str(err))
    
    file_length = info["duration"]
    
    if file_length > YT_LENGTH_LIMIT_S:
        yt_length_limit_hms = time.strftime("%H:%M:%S", time.gmtime(YT_LENGTH_LIMIT_S))
        file_length_hms = time.strftime("%H:%M:%S", time.gmtime(file_length))
        raise gr.Error(f"La durée maximale YouTube est de {yt_length_limit_hms}, la vidéo YouTube dure {file_length_hms}.")
    
    ydl_opts = {"outtmpl": filename, "format": "bestaudio/best"}
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
        except youtube_dl.utils.ExtractorError as err:
            raise gr.Error(str(err))

@spaces.GPU(duration=120)
def yt_transcribe(yt_url, task):
    html_embed_str = _return_yt_html_embed(yt_url)

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "video.mp4")
        download_yt_audio(yt_url, filepath)
        with open(filepath, "rb") as f:
            inputs = f.read()

    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    transcription_result = pipe(inputs, generate_kwargs={"task": task}, return_timestamps=True)
    transcription_text = transcription_result['text']

    if diarization_pipeline:
        diarization = diarization_pipeline(filepath)
        speaker_transcription = associate_speakers_with_timestamps(transcription_result, diarization)
    else:
        speaker_transcription = [(None, transcription_text)]

    return html_embed_str, transcription_text, speaker_transcription

def create_info_box(title, content):
    return gr.Markdown(f"### {title}\n{content}")

css_file_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_file_path, "r") as f:
    custom_css = f.read()

theme = gr.themes.Default().set(
    body_background_fill="#f0f2f5",
    body_background_fill_dark="#2c3e50",
    button_primary_background_fill="#3498db",
    button_primary_background_fill_dark="#2980b9",
    button_secondary_background_fill="#2ecc71",
    button_secondary_background_fill_dark="#27ae60",
)

demo = gr.Blocks(
    theme=gr.themes.Default(),
    title="Scribe - Assistant de Transcription Audio 🎙️📝",
    css=custom_css
)


with demo:
    gr.Markdown("""# 🎙️ **Scribe** : L'assistant de Transcription Audio Intelligent 📝 
    ### ⚠️ Cette version est une maquette publique. Ne pas mettre de données sensibles, privées ou confidentielles. ⚠️""")
    gr.HTML(
        """
        <div class="logo">
            <img src="https://image.noelshack.com/fichiers/2024/33/4/1723713257-dbe58773-0638-445b-a88c-3fc1f2002408.jpg" alt="Scribe Logo">
        </div>
        """
    )
    gr.Markdown("## **Bienvenue sur Scribe, une solution pour la transcription audio sécurisée. Transformez efficacement vos fichiers audio, enregistrements en direct ou vidéos YouTube en texte précis.**")

    gr.Markdown("""
    ### 🔍 **Fonctionnement du Modèle** :
    Scribe utilise une approche en deux étapes pour transformer l'audio en texte structuré :

    1. **Transcription avec Whisper Medium** :
       - Modèle de reconnaissance vocale développé par OpenAI
       - Utilise un réseau neuronal encodeur-décodeur avec attention
       - Capable de traiter divers accents et bruits de fond
       - Optimisé pour un équilibre entre précision et rapidité

    2. **Diarisation avec pyannote/speaker-diarization-3.1** :
       - Identifie et segmente les différents locuteurs dans l'audio
       - Utilise des techniques d'apprentissage profond pour l'extraction de caractéristiques vocales
       - Applique un algorithme de clustering pour regrouper les segments par locuteur



    ### 💡 **Conseils pour de Meilleurs Résultats**
    - Utilisez des enregistrements de haute qualité avec peu de bruit de fond.
    - Pour les longs enregistrements, il est recommandé de segmenter votre audio.
    - Vérifiez toujours la transcription, en particulier pour les termes techniques ou les noms propres.
    - Utilisez des microphones externes pour les enregistrements en direct si possible.

    ### ⚙️ Spécifications Techniques :
    - Modèle de transcription : Whisper Medium
    - Pipeline de diarisation : pyannote/speaker-diarization-3.1
    - Limite de taille de fichier : _(Nous n'avons, à ce jour, pas de limite précise. Cependant, **nous vous recommandons de ne pas dépasser 5 minutes.** )_
    - Durée maximale pour les vidéos YouTube : _(Nous n'avons, à ce jour, pas de limite précise. Cependant, pour une utilisation optimale, l'audio ne doit pas dépasser 30 minutes. )_
    - Formats audio supportés : MP3, WAV, M4A, et plus
    """)
    with gr.Accordion("🔐 Sécurité des Données et Pipelines", open=False):
        gr.Markdown("""

    #### Qu'est-ce qu'une pipeline ?
    Une pipeline dans le contexte de l'apprentissage automatique est une série d'étapes de traitement des données, allant de l'entrée brute à la sortie finale. Dans Scribe, nous utilisons deux pipelines principales :

    1. **Pipeline de Transcription** : Basée sur le modèle Whisper Medium, elle convertit l'audio en texte.
    2. **Pipeline de Diarisation** : Identifie les différents locuteurs dans l'audio.

    #### Comment fonctionnent nos pipelines ?
    1. **Chargement Local** : Les modèles sont chargés localement sur votre machine ou serveur.
    2. **Traitement In-Situ** : Toutes les données sont traitées sur place, sans envoi à des serveurs externes.
    3. **Mémoire Volatile** : Les données sont stockées temporairement en mémoire vive et effacées après utilisation.

    #### Sécurité et Confidentialité
    - **Pas de Transmission Externe** : Vos données audio et texte restent sur votre système local.
    - **Isolation** : Chaque session utilisateur est isolée des autres.
    - **Nettoyage Automatique** : Les fichiers temporaires sont supprimés après chaque utilisation.
    - **Mise à Jour Sécurisée** : Les modèles sont mis à jour de manière sécurisée via Hugging Face.

    #### Mesures de Sécurité Supplémentaires
    - Nous utilisons des tokens d'authentification sécurisés pour accéder aux modèles.
    - Les fichiers YouTube sont téléchargés et traités localement, sans stockage permanent.
    - Aucune donnée utilisateur n'est conservée après la fermeture de la session.

    En utilisant Scribe, vous bénéficiez d'un traitement de données hautement sécurisé et respectueux de la vie privée, tout en profitant de la puissance des modèles d'IA de pointe.
    """)
    # ... (le reste du fichier reste inchangé)    
    with gr.Tabs():
        with gr.Tab("Fichier audio 📁"):
            gr.Markdown("### 📂 Transcription de fichiers audio")
            audio_input = gr.Audio(type="filepath", label="Chargez votre fichier audio")
            task_input = gr.Radio(["transcribe", "translate"], label="Choisissez la tâche", value="transcribe")
            transcribe_button = gr.Button("🚀 Lancer la transcription", elem_classes="button-primary")
            
            progress_display = gr.Markdown(label="État de la progression")
            
            with gr.Accordion("Résultats 📊", open=True):
                raw_output = gr.Textbox(label="📝 Transcription brute", info="Texte généré par le modèle. Modifiable si nécessaire.")
                speaker_output = gr.Textbox(label="👥 Diarisation (format simplifié)", info="Identification des locuteurs. Format : 'SPEAKER_XX: texte'")
            with gr.Accordion("Métadonnées (optionnel) 📌", open=False):
                audio_duration = gr.Textbox(label="⏱️ Durée de l'audio (mm:ss)")
                location = gr.Textbox(label="📍 Lieu de l'enregistrement")
                speaker_age = gr.Number(label="👤 Âge de l'intervenant principal")
                context = gr.Textbox(label="📝 Contexte de l'enregistrement")
            
            format_button = gr.Button("✨ Générer la transcription formatée", elem_classes="button-secondary")
            formatted_output = gr.Markdown(label="📄 Transcription formatée :")


        with gr.Tab("Microphone 🎤"):
            gr.Markdown("### 🗣️ Enregistrement et transcription en direct")
            mic_input = gr.Audio(type="filepath", label="Enregistrez votre voix")
            mic_task_input = gr.Radio(["transcribe", "translate"], label="Choisissez la tâche", value="transcribe")
            mic_transcribe_button = gr.Button("🚀 Transcrire l'enregistrement", elem_classes="button-primary")
            
            mic_progress_display = gr.Markdown(label="État de la progression")
            
            with gr.Accordion("Résultats 📊", open=True):
                mic_raw_output = gr.Textbox(label="📝 Transcription brute", info="Texte généré par le modèle. Modifiable si nécessaire.")
                mic_speaker_output = gr.Textbox(label="👥 Diarisation (format simplifié)", info="Identification des locuteurs. Format : 'SPEAKER_XX: texte'")
            with gr.Accordion("Métadonnées (optionnel) 📌", open=False):
                mic_audio_duration = gr.Textbox(label="⏱️ Durée de l'enregistrement (mm:ss)")
                mic_location = gr.Textbox(label="📍 Lieu de l'enregistrement")
                mic_speaker_age = gr.Number(label="👤 Âge de l'intervenant principal")
                mic_context = gr.Textbox(label="📝 Contexte de l'enregistrement")
            
            mic_format_button = gr.Button("✨ Générer la transcription formatée", elem_classes="button-secondary")
            mic_formatted_output = gr.Markdown(label="📄 Transcription formatée :")
            
        with gr.Tab("YouTube 🎥"):
            gr.Markdown("### 🌐 Transcription à partir de vidéos YouTube")
            yt_input = gr.Textbox(lines=1, placeholder="Collez l'URL d'une vidéo YouTube ici", label="🔗 URL YouTube")
            yt_task_input = gr.Radio(["transcribe", "translate"], label="Choisissez la tâche", value="transcribe")
            yt_transcribe_button = gr.Button("🚀 Transcrire la vidéo", elem_classes="button-primary")
            
            yt_progress_display = gr.Markdown(label="État de la progression")
            
            yt_html_output = gr.HTML(label="▶️ Aperçu de la vidéo")
            
            with gr.Accordion("Résultats 📊", open=True):
                yt_raw_output = gr.Textbox(label="📝 Transcription brute", info="Texte généré par le modèle. Modifiable si nécessaire.")
                yt_speaker_output = gr.Textbox(label="👥 Diarisation (format simplifié)", info="Identification des locuteurs. Format : 'SPEAKER_XX: texte'")
            with gr.Accordion("Métadonnées (optionnel) 📌", open=False):
                yt_audio_duration = gr.Textbox(label="⏱️ Durée de la vidéo (mm:ss)")
                yt_channel = gr.Textbox(label="📺 Nom de la chaîne YouTube")
                yt_publish_date = gr.Textbox(label="📅 Date de publication")
                yt_context = gr.Textbox(label="📝 Contexte de la vidéo")
            
            yt_format_button = gr.Button("✨ Générer la transcription formatée", elem_classes="button-secondary")
            yt_formatted_output = gr.Markdown(label="📄 Transcription formatée :")


    gr.Markdown("""### 🛠️ Capacités :
    - Transcription multilingue avec détection automatique de la langue
    - Traduction vers le français (pour les contenus non francophones)
    - Identification précise des changements de locuteurs
    - Traitement de fichiers audio, enregistrements en direct et vidéos YouTube
    - Gestion de divers formats audio et qualités d'enregistrement

    """)
        
    with gr.Accordion("❓ README :", open=False):
        gr.Markdown("""
    - Concepteur : Woziii
    - Modèles :
        - [Whisper-médium](https://huggingface.co/openai/whisper-medium) : Model size - 764M params - Tensor type F32 -
        - [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) : Model size - Unknow - Tensor type F32 -
    - Version : V.2.0.0-Bêta
    - Langues : FR, EN
    - Copyright : cc-by-nc-4.0
    - [En savoir +](https://huggingface.co/spaces/Woziii/scribe/blob/main/README.md)
    """)

    # Connexions des boutons aux fonctions appropriées
    transcribe_button.click(
    process_transcription,
    inputs=[audio_input, task_input],
    outputs=[progress_display, raw_output, speaker_output]
    )

    format_button.click(
        format_to_markdown,
        inputs=[raw_output, speaker_output, audio_duration, location, speaker_age, context],
        outputs=formatted_output
    )

    mic_transcribe_button.click(
    process_transcription,
    inputs=[mic_input, mic_task_input],
    outputs=[mic_progress_display, mic_raw_output, mic_speaker_output]
    )

    mic_format_button.click(
        format_to_markdown,
        inputs=[mic_raw_output, mic_speaker_output, audio_duration, location, speaker_age, context],
        outputs=mic_formatted_output
    )

    yt_transcribe_button.click(
    process_yt_transcription,
    inputs=[yt_input, yt_task_input],
    outputs=[yt_html_output, yt_raw_output, yt_speaker_output]
    )

    yt_format_button.click(
        format_to_markdown,
        inputs=[yt_raw_output, yt_speaker_output, audio_duration, location, speaker_age, context],
        outputs=yt_formatted_output
    )
    

if __name__ == "__main__":
    demo.queue().launch()
