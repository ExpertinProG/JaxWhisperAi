
# Whisper JAX Large Colab

This project leverages the Whisper JAX library to transcribe audio files using the Whisper large-v2 model on Google Colab. It includes steps to install necessary dependencies, download audio from YouTube, and perform transcription with timestamped outputs.

## Features

- **Transcription**: Converts audio files to text with timestamps.
- **YouTube Audio Download**: Downloads audio from YouTube videos.
- **Interactive Widgets**: Monitors download progress interactively.

## Getting Started

### Prerequisites

- A Google account to access Google Colab.
- Basic knowledge of Python and Google Colab.

### Installation

1. **Clone the Repository**: Clone this repository to your local machine or open it directly in Google Colab.

2. **Install Dependencies**: Run the following commands to install the necessary libraries:
    ```python
    !pip install -q transformers==4.31.0
    !pip install -U flax==0.7.2 "jax[cuda11_local]==0.4.13" "jaxlib[cuda11_local]==0.4.13" -f https://bing.com/search?q=
    !pip install -q git+https://github.com/camenduru/whisper-jax.git datasets soundfile librosa yt_dlp cached_property
    ```

### Usage

1. **Initialize Pipeline**: Initialize the Whisper JAX pipeline:
    ```python
    import jax
    from whisper_jax import FlaxWhisperPipline
    import jax.numpy as jnp

    pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.float16)
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.initialize_cache("/content/jax_cache")
    ```

2. **Download Audio from YouTube**: Use yt-dlp to download audio:
    ```python
    from yt_dlp import YoutubeDL
    with YoutubeDL({'overwrites': True, 'format': 'bestaudio[ext=m4a]', 'outtmpl': '/content/audio.m4a'}) as ydl:
        ydl.download("https://youtu.be/LXEAkeh7OTE")
    ```

3. **Transcribe Audio**: Transcribe the downloaded audio file:
    ```python
    outputs = pipeline("/content/audio.m4a", return_timestamps=True)
    text = outputs["text"]
    chunks = outputs["chunks"]

    print(text)
    print(chunks)
    ```

## License

This project is licensed under the MIT License.

---
