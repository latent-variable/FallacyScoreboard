# FallacyScoreboard

FallacyScoreboard is an offline tool that downloads YouTube videos, processes them using WhisperX for automatic speech recognition and speaker diarization, detects logical fallacies, and overlays the results on the video like a sporting event scoreboard.

## Features

- Download YouTube videos and extract audio
- Transcribe audio using WhisperX with word-level timestamps and speaker diarization
- Detect logical fallacies in transcribed text
- Overlay fallacy detection results on the video with a scoreboard and side panel

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/FallacyScoreboard.git
    cd FallacyScoreboard
    ```

2. **Create a Python environment**:
    ```bash
    conda create --name fallacyscoreboard python=3.10
    conda activate fallacyscoreboard
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Download and process a YouTube video**:
    ```python
    from fallacyscoreboard import download_youtube_video, extract_audio_from_video, transcribe_audio_with_whisperx, detect_fallacies, overlay_fallacies_on_video, upload_video

    output_path = Path("/path/to/save")
    video_url = "https://www.youtube.com/watch?v=example"
    video_path = download_youtube_video(video_url, output_path)
    audio_path = output_path / "audio.wav"
    extract_audio_from_video(video_path, audio_path)
    transcription_result = transcribe_audio_with_whisperx(audio_path)
    fallacy_results = detect_fallacies(transcription_result["segments"])
    overlay_fallacies_on_video(video_path, fallacy_results)
    upload_video("output_with_fallacies.mp4", "Title of Video", "Description of Video", "22", ["tag1", "tag2"])
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
