# FallacyScoreboard

FallacyScoreboard is an offline tool that downloads YouTube videos, processes them using WhisperX for automatic speech recognition and speaker diarization, detects logical fallacies using LLMs with OLLAMA, and overlays the results on the video with a scoreboard.

## Demo Video

[![FallacyScoreboard Demo](https://img.youtube.com/vi/1plEKbSHJMM/0.jpg)](https://youtu.be/1plEKbSHJMM)

Click the image above to watch a demonstration of FallacyScoreboard in action!

## Features

- Download YouTube videos and extract audio
- Transcribe audio using WhisperX with word-level timestamps and speaker diarization
- Detect logical fallacies in transcribed text
- Overlay fallacy detection results on the video with a scoreboard and side panel
- Handle multiple fallacies per speaker segment
- Color-code speakers for easy identification

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/latent-variable/FallacyScoreboard.git
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

4. **Install ImageMagick**:
   FallacyScoreboard uses MoviePy, which requires ImageMagick. Make sure to install ImageMagick and set the path in the script:
   ```python
   change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})
   ```

## Usage

1. **Process a YouTube video**:
    ```python
    from fallacyscoreboard import fallacy_detection_pipeline

    output_path = "path/to/output/directory"
    video_url = "https://www.youtube.com/watch?v=example"
    
    fallacy_detection_pipeline(video_url, output_path)
    ```

   This function will:
   - Download the YouTube video
   - Extract audio from the video
   - Transcribe the audio using WhisperX
   - Detect fallacies in the transcription
   - Overlay the fallacies and scoreboard on the video

2. **Customize OLLAMA settings**:
   You can customize the OLLAMA host and model by modifying these variables in the script:
   ```python
   OLLAMA_HOST = 'http://192.168.50.81:11434'
   OLLAMA_MODEL = 'llama3.1'
   ```

## Output

The script will generate several files in the output directory:
- The downloaded video file
- An extracted audio file
- A JSON file with the transcription and diarization results
- A text file with formatted transcription
- A JSON file with fallacy analysis results
- The final video with overlaid fallacy scoreboard and explanations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Notes

- Ensure you have sufficient disk space and computational resources, as processing videos can be resource-intensive.
- The quality of fallacy detection depends on the OLLAMA model used. Experiment with different models for best results.
- This tool is for educational and research purposes. Always respect copyright and terms of service when using YouTube content.
