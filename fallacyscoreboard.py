import os
import json

import yt_dlp
import whisperx

from pathlib import Path
from moviepy.config import change_settings
from moviepy.editor import VideoFileClip
from multiprocessing import Process

from llm import detect_fallacies
from video_edit import overlay_fallacies_on_video, overlay_fallacies_on_vertical_video_with_bars
from dashboard import create_dashboard

# Set the path to the ImageMagick binary
IMAGEMAGIK = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGIK})

# Needed to download Hugging Face models Pyannote models for whisperx
YOUR_HF_TOKEN = None


def read_config(config_file="config.json"):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config["GIPHY_API_KEY"], config["YOUR_HF_TOKEN"]

GIPHY_API_KEY, YOUR_HF_TOKEN = read_config()

def download_youtube_video(url, video_name):
    ydl_opts = {
        'format': 'bestvideo[height=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best',
        'outtmpl': str(video_name),
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(video_name):
        raise FileNotFoundError("No MP4 files found in the output directory.")

def extract_audio_from_video(video_path, audio_path):
    video = VideoFileClip(str(video_path))
    video.audio.write_audiofile(audio_path)

def transcribe_audio_with_whisperx(audio_path, output_path):
    device = "cuda"
    model = whisperx.load_model("large-v3", device)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)

    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result_aligned)
    
    # write results to a text file 
    with open(os.path.join(output_path), 'w') as f:
        f.write(json.dumps(result, indent=2))

def format_text(json_file, output_path, min_length=2):
    # Load the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Process the segments to extract speaker labels and text
    formatted_text = []
    accumulated_text = ""
    accumulated_start = None
    previous_speaker = None

    for i, segment in enumerate(data['segments']):
        speaker = 'multiple'
        if 'speaker' in segment:
            speaker = segment['speaker']
        
        start = segment["start"]
        end = segment["end"]
        text = segment['text'].strip()
        segment_length = end - start

        # Check if the speaker is consistent
        if speaker == previous_speaker:
            # Accumulate text if the segment is too short
            if segment_length < min_length:
                if accumulated_start is None:
                    accumulated_start = start
                accumulated_text += " " + text
            else:
                # If there's accumulated text, append it to this segment
                if accumulated_text:
                    text = accumulated_text.strip() + " " + text
                    start = accumulated_start
                    accumulated_text = ""
                    accumulated_start = None
                formatted_text.append(f"{start}-{end} {speaker}: {text}")
        else:
            # If speakers don't line up, break and append the current segment
            if accumulated_text:
                # Append any accumulated text from the previous speaker
                formatted_text.append(f"{accumulated_start}-{end} {previous_speaker}: {accumulated_text.strip()}")
                accumulated_text = ""
                accumulated_start = None

            # Append the current segment directly
            formatted_text.append(f"{start}-{end} {speaker}: {text}")

        # Update the previous speaker
        previous_speaker = speaker

    # Handle any remaining accumulated text at the end
    if accumulated_text:
        formatted_text.append(f"{accumulated_start}-{end} {previous_speaker}: {accumulated_text.strip()}")

    # Join the segments into a single string with each segment on a new line
    formatted_text_str = "\n".join(formatted_text)

    # Save the formatted text to a new text file
    with open(output_path, 'w') as output_file:
        output_file.write(formatted_text_str)

# Define the processing functions at the top level
def process_regular_video(video_name, fallacy_analysis_path, final_video_name):
    if not os.path.exists(final_video_name):
        print("Overlaying fallacies on horizontal video...")
        overlay_fallacies_on_video(video_name, fallacy_analysis_path, final_video_name)
    else:
        print("Video already updated with fallacies.")

def process_vertical_video(video_name, fallacy_analysis_path, final_vertical_video_name):
    if not os.path.exists(final_vertical_video_name):
        print("Overlaying fallacies on vertical video...")
        overlay_fallacies_on_vertical_video_with_bars(video_name, fallacy_analysis_path, final_vertical_video_name)
    else:
        print("Vertical video already updated with fallacies.")

def fallacy_detection_pipeline(youtube_url, output_path, create_video=True):
    name = youtube_url.split('=')[-1]

    output_path = os.path.join(output_path, name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # download the youtube video
    video_name = os.path.join(output_path, name + ".mp4")
    if not os.path.exists(video_name):
        print("Downloading video...")
        download_youtube_video(youtube_url, video_name)
    else:
        print("Video already downloaded.")

    # extract the audio from the video file
    audio_name = os.path.join(output_path, name + ".wav" )
    if not os.path.exists(audio_name):
        print("Extracting audio...")
        extract_audio_from_video(video_name, audio_name)
    else:
        print("Audio already extracted.")

    # transcribe the audio file using whisperx
    transcript_path = os.path.join(output_path, name + ".json")
    if not os.path.exists(transcript_path):
        print("Transcribing audio...")
        transcribe_audio_with_whisperx(audio_name, transcript_path)
    else:
        print("Audio already transcribed.")
        
    # format text
    text_path = os.path.join(output_path, name + ".txt")
    if not os.path.exists(text_path):
        print("Formatting text...")
        format_text(transcript_path, text_path)
    else:
        print("Text already formatted.")

    fallacy_analysis_path = os.path.join(output_path, name + "_fallacy_analysis.json")
    temp_file = os.path.join(output_path, name + "_temp.json")
    if not os.path.exists(fallacy_analysis_path):
        print("Detecting fallacies...")
        detect_fallacies(text_path, fallacy_analysis_path, temp_file)
    else:
        print("Fallacies already detected.")
        
    dashboard_file = os.path.join(output_path, name + "_dashboard.html")
    if not os.path.exists(dashboard_file):
        print("Creating dashboard...")
        create_dashboard(fallacy_analysis_path, dashboard_file)
    else:
        print("Dashboard already created.")

    if create_video:
        # Prepare arguments for processing functions
        final_video_name = os.path.join(output_path, name + "_fallacy_analysis.mp4")
        final_vertical_video_name = os.path.join(output_path, name + "_vertical_fallacy_analysis.mp4")
        
        # Use multiprocessing to run both video processes in parallel
        regular_video_process = Process(target=process_regular_video, args=(video_name, fallacy_analysis_path, final_video_name))
        vertical_video_process = Process(target=process_vertical_video, args=(video_name, fallacy_analysis_path, final_vertical_video_name))

        # Start both processes
        regular_video_process.start()
        vertical_video_process.start()

        # Wait for both processes to complete
        regular_video_process.join()
        vertical_video_process.join()

    
    print("Done!")

if __name__ == "__main__": 
    output_path = os.path.abspath(Path("./Files"))
    video_url =  'https://www.youtube.com/watch?v=LM-bjbeqaqE' #"https://www.youtube.com/watch?v=jcsncEfRpgs" #
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    fallacy_detection_pipeline(video_url, output_path)
