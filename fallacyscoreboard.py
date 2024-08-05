import os
import json
import ollama
import yt_dlp
import random
import colorsys
import whisperx

from pathlib import Path
from moviepy.config import change_settings
from moviepy.editor import VideoFileClip, TextClip,  ColorClip, concatenate_videoclips
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip



# Set the path to the ImageMagick binary
IMAGEMAGIK = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGIK})

# Needed to download Hugging Face models Pyannote models for whisperx
YOUR_HF_TOKEN = None

# OLLAMA host & model
OLLAMA_HOST  = 'http://192.168.50.81:11434' #'http://localhost:11434'
OLLAMA_MODEL = 'llama3.1:70b' # 'llama3.1' 

def download_youtube_video(url, video_name):
    ydl_opts = {
        'format': 'bestvideo[height=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
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
        

def format_text(json_file, output_path):
    # load the json file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Process the segments to extract speaker labels and text
    formatted_text = []
    for segment in data['segments']:
        speaker = segment['speaker']
        start = segment["start"]
        end = segment["end"]
        text = segment['text'].strip()
        formatted_text.append(f"{start}-{end} {speaker}: {text}")

    # Join the segments into a single string with each segment on a new line
    formatted_text_str = "\n".join(formatted_text)

    # Save the formatted text to a new text file
    with open(output_path, 'w') as output_file:
        output_file.write(formatted_text_str)

    

def detect_fallacies(text_path, fallacy_analysis_path):
   
    # load the system_prompt from file
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirname, 'prompt_system.txt'), 'r') as file:
        system_prompt = file.read()
   
    
    # load task prompt from file 
    with open(os.path.join(dirname, 'prompt_task.txt'), 'r') as file:
        task_prompt = file.read()
    
    history = [{'role': 'system','content': f'{system_prompt}'}]
    client = ollama.Client(host=OLLAMA_HOST)
    
    llm_outputs = []
    for i, line in enumerate(read_line_from_file(text_path)):
        print(line)
        if i == 0:  
            history.append({'role': 'user', 'content': f'{task_prompt}\n{line}'})
        else: 
            history.append({'role': 'user', 'content': line})
            
        response = client.chat(model=OLLAMA_MODEL, messages=history)
        print(response['message']['content'])
        history.append({'role': 'assistant', 'content': response['message']['content']})
        llm_outputs.append(response['message']['content'])
    
    # print(response['message']['content'])
    save_llm_output_to_json(llm_outputs, fallacy_analysis_path)
    
    
def save_llm_output_to_json(llm_string_outputs, output_file):
    output_data = []
    for output_str in llm_string_outputs:
        try:
            # Parse the string output to a dictionary
            output_dict = json.loads(output_str)
            output_data.append(output_dict)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            print("Skipping output:", output_str)
    
    # Save to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(output_data, json_file, indent=2)
        
        
def read_line_from_file(file_path):
    
    # load data line by line
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()
    
def generate_pastel_color():
    h = random.random()
    s = 0.5 + random.random() * 0.5  # 0.5 to 1.0
    l = 0.7 + random.random() * 0.2  # 0.7 to 0.9
    r, g, b = [int(256*i) for i in colorsys.hls_to_rgb(h, l, s)]
    return f'#{r:02x}{g:02x}{b:02x}'

def overlay_fallacies_on_video(video_path, fallacy_results_file, final_video_name):
    try:
        video = VideoFileClip(str(video_path))
        original_width, original_height = video.size
        video_duration = video.duration

        # Calculate sizes based on video dimensions
        sidebar_width = int(original_width * 0.25)  # 25% of original width
        bottom_height = int(original_height * 0.2)  # 20% of original height
        new_width = original_width + sidebar_width
        new_height = original_height + bottom_height

        # Calculate font sizes based on video height
        main_font_size = max(int(original_height * 0.04), 12)  # 2% of height, minimum 12
        subtitle_font_size = max(int(original_height * 0.033), 10)  # 1.5% of height, minimum 10
        
        # Define the position for the explanation box (e.g., 30% down from the top of the video)
        explanation_top_position = int(original_height * 0.25)  

        background = ColorClip(size=(new_width, new_height), color=(0, 0, 0)).set_duration(video_duration)
        video = video.set_position((0, 0))

        with open(fallacy_results_file, 'r') as json_file:
            fallacy_results = json.load(json_file)

        fallacy_counters = {}
        speaker_colors = {}
        for result in fallacy_results:
            speaker = result["speaker"]
            if speaker not in fallacy_counters:
                fallacy_counters[speaker] = 0
                speaker_colors[speaker] = generate_pastel_color()

        def create_scoreboard(counters):
            scoreboard_text = "Fallacy Scoreboard\n" + "\n".join([f"{s}: {c}" for s, c in counters.items()])
            return TextClip(scoreboard_text, fontsize=main_font_size, color='white', bg_color='black', size=(sidebar_width, None), method='caption', align='center')

        scoreboard = create_scoreboard(fallacy_counters)
        scoreboard = scoreboard.set_position((original_width, 0)).set_duration(video_duration)

        text_clips = [scoreboard]
        current_fallacy = None

        for result in fallacy_results:
            start_time = max(0, min(result["start"], video_duration))
            end_time = max(start_time, min(result["end"], video_duration))
            speaker = result["speaker"]
            fallacies = result["fallacy_type"]
            reason = result["fallacy_explanation"]
            text_segment = result["text_segment"]

            if not isinstance(fallacies, list):
                fallacies = [fallacies]

            valid_fallacies = [f for f in fallacies if f not in [None, "None"]]

            if valid_fallacies:
                fallacy_counters[speaker] += len(valid_fallacies)
                new_scoreboard = create_scoreboard(fallacy_counters)
                new_scoreboard = new_scoreboard.set_position((original_width, 0)).set_start(start_time)
                text_clips.append(new_scoreboard)

                fallacy_text = f"Speaker: {speaker}\nFallacies: {', '.join(valid_fallacies)}\n\nExplanation:\n{reason}"
                
                # Create the fallacy box without specifying a fixed height
                fallacy_box = TextClip(fallacy_text, fontsize=subtitle_font_size, color=speaker_colors[speaker], 
                                    bg_color='black', method='caption', 
                                    size=(sidebar_width - 20, None),
                                    stroke_color='white', stroke_width=1)
                
                fallacy_composite = CompositeVideoClip([fallacy_box.set_opacity(0.8)], size=fallacy_box.size)
                
                # Set the position of the fallacy box lower on the screen
                fallacy_composite = fallacy_composite.set_position((original_width + 10, explanation_top_position)).set_start(start_time).set_end(end_time)
                
                text_clips.append(fallacy_composite)
                current_fallacy = fallacy_composite
            elif current_fallacy:
                extended_fallacy = current_fallacy.set_end(end_time)
                text_clips.append(extended_fallacy)

            subtitle = TextClip(text_segment, fontsize=subtitle_font_size, color=speaker_colors[speaker], bg_color='black', method='caption', size=(original_width, None))
            subtitle = subtitle.set_position((0, original_height)).set_start(start_time).set_end(end_time)
            text_clips.append(subtitle)

        # If there's remaining video duration, extend the last fallacy explanation
        if current_fallacy and current_fallacy.end < video_duration:
            final_extension = current_fallacy.set_end(video_duration)
            text_clips.append(final_extension)

        final_clip = CompositeVideoClip([background, video] + text_clips, size=(new_width, new_height))
        final_clip = final_clip.set_duration(video_duration)

        final_clip.write_videofile(final_video_name, codec="libx264", audio_codec="aac", threads=16, fps=24)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def fallacy_detection_pipeline(youtube_url, output_path):
    # get unique naem from the youtube video url link 
    name = video_url.split('=')[-1]

    # download the youtube video
    video_name = os.path.join(output_path, name + ".mp4")
    if not os.path.exists(video_name):
        print("Downloading video...")
        download_youtube_video(youtube_url, video_name)

    # extract the audio from the video file
    audio_name = os.path.join(output_path, name + ".wav" )
    if not os.path.exists(audio_name):
        print("Extracting audio...")
        extract_audio_from_video(video_name, audio_name)

    # transcribe the audio file using whisperx
    transcript_path = os.path.join(output_path, name + ".json")
    if not os.path.exists(transcript_path):
        print("Transcribing audio...")
        transcribe_audio_with_whisperx(audio_name, transcript_path)
        
    # format text
    text_path = os.path.join(output_path, name + ".txt")
    if not os.path.exists(text_path):
        print("Formatting text...")
        format_text(transcript_path, text_path)

        
    fallacy_analysis_path = os.path.join(output_path, name + "_fallacy_analysis.json")
    if not os.path.exists(fallacy_analysis_path):
        print("Detecting fallacies...")
        detect_fallacies(text_path,fallacy_analysis_path)
        
    #update video with fallacy analysis
    final_video_name = os.path.join(output_path, name + "_fallacy_analysis.mp4")
    if not os.path.exists(final_video_name):
        print("Overlaying fallacies on video...")
        overlay_fallacies_on_video(video_name, fallacy_analysis_path, final_video_name)

    print("Done!")
    
if __name__ == "__main__": 
    # Example usage
    output_path = os.path.abspath(Path("./Files"))
    video_url = "https://www.youtube.com/watch?v=LM-bjbeqaqE"
    # create output folder if it does not exist 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    fallacy_detection_pipeline(video_url, output_path)

