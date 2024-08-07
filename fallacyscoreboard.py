import os
import json
import time
import ollama
import yt_dlp
import random
import colorsys
import whisperx

from pathlib import Path
from moviepy.config import change_settings
from moviepy.editor import VideoFileClip, TextClip,  ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

# Set the path to the ImageMagick binary
IMAGEMAGIK = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGIK})

# Needed to download Hugging Face models Pyannote models for whisperx
YOUR_HF_TOKEN = None

# OLLAMA host & model
OLLAMA_HOST  = 'http://192.168.50.81:11434' #'http://localhost:11434'
OLLAMA_MODEL =  'lucas2024/gemma-2-9b-it-sppo-iter3:q8_0' #'llama3.1' #'gemma2:27b' #'llama3.1:70b' 
OLLAMA_CONTEXT_SIZE = 4_096 # Max context size for OLLAMA is 

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
        speaker = 'multiple'
        if 'speaker' in segment:
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

def initialize_history():
    # Load the system_prompt from file
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirname, 'prompt_system.txt'), 'r') as file:
        system_prompt = file.read()

    history = [{'role': 'system', 'content': f'{system_prompt}'}]
    
    return history

def detect_fallacies(text_path, fallacy_analysis_path, temp_file):
    history = initialize_history()
    client = ollama.Client(host=OLLAMA_HOST)
    
    llm_outputs = []
    
    # Load from temp file if it exists
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            temp_data = json.load(f)
            history = temp_data.get('history', history)
            llm_outputs = temp_data.get('llm_outputs', [])
            processed_lines = temp_data.get('processed_lines', [])
    else:
        processed_lines = []

    for line in read_line_from_file(text_path):
        if line in processed_lines:
            continue
        history, llm_outputs = prompt_ollama(line, history, llm_outputs, client)
        processed_lines.append(line)
        save_intermediate_results(temp_file, history, llm_outputs, processed_lines)
    
    save_llm_output_to_json(llm_outputs, fallacy_analysis_path)
    # Remove the temp file after completion
    if os.path.exists(temp_file):
        os.remove(temp_file)

def prompt_ollama(line, history, llm_outputs, client, pre_prompt='Input Text:'):
    max_retries = 3
    retry_count = 0
    valid_output = False
    
    # Extract metadata from the line
    try:
        start = float(line.split()[0].split('-')[0])
        end = float(line.split()[0].split('-')[1])
        speaker = line.split()[1].replace(':', '')
    except (IndexError, ValueError):
        start, end, speaker = 0, 0, 'SPEAKER_00'
        
    while not valid_output and retry_count < max_retries:
        history.append({'role': 'user', 'content': f'{pre_prompt} {line}'})
        
        # Start timing
        start_time = time.time()
        
        response = client.chat(model=OLLAMA_MODEL, messages=history, options={'temperature': 0.5, 'num_ctx': OLLAMA_CONTEXT_SIZE})
        
        # End timing
        end_time = time.time()
        duration = end_time - start_time
        
        token_count = response['eval_count'] + response['prompt_eval_count']
        print(f'token_count: {token_count}, duration: {duration:.2f} seconds')
        
        llm_response = response['message']['content']

        # Ensure the LLM response is properly formatted by stripping to the JSON content
        json_response = extract_json_from_text(llm_response)

        if json_response:
            llm_response = json_response
            actual_text_segment = extract_text_segment(line)
            corrected_response = correct_llm_output(llm_response, actual_text_segment, start, end, speaker)
            valid_output, error_message = validate_llm_output(corrected_response, actual_text_segment)
            if valid_output:
                history.append({'role': 'assistant', 'content': json.dumps(llm_response)})
                llm_outputs.append(llm_response)  # Append the JSON object directly
                
                # Check if the token count exceeds and reset the context history if necessary
                if token_count > OLLAMA_CONTEXT_SIZE * 0.9:
                    history = initialize_history()
                    history.append({'role': 'user', 'content': f'{pre_prompt} {line}'})
                    history.append({'role': 'assistant', 'content': json.dumps(llm_response)})
                    
            else:
                retry_count += 1
                print(f"Invalid format for response: {llm_response}")
                print(f"Error: {error_message}")
                history.append({'role': 'user', 'content': f"The previous response was invalid because: {error_message}. Please correct it."})
        else:
            retry_count += 1
            print(f"Response is not properly formatted JSON: {llm_response}")
            history.append({'role': 'user', 'content': "The previous response was not properly formatted JSON. Please correct it."})
            
    if not valid_output:
        print(f"Failed to get a valid response after {max_retries} attempts")
        fallback_response = create_fallback_response(line, start, end, speaker)
        llm_outputs.append(fallback_response)

    return history, llm_outputs

def extract_json_from_text(text):
    try:
        # Find the first '{' and the last '}' to ensure we are only taking the JSON content
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_content = text[start:end + 1]
            return json.loads(json_content)
        else:
            return None
    except json.JSONDecodeError:
        return None

def correct_llm_output(response, text_segment, start, end, speaker):
    response['text_segment'] = text_segment
    response['start'] = start
    response['end'] = end
    response['speaker'] = speaker
    return response

def create_fallback_response(line, start, end, speaker):
    return {
        'text_segment': line,
        'fallacy_explanation': "NA",
        'fallacy_type': ["NA"],
        'speaker': speaker,
        'start': start,
        'end': end
    }

def extract_text_segment(line):
    # Extract the text segment by splitting on the first colon and taking the second part
    parts = line.split(':', 1)
    if len(parts) > 1:
        return parts[1].strip()
    return line.strip()

def validate_llm_output(output, input_text_segment):
    try:
        data = output
        required_fields = ["text_segment", "fallacy_explanation", "fallacy_type", "speaker", "start", "end"]
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data:
                return False, f"Missing field: {field}"

        # Check if fallacy_type is a list
        if not isinstance(data["fallacy_type"], list):
            return False, "fallacy_type should be a list"

        # Validate data types of fields
        if not isinstance(data["text_segment"], str):
            return False, "text_segment should be a string"
        if not isinstance(data["fallacy_explanation"], str):
            return False, "fallacy_explanation should be a string"
        if not isinstance(data["speaker"], str):
            return False, "speaker should be a string"
        if not isinstance(data["start"], (int, float)):
            return False, "start should be a number"
        if not isinstance(data["end"], (int, float)):
            return False, "end should be a number"

        # Check if text_segment matches the input text segment content
        if data["text_segment"].strip() != input_text_segment.strip():
            return False, "text_segment does not match the input text segment content"

        return True, ""
    except (json.JSONDecodeError, TypeError):
        return False, "Invalid JSON format"

def read_line_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

def save_llm_output_to_json(llm_outputs, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(llm_outputs, json_file, indent=2)

def save_intermediate_results(temp_file, history, llm_outputs, processed_lines):
    temp_data = {
        'history': history,
        'llm_outputs': llm_outputs,
        'processed_lines': processed_lines
    }
    with open(temp_file, 'w') as f:
        json.dump(temp_data, f, indent=2)

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
    name = youtube_url.split('=')[-1]

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
        
    #update video with fallacy analysis
    final_video_name = os.path.join(output_path, name + "_fallacy_analysis.mp4")
    if not os.path.exists(final_video_name):
        print("Overlaying fallacies on video...")
        overlay_fallacies_on_video(video_name, fallacy_analysis_path, final_video_name)
    else:
        print("Video already updated with fallacies.")

    print("Done!")

if __name__ == "__main__": 
    output_path = os.path.abspath(Path("./Files"))
    video_url = "https://www.youtube.com/watch?v=Z3eCCbVr3EU"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    fallacy_detection_pipeline(video_url, output_path)
