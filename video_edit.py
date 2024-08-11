import os 
import json
import random
import requests
import colorsys
from contextlib import contextmanager
from moviepy.video.fx.all import crop
from moviepy.editor import VideoFileClip, TextClip,  ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

def read_config(config_file="config.json"):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config["GIPHY_API_KEY"], config["YOUR_HF_TOKEN"]


GIPHY_API_KEY, YOUR_HF_TOKEN = read_config()

def search_and_download_gif(query, output_path, limit = 5):
    if query is None or output_path is None or query == "" or output_path == "":
        return None
    
    url = f"https://api.giphy.com/v1/gifs/search"
    params = {
        "api_key": GIPHY_API_KEY,
        "q": query,
        "limit": limit  # Get limit number of options to choose from
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if data["data"]:
        selected_gif = random.choice(data["data"])
        gif_url = selected_gif["images"]["original"]["url"]
        gif_response = requests.get(gif_url)
        
        with open(output_path, 'wb') as f:
            f.write(gif_response.content)
        
        return output_path
    else:
        print(f"No GIF found for the query: {query}")
        return None

def generate_pastel_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        saturation = 0.4 + random.random() * 0.2  # Keep saturation low for pastel colors
        lightness = 0.6 + random.random() * 0.2  # Keep lightness high for pastel colors
        r, g, b = [int(256 * channel) for channel in colorsys.hls_to_rgb(hue, lightness, saturation)]
        colors.append(f'#{r:02x}{g:02x}{b:02x}')
    random.shuffle(colors)  # Shuffle to randomize the order
    return colors


@contextmanager
def close_clip(clip):
    try:
        yield clip
    finally:
        if hasattr(clip, 'reader'):
            clip.reader.close()
        if hasattr(clip, 'audio') and hasattr(clip.audio, 'reader'):
            clip.audio.reader.close_proc()
        clip.close()
        
def add_gif_to_video(original_height, new_height, new_width, gif_path, start_time, end_time, sidebar_width):

    gif = VideoFileClip(gif_path)
    # Resize GIF to fit within the video frame if necessary
    gif = gif.resize(height=original_height // 4.5)
    
    # Calculate position for the center of the black vertical panel on the right side
    position = (new_width - sidebar_width + (sidebar_width - gif.size[0]) // 2, (new_height - gif.size[1]) )
    
    # Calculate the duration for the GIF to play
    duration = end_time - start_time
    
    # Loop the GIF to match the desired duration
    gif = gif.loop(duration=duration).set_start(start_time).set_position(position)
    
    return gif


def overlay_fallacies_on_video(video_path, fallacy_results_file, final_video_name):
    try:
        video = VideoFileClip(str(video_path))
        original_width, original_height = video.size
        video_duration = video.duration

        # Sizes based on video dimensions
        sidebar_width = int(original_width * 0.25)
        bottom_height = int(original_height * 0.15)
        new_width = original_width + sidebar_width
        new_height = original_height + bottom_height

        # Font sizes based on video height
        main_font_size = max(int(original_height * 0.04), 12)
        subtitle_font_size = max(int(original_height * 0.033), 10)
        explanation_top_position = int(original_height * 0.25)

        background = ColorClip(size=(new_width, new_height), color=(0, 0, 0)).set_duration(video_duration)
        video = video.set_position((0, 0))

        with open(fallacy_results_file, 'r') as json_file:
            fallacy_results = json.load(json_file)

        # Assignment of colors based on the number of unique speakers
        speakers = [result["speaker"] for result in fallacy_results if 'SPEAKER' in result["speaker"]]
        unique_speakers = list(set(speakers))
        speaker_colors = dict(zip(unique_speakers, generate_pastel_colors(len(unique_speakers))))
        
        
        fallacy_counters = {}
        start_end_times = []  # Store start and end times
        for result in fallacy_results:
            speaker = result["speaker"]
            if speaker not in fallacy_counters and 'SPEAKER' in speaker:
                fallacy_counters[speaker] = 0

            start_time = max(0, min(result["start"], video_duration))
            end_time = max(start_time, min(result["end"], video_duration))
            start_end_times.append((start_time, end_time))

        def create_scoreboard(counters):
            scoreboard_text = "Fallacy Scoreboard\n" + "\n".join([f"{s}: {c}" for s, c in counters.items()])
            return TextClip(scoreboard_text, fontsize=main_font_size, color='white', bg_color='black', size=(sidebar_width, None), method='caption', align='West', stroke_color='white', stroke_width=1.0)

        scoreboard = create_scoreboard(fallacy_counters)
        scoreboard = scoreboard.set_position((original_width, 0)).set_duration(video_duration)

        text_clips = [scoreboard]
        gif_clips = []
        current_fallacy = None

        for i, result in enumerate(fallacy_results):
            start_time = max(0, min(result["start"], video_duration))
            end_time = max(start_time, min(result["end"], video_duration))
            speaker = result["speaker"]
            fallacies = result["fallacy_type"]
            reason = result["fallacy_explanation"]
            text_segment = result["text_segment"]
            gif_query = result["gif_query"]

            if not isinstance(fallacies, list):
                fallacies = [fallacies]

            valid_fallacies = [f for f in fallacies if f not in [None, "None"]]

            if valid_fallacies and 'SPEAKER' in speaker:
                fallacy_counters[speaker] += len(valid_fallacies)
                new_scoreboard = create_scoreboard(fallacy_counters)
                new_scoreboard = new_scoreboard.set_position((original_width, 0)).set_start(start_time)
                text_clips.append(new_scoreboard)

                fallacy_text = f"Speaker: {speaker}\nFallacies: {', '.join(valid_fallacies)}\n\nExplanation:\n{reason}"
                
                fallacy_box = TextClip(fallacy_text, fontsize=subtitle_font_size, color=speaker_colors[speaker], bg_color='black', method='caption', align='West', size=(sidebar_width - 20, None))
                
                fallacy_composite = CompositeVideoClip([fallacy_box.set_opacity(0.8)], size=fallacy_box.size)
                fallacy_composite = fallacy_composite.set_position((original_width + 10, explanation_top_position)).set_start(start_time).set_end(end_time)
                
                text_clips.append(fallacy_composite)
                current_fallacy = fallacy_composite
            elif current_fallacy:
                extended_fallacy = current_fallacy.set_end(end_time)
                text_clips.append(extended_fallacy)
                
            gif_dir = os.path.join(os.path.dirname(os.path.dirname(final_video_name)), 'gifs')
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            gif_path = os.path.join(gif_dir, f"{gif_query}.gif")
            if not os.path.exists(gif_path):
                gif_path = search_and_download_gif(gif_query, gif_path, limit=1)
                
            if gif_path: # skip if just added the gif
                gif_clip = add_gif_to_video(original_height, new_height, new_width, gif_path, start_time, end_time, sidebar_width)
                gif_clips.append(gif_clip)
                
            else:
                gif_clips.append(None)
                
            subtitle = TextClip(text_segment, fontsize=subtitle_font_size, color=speaker_colors[speaker], bg_color='black', method='caption', size=(original_width, None))
            subtitle = subtitle.set_position((0, original_height)).set_start(start_time).set_end(end_time)
            text_clips.append(subtitle)

        if current_fallacy and current_fallacy.end < video_duration:
            final_extension = current_fallacy.set_end(video_duration)
            text_clips.append(final_extension)

        final_clip = CompositeVideoClip([background, video] + text_clips + [gif for gif in gif_clips if gif], size=(new_width, new_height))
        final_clip = final_clip.set_duration(video_duration)

        final_clip.write_videofile(final_video_name, codec="libx264", audio_codec="aac", threads=8, fps=24, ffmpeg_params=['-c:v', 'h264_nvenc','-preset', 'fast'])

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
        
def overlay_fallacies_on_vertical_video_with_bars(video_path, fallacy_results_file, final_video_name):
    try:
        video = VideoFileClip(str(video_path))
        original_width, original_height = video.size
        video_duration = video.duration

        # Target sizes for vertical format with black bars
        new_width = 1080
        new_height = 1920
        sidebar_width = int(new_width * 0.5)
        black_bar_height = (new_height - original_height) // 2

        # Font sizes based on video height
        main_font_size = max(int(black_bar_height * 0.1), 12)
        subtitle_font_size = max(int(original_height * 0.05), 12)

        # Background with black bars
        background = ColorClip(size=(new_width, new_height), color=(0, 0, 0)).set_duration(video_duration)
        video = video.resize(width=new_width).set_position(("center", black_bar_height))

        with open(fallacy_results_file, 'r') as json_file:
            fallacy_results = json.load(json_file)

        # Assign colors to speakers
        speakers = [result["speaker"] for result in fallacy_results if 'SPEAKER' in result["speaker"]]
        unique_speakers = list(set(speakers))
        speaker_colors = dict(zip(unique_speakers, generate_pastel_colors(len(unique_speakers))))
        
        fallacy_counters = {}
        start_end_times = []  # Store start and end times
        for result in fallacy_results:
            speaker = result["speaker"]
            if speaker not in fallacy_counters and 'SPEAKER' in speaker:
                fallacy_counters[speaker] = 0

            start_time = max(0, min(result["start"], video_duration))
            end_time = max(start_time, min(result["end"], video_duration))
            start_end_times.append((start_time, end_time))
            
            
        def create_scoreboard(fallacy_counters):
            scoreboard_text = "Fallacy Scoreboard\n" + "\n".join([f"{s}: {c}" for s, c in fallacy_counters.items()])
            return TextClip(scoreboard_text, fontsize=main_font_size, color='white', bg_color='black', size=(sidebar_width, black_bar_height - 10), method='caption', align='West', stroke_color='white', stroke_width=1.0)

        scoreboard = create_scoreboard(fallacy_counters)
        scoreboard = scoreboard.set_position((10, 10)).set_duration(video_duration)

        text_clips = [scoreboard]
        gif_clips = []
        current_fallacy = None

        for i, result in enumerate(fallacy_results):
            start_time = max(0, min(result["start"], video_duration))
            end_time = max(start_time, min(result["end"], video_duration))
            speaker = result["speaker"]
            fallacies = result["fallacy_type"]
            reason = result["fallacy_explanation"]
            text_segment = result["text_segment"]
            gif_query = result["gif_query"]

            if not isinstance(fallacies, list):
                fallacies = [fallacies]

            valid_fallacies = [f for f in fallacies if f not in [None, "None"]]

            if valid_fallacies and 'SPEAKER' in speaker:
                # Update fallacy counters
                fallacy_counters[speaker] = fallacy_counters.get(speaker, 0) + len(valid_fallacies)
                
                # Create a new scoreboard only when a new fallacy is detected
                scoreboard = create_scoreboard(fallacy_counters)
                scoreboard = scoreboard.set_position((10, 10)).set_start(start_time)
                text_clips.append(scoreboard)

                fallacy_text = f"Speaker: {speaker}\nFallacies: {', '.join(valid_fallacies)}\n\nExplanation:\n{reason}"

                fallacy_box = TextClip(fallacy_text, fontsize=subtitle_font_size, color=speaker_colors[speaker], bg_color='black', method='caption', align='West', size=(new_width - 20, None))
                fallacy_box = fallacy_box.set_position((10, original_height + black_bar_height)).set_start(start_time).set_end(end_time)
                text_clips.append(fallacy_box)
                current_fallacy = fallacy_box
            elif current_fallacy:
                current_fallacy = current_fallacy.set_end(end_time)
                text_clips.append(current_fallacy)

            gif_dir = os.path.join(os.path.dirname(os.path.dirname(final_video_name)), 'gifs')
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            gif_path = os.path.join(gif_dir, f"{gif_query}.gif")
            if not os.path.exists(gif_path):
                gif_path = search_and_download_gif(gif_query, gif_path, limit=1)

            if gif_path:
                gif_clip = add_gif_to_video_with_black_bars(black_bar_height, gif_path, start_time, end_time, new_width)
                gif_clips.append(gif_clip)

            subtitle = TextClip(text_segment, fontsize=subtitle_font_size, color=speaker_colors[speaker], bg_color='black', method='caption', size=(new_width, None))
            subtitle = subtitle.set_position((0, original_height + black_bar_height - 2 * subtitle.h)).set_start(start_time).set_end(end_time)
            text_clips.append(subtitle)

        final_clip = CompositeVideoClip([background, video] + text_clips + [gif for gif in gif_clips if gif], size=(new_width, new_height))
        final_clip = final_clip.set_duration(video_duration)

        final_clip.write_videofile(final_video_name, codec="libx264", audio_codec="aac", threads=8, fps=24, ffmpeg_params=['-c:v', 'h264_nvenc', '-preset', 'fast'] )

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def add_gif_to_video_with_black_bars(black_bar_height, gif_path, start_time, end_time, new_width):
    gif = VideoFileClip(gif_path)
    # Resize GIF to fit within the black bar area
    gif = gif.resize(height=black_bar_height // 2.5)

    # Position the GIF in the top-right corner of the top black bar
    position = (new_width - gif.size[0] - 10, black_bar_height - gif.size[1] -10)

    # Loop the GIF to match the desired duration
    gif = gif.loop(duration=end_time - start_time).set_start(start_time).set_position(position)

    return gif


