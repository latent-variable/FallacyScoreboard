import os
import yt_dlp
from pathlib import Path
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import whisperx
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload
import json

def download_youtube_video(url, output_path):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Check if the file was downloaded successfully
    mp4_files = list(output_path.glob('*.mp4'))
    if not mp4_files:
        raise FileNotFoundError("No MP4 files found in the output directory.")
    
    # Return the first found MP4 file
    return mp4_files[0]

def extract_audio_from_video(video_path, audio_path):
    video = VideoFileClip(str(video_path))
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio_with_whisperx(audio_path, output_path):
    device = "cuda"
    model = whisperx.load_model("large-v3", device)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio)

    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)

    diarize_model = whisperx.DiarizationPipeline(device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result_aligned)
    
    # write results to a text file 
    with open(os.path.join(output_path, 'transcription.txt'), 'w') as f:
        f.write(json.dumps(result, indent=2))
        
    
    return result

def detect_fallacies(transcript):
    # Implement or integrate your fallacy detection logic here
    # Return a list of dictionaries with fallacies and corresponding speaker info
    pass

def overlay_fallacies_on_video(video_path, fallacy_results):
    video = VideoFileClip(str(video_path))
    clips = [video]

    for result in fallacy_results:
        start_time = result["start"]
        end_time = result["end"]
        speaker = result["speaker"]
        fallacy = result["fallacy"]
        reason = result["reason"]

        txt_clip = TextClip(f"Speaker {speaker} committed a fallacy: {fallacy}\nReason: {reason}", fontsize=24, color='white')
        txt_clip = txt_clip.set_pos(('center', 'bottom')).set_duration(end_time - start_time).set_start(start_time)
        
        clips.append(txt_clip)

    final_clip = CompositeVideoClip(clips)
    final_clip.write_videofile("output_with_fallacies.mp4", codec="libx264")

def upload_video(video_file, title, description, category, tags):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "YOUR_CLIENT_SECRET_FILE.json"

    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
    credentials = flow.run_console()

    youtube = googleapiclient.discovery.build(api_service_name, api_version, credentials=credentials)

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category
            },
            "status": {
                "privacyStatus": "public"
            }
        },
        media_body=MediaFileUpload(video_file)
    )
    response = request.execute()

    print("Video uploaded successfully. Video ID:", response.get("id"))

# Example usage
output_path = Path("./Files")
video_url = "https://www.youtube.com/watch?v=LM-bjbeqaqE"
# try:
video_path = download_youtube_video(video_url, output_path)

audio_path = output_path / "audio.wav"
extract_audio_from_video(video_path, audio_path)
transcription_result = transcribe_audio_with_whisperx(audio_path, output_path)
# fallacy_results = detect_fallacies(transcription_result["segments"])
# overlay_fallacies_on_video(video_path, fallacy_results)
# upload_video("output_with_fallacies.mp4", "Title of Video", "Description of Video", "22", ["tag1", "tag2"])
# except Exception as e:
#     print(f"An error occurred: {e}")
