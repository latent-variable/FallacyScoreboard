
from pathlib import Path

import yt_dlp
from pytube import YouTube

def download_youtube_video(url, output_path):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return next(output_path.glob('*.mp4'))


# def download_youtube_video(video_url, output_path):
    
#     yt = YouTube(video_url)
#     print(yt.title)
#     print(yt.thumbnail_url)
    
#     stream = yt.streams.filter(file_extension='mp4').first()
#     if not stream:
#         raise Exception("No suitable streams found for this video")
#     stream.download(output_path)
#     return output_path / stream.default_filename
    
# Usage
output_path =  Path("/Files/out.mp4")
video_url = "https://www.youtube.com/watch?v=LM-bjbeqaqE"
video_path = download_youtube_video(video_url, output_path)
