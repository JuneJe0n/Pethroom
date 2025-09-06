import subprocess
import json


# --- Config ---
file_path = '/data2/jiyoon/Pethroom/video_urls.txt'  
channel_url = "https://www.youtube.com/@yoonsem_dog"




def get_channel_urls(channel_url):  
    command = ['yt-dlp', '--flat-playlist', '--print', 'url', channel_url]
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        urls = result.stdout.strip().split('\n')
        return [url for url in urls if url]  # Filter out empty lines
    else:
        print(f"Error: {result.stderr}")
        return []



video_urls = get_channel_urls(channel_url)
print(f"Found {len(video_urls)} videos")


with open(file_path, 'w') as f:
    for url in video_urls:
        f.write(url + '\n')

print(f"URLs saved to: {file_path}")