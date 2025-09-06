"""
Codes for transcribing youtube videos
- input : txt file for channel urls
- output : csv, json
"""

import os
import whisper
import json
import csv

# Config 
txt_file_path = "/data2/jiyoon/Pethroom/video_urls.txt"  
csv_output_dir = "/data2/jiyoon/Pethroom/subtitles/csv"
json_output_dir = "/data2/jiyoon/Pethroom/subtitles/json"

os.makedirs(csv_output_dir, exist_ok=True)
os.makedirs(json_output_dir, exist_ok=True)


# Load Whisper
model = whisper.load_model("medium")

# Read URLs from file
with open(txt_file_path, 'r', encoding='utf-8') as f:
    video_urls = [line.strip() for line in f if line.strip()]

print(f"Found {len(video_urls)} videos to process")



for i, video_url in enumerate(video_urls, 1):
    print(f"\n--- Processing video {i}/{len(video_urls)} ---")
    print(f"URL: {video_url}")
    
    # Extract video ID for filename
    video_id = video_url.split('/')[-1].split('?')[0].replace('watch?v=', '')
    
    audio_path = f"video_{video_id}.mp3"
    csv_path = os.path.join(csv_output_dir, f"subtitle_{video_id}.csv")
    json_path = os.path.join(json_output_dir, f"subtitle_{video_id}.json")
    
    try:
        # Download mp3 using yt-dlp
        print("Downloading audio...")
        os.system(f'yt-dlp -x --audio-format mp3 -o "video_{video_id}.%(ext)s" "{video_url}"')
        
        if not os.path.exists(audio_path):
            print(f"❌ Failed to download audio for {video_url}")
            continue
        
        # Whisper transcription
        print("Transcribing...")
        result = model.transcribe(audio_path, language="ko", word_timestamps=False)
        segments = result['segments']
        
        # Save CSV
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["start", "end", "text"])
            for seg in segments:
                writer.writerow([seg['start'], seg['end'], seg['text']])
        
        # Save JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Successfully processed! CSV: {csv_path}, JSON: {json_path}")
        
        # Clean up audio file to save space
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"🗑️ Cleaned up {audio_path}")
            
    except Exception as e:
        print(f"❌ Error processing {video_url}: {str(e)}")
        continue

print("\n🎉 All videos processed!")