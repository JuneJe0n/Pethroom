import os
import whisper
import json
import csv

# Set video url
video_url = "https://youtu.be/OU920oKvKzY"
audio_path = "video.mp3"  
csv_path = "subtitle.csv"
json_path = "subtitle.json"


# Download mp3 using yt-plp
os.system(f'yt-dlp -x --audio-format mp3 -o "video.%(ext)s" "{video_url}"')

# Whisper
model = whisper.load_model("medium")  # small/medium/large 

result = model.transcribe(audio_path, language="ko", word_timestamps=False) # word_timestamps=True : word-wise timeline
segments = result['segments']

# Save csv
with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["start", "end", "text"])
    for seg in segments:
        writer.writerow([seg['start'], seg['end'], seg['text']])

# Save path
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(segments, f, ensure_ascii=False, indent=2)

print(f"✅ Successfully downloaded! CSV: {csv_path}, JSON: {json_path}")