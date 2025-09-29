"""
Preprocess dog data using GPT4o
"""
import pandas as pd
import os
import re
import zipfile
import openai
import json
import time

import sys
sys.path.append('/home/jiyoon/Pethroom')
from credentials import OPENAI_API_KEY


# --- Config ---
openai.api_key = OPENAI_API_KEY
input_csv_folder = "/data2/jiyoon/Pethroom/whisper/subtitles/csv/long_form"
root_output_path = "/data2/jiyoon/Pethroom/data/chapters/long_form"

# --- Utils ---
def format_time_to_hms(seconds):
    """
    Convert start_time to HH:MM:SS format
    """

    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def generate_chapters_with_llm(script):
    """
    Generate llm output
    """

    print("Processing text ...")

    system_prompt = (
        "당신은 동영상 스크립트의 내용을 분석하여 논리적인 챕터(장)를 나누고 제목을 생성하는 전문가입니다. "
        "사용자가 제공하는 스크립트와 타임스탬프 정보를 바탕으로, 영상의 내용 흐름이 바뀌는 지점을 정확하게 포착하세요. "
        "응답은 반드시 '00:00:00 챕터 제목' 형식의 텍스트 리스트로만 구성되어야 합니다. "
        "어떤 설명이나 머리말, 꼬리말도 붙이지 마세요. 항상 00:00:00 부터 시작해야 합니다. "
        "챕터 제목은 25자 이내로 명확하게 요약해야 합니다."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 동영상 스크립트를 분석하여 챕터를 생성해 주세요:\n\n{script}"}
            ],
            temperature=0.0
        )

        llm_output = response.choices[0].message.content.strip()
        print("Done")
        return llm_output

    except openai.AuthenticationError:
        print("\n❌ Invalid OpenAI API key")
        return None
    except Exception as e:
        print(f"\n❌ Error : {e}")
        return None


def parse_llm_chapters(llm_output):
    """
    Post process llm output
    """
    chapters_data = []

    # Search for 'HH:MM:SS title' format
    pattern = re.compile(r'(\d{1,2}:\d{2}:\d{2})\s+(.+)')


    # Post process llm output format
    for line in llm_output.split('\n'):
        match = pattern.match(line.strip())
        if match:
            start_time_str = match.group(1).strip() # time
            title = match.group(2).strip() # title

            chapters_data.append({
                'start_time': start_time_str,
                'title': title
            })

    # Chunk text to chapters
    final_chapters = []

    for i, chapter in enumerate(chapters_data):
        # Convert time to seconds fmt
        start_time_seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], chapter['start_time'].split(':')))

        # Find the start time for chapter
        start_index = df[df['start'] >= start_time_seconds].index.min()
        if pd.isna(start_index):
            start_index = 0 # Fall to 0 if start time wasn't found or 00:00:00 

        
        if i + 1 < len(chapters_data):
            next_chapter_time_seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], chapters_data[i+1]['start_time'].split(':')))
            end_index = df[df['start'] >= next_chapter_time_seconds].index.min()
        else:
            end_index = len(df) 


        chapter_chunk = df.iloc[start_index:end_index]

        if not chapter_chunk.empty:
            final_chapters.append({
                'start_time': chapter['start_time'],
                'title': chapter['title'],
                'text': " ".join(chapter_chunk['text'].dropna().astype(str).tolist()),
                'start_sec': chapter_chunk.iloc[0]['start']
            })

    return final_chapters


def save_chapters_to_files(chapters, video_id, root_path):
    video_dir = os.path.join(root_path, video_id)
    chunks_dir = os.path.join(video_dir, "chunks")
    
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Save chapter titles in titles_<video_id>.txt 
    youtube_chapter_list_path = os.path.join(video_dir, f"titles_{video_id}.txt")
    with open(youtube_chapter_list_path, 'w', encoding='utf-8') as f_yt:
        for ch in chapters:
            f_yt.write(f"{ch['start_time']} {ch['title']}\n")

    print(f"YouTube chapters list saved in: {youtube_chapter_list_path}")

    # Save each chapters in txt files
    for ch in chapters:
        # Create file name based on title
        safe_title = re.sub(r'[\\/:*?"<>|]', '', ch['title']).strip()
        file_name = f"{ch['start_sec']}_{safe_title}.txt"
        file_path = os.path.join(chunks_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(ch['text'])

    print(f"Chapter content files saved in: {chunks_dir}")
    return youtube_chapter_list_path


def process_single_csv(csv_file_path):
    """Process a single CSV file"""
    global df 
    
    if not os.path.exists(csv_file_path):
        print(f"❌ CSV file not found: {csv_file_path}")
        return False
    
    # Extract video ID from filename
    video_id = os.path.basename(csv_file_path).replace('subtitle_', '').replace('.csv', '')
    print(f"Processing Video ID: {video_id}")
    
    # Load CSV data
    df = pd.read_csv(csv_file_path)
    if df.empty:
        print(f"❌ CSV file {csv_file_path} contains no data.")
        return False
    
    # Prepare script for LLM
    script_lines = []
    for idx, row in df.iterrows():
        start_time_str = format_time_to_hms(row['start'])
        # "[HH:MM:SS] text" format 
        line = f"[{start_time_str}] {row['text']}"
        script_lines.append(line)
    llm_input_script = "\n".join(script_lines)

    # Generate chapters with LLM
    llm_output = generate_chapters_with_llm(llm_input_script)
    
    if llm_output:
        chapters = parse_llm_chapters(llm_output)
        
        if chapters:
            print("\n--- Final chapters list ---")
            for ch in chapters:
                print(f"{ch['start_time']} {ch['title']}")
            
            # Save chapters to files
            save_chapters_to_files(chapters, video_id, root_output_path)
            print(f"✅ Processing complete for {video_id}")
            return True
        else:
            print(f"\n❌ Failed to create chapters for {video_id}")
            return False
    else:
        print(f"\n❌ Failed to generate chapters with LLM for {video_id}")
        return False


def main():
    """Process all CSV files in the input folder"""
    if not os.path.exists(input_csv_folder):
        print(f"❌ Input folder not found: {input_csv_folder}")
        return False
    
    # Find all CSV files in the folder
    csv_files = [f for f in os.listdir(input_csv_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ No CSV files found in {input_csv_folder}")
        return False
    
    # Filter out already processed files
    unprocessed_files = []
    for csv_file in csv_files:
        video_id = csv_file.replace('subtitle_', '').replace('.csv', '')
        output_dir = os.path.join(root_output_path, video_id)
        titles_file = os.path.join(output_dir, f"titles_{video_id}.txt")
        
        if os.path.exists(titles_file):
            print(f"⏭️  Skipping already processed: {video_id}")
        else:
            unprocessed_files.append(csv_file)
    
    if not unprocessed_files:
        print("✅ All files have already been processed!")
        return True
    
    print(f"Found {len(csv_files)} total CSV files, {len(unprocessed_files)} remaining to process")
    
    success_count = 0
    for csv_file in unprocessed_files:
        csv_path = os.path.join(input_csv_folder, csv_file)
        print(f"\n{'='*50}")
        if process_single_csv(csv_path):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"✅ Processing complete: {success_count}/{len(unprocessed_files)} files processed successfully")
    return success_count == len(unprocessed_files)


if __name__ == "__main__":
    main()