"""
Preprocess **short form** videos using GPT4o - Generate titles only
"""
import pandas as pd
import os
import re
import openai
import time

import sys
sys.path.append('/home/jiyoon/Pethroom')
from credentials import OPENAI_API_KEY


# --- Config ---
openai.api_key = OPENAI_API_KEY
input_csv_folder = "/data2/jiyoon/Pethroom/whisper/subtitles/csv/short_form"
root_output_path = "/data2/jiyoon/Pethroom/data/chapters/short_form"

# --- Utils ---
def generate_title_with_llm(script):
    """
    Generate title for short form video
    """
    print("Generating title...")

    system_prompt = (
        "당신은 동영상 스크립트의 내용을 분석하여 제목을 생성하는 전문가입니다. "
        "사용자가 제공하는 스크립트를 바탕으로, 영상의 핵심 내용을 담은 짧고 제목을 생성하세요. "
        "제목은 15자 이내로 간결하게 작성하세요."
        "제목만 반환하고 다른 설명은 포함하지 마세요."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 동영상 스크립트를 분석하여 제목을 생성해 주세요:\n\n{script}"}
            ],
            temperature=0.0
        )

        llm_output = response.choices[0].message.content.strip()
        print("Title generated successfully")
        return llm_output

    except openai.AuthenticationError:
        print("\n❌ Invalid OpenAI API key")
        return None
    except Exception as e:
        print(f"\n❌ Error : {e}")
        return None


def save_content_to_file(title, subtitle_text, video_id, output_path):
    """Save generated title and subtitle content to file"""
    os.makedirs(output_path, exist_ok=True)
    
    # Create safe filename from title
    safe_title = re.sub(r'[\\/:*?"<>|]', '', title).strip()
    
    # Create filename in format: <generated title>.txt
    file_name = f"{video_id}_0.0_{safe_title}.txt"
    file_path = os.path.join(output_path, file_name)
    
    # Save the subtitle text as content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(subtitle_text)
    
    print(f"Content saved: {file_path}")
    return file_path


def process_single_csv(csv_file_path):
    """Process a single CSV file to generate title"""
    
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
    
    # Prepare script for LLM (combine all text)
    script_text = " ".join(df['text'].tolist())
    
    # Generate title with LLM
    generated_title = generate_title_with_llm(script_text)
    
    if generated_title:
        print(f"Generated title: {generated_title}")
        
        # Save content to file with generated title as filename
        save_content_to_file(generated_title, script_text, video_id, root_output_path)
        print(f"✅ Processing complete for {video_id}")
        return True
    else:
        print(f"\n❌ Failed to generate title for {video_id}")
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
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    success_count = 0
    for csv_file in csv_files:
        csv_path = os.path.join(input_csv_folder, csv_file)
        print(f"\n{'='*50}")
        if process_single_csv(csv_path):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"✅ Processing complete: {success_count}/{len(csv_files)} files processed successfully")
    return success_count == len(csv_files)


if __name__ == "__main__":
    main()