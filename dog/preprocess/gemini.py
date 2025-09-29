# ==============================
# Colab용: LLM (Gemini) 기반 챕터 생성
# ==============================
import pandas as pd
import os
import re
from google.colab import files
import zipfile
import google.generativeai as genai
import json
import time

from credentials import GEMINI_API_KEY
API_KEY = GEMINI_API_KEY

# API 키 설정
if API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not API_KEY:
    print("❌ 오류: API_KEY 변수에 유효한 Gemini API 키를 입력해주세요.")
    exit()

genai.configure(api_key=API_KEY)
print("Gemini API 키 설정 완료.")

# -------------------------------
# 2. CSV 업로드 (기존 코드와 동일)
# -------------------------------
uploaded = files.upload()
if not uploaded:
    print("파일 업로드가 취소되었거나 실패했습니다.")
    exit()

csv_file_name = list(uploaded.keys())[0]
video_id = os.path.basename(csv_file_name).replace('subtitle_', '').replace('.csv', '')

# -------------------------------
# 3. 데이터 준비 및 시간 포맷 함수 (기존 코드와 동일)
# -------------------------------
df = pd.read_csv(csv_file_name)
if df.empty:
    print("CSV 파일에 데이터가 없습니다.")
    exit()

def format_time_to_hms(seconds):
    """초를 HH:MM:SS 포맷으로 변환"""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

llm_input_script = "\n".join(
    [f"[{format_time_to_hms(row['start'])}] {row['text']}"
     for index, row in df.iterrows()]
)

# -------------------------------
# 4. LLM 호출 및 챕터 생성 (기존 코드와 동일)
# -------------------------------
def generate_chapters_with_llm(script):
    """Gemini API를 사용하여 챕터 목록 생성"""
    print("--- LLM에 챕터 생성을 요청 중입니다 (최대 1~2분 소요) ---")
    user_prompt = (
        "당신은 동영상 스크립트의 내용을 분석하여 논리적인 챕터(장)를 나누고 제목을 생성하는 전문가입니다. "
        "사용자가 제공하는 스크립트와 타임스탬프 정보를 바탕으로, 영상의 내용 흐름이 바뀌는 지점을 정확하게 포착하세요. "
        "응답은 반드시 '00:00:00 챕터 제목' 형식의 텍스트 리스트로만 구성되어야 합니다. "
        "어떤 설명이나 머리말, 꼬리말도 붙이지 마세요. 항상 00:00:00 부터 시작해야 합니다. "
        "챕터 제목은 25자 이내로 명확하게 요약해야 합니다."
        f"다음 동영상 스크립트를 분석하여 챕터를 생성해 주세요:\n\n{script}"
    )

    try:
        #model = genai.GenerativeModel('gemini-1.5-pro-latest')
        # 파이썬 코드에서 모델명 변경
        model = genai.GenerativeModel(model_name='models/gemini-2.0-flash')
        response = model.generate_content(
            user_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
            )
        )
        llm_output = response.text.strip()
        print("--- LLM 응답 수신 완료 ---")
        return llm_output

    except Exception as e:
        print(f"\n❌ 오류: LLM 호출 중 문제가 발생했습니다: {e}")
        return None

# LLM 챕터 결과 파싱 및 데이터프레임 구성 (기존 코드와 동일)
def parse_llm_chapters(llm_output):
    """LLM의 텍스트 응답을 챕터 목록으로 파싱"""
    chapters_data = []
    pattern = re.compile(r'(\d{1,2}:\d{2}:\d{2})\s+(.+)')
    for line in llm_output.split('\n'):
        match = pattern.match(line.strip())
        if match:
            start_time_str = match.group(1).strip()
            title = match.group(2).strip()
            chapters_data.append({'start_time': start_time_str, 'title': title})

    final_chapters = []
    for i, chapter in enumerate(chapters_data):
        start_time_seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], chapter['start_time'].split(':')))
        start_index = df[df['start'] >= start_time_seconds].index.min()
        if pd.isna(start_index): start_index = 0

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
                'text': " ".join(chapter_chunk['text'].tolist()),
                'start_sec': chapter_chunk.iloc[0]['start']
            })
    return final_chapters

# -------------------------------
# 5. 챕터 파일로 저장 및 압축 (기존 코드와 동일)
# -------------------------------
def save_chapters_to_files(chapters, video_id, output_dir='chapters'):
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []
    youtube_chapter_list_path = os.path.join(output_dir, f"youtube_chapters_{video_id}.txt")
    with open(youtube_chapter_list_path, 'w', encoding='utf-8') as f_yt:
        for ch in chapters:
            f_yt.write(f"{ch['start_time']} {ch['title']}\n")
    file_paths.append(youtube_chapter_list_path)
    print(f"YouTube 챕터 리스트 생성 완료: {youtube_chapter_list_path}")

    for ch in chapters:
        safe_title = re.sub(r'[\\/:*?"<>|]', '', ch['title']).strip()
        file_name = f"{ch['start_sec']}_{safe_title}.txt"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(ch['text'])
        file_paths.append(file_path)
    return file_paths

def zip_chapters(file_paths, output_zip='llm_chapters_output.zip'):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for file in file_paths:
            zipf.write(file, arcname=os.path.join('chapters', os.path.basename(file)))
    print(f"모든 챕터가 압축 완료: {output_zip}")
    return output_zip

# -------------------------------
# 6. 실행 (기존 코드와 동일)
# -------------------------------
llm_output = generate_chapters_with_llm(llm_input_script)

if llm_output:
    chapters = parse_llm_chapters(llm_output)

    if chapters:
        print("\n--- 최종 챕터 목록 ---")
        for ch in chapters:
             print(f"{ch['start_time']} {ch['title']}")

        file_paths = save_chapters_to_files(chapters, video_id)
        zip_file = zip_chapters(file_paths)
        files.download(zip_file)
    else:
        print("\n❌ 파싱 결과 챕터 목록을 생성하지 못했습니다. LLM 응답 형식을 확인하세요.")