"""
Codes to test kb and save results in csv format
"""

import boto3
import json
import os
import csv
from botocore.exceptions import ClientError
from credentials import aws_access_key_id, aws_secret_access_key, model_arn, knowledge_base_id

# --- Config ---
prompt_template = "{}"
questions_file_path = '/data2/jiyoon/Pethroom/data/dog_questions.py'  
output_csv_path = '/data2/jiyoon/Pethroom/data/bedrock_results.csv' 


# --- Generate response ---
# Load questions 
with open(questions_file_path, 'r', encoding='utf-8') as f:
    questions = eval(f.read())

# Create Amazon Bedrock Runtime client
brt = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="ap-northeast-2",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Create CSV file
csv_filename = output_csv_path
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['question_number', 'question', 'output_text', 'start_time', 'content_text', 'location']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate through all questions
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/{len(questions)}: {question}")
        print(f"{'='*60}")
        
        prompt = prompt_template.format(question)

        try:
            response = brt.retrieve_and_generate(
            input={
                'text': prompt
            },
            retrieveAndGenerateConfiguration={
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id ,  
                    'modelArn': model_arn,       
                    "orchestrationConfiguration": { 
                    "queryTransformationConfiguration": {
                        "type": "QUERY_DECOMPOSITION"
                    }},       
                    'generationConfiguration': {
                        'inferenceConfig': {
                            'textInferenceConfig': {
                                'maxTokens': 512,
                                'temperature': 0,
                                'topP': 0.9,
                            }
                        }
                    }
                },
                'type': 'KNOWLEDGE_BASE'
            },
        )

            
        except (ClientError, Exception) as e:
            print(f"ERROR: {e}")
            continue


        # Decode the response body
        output_text = response['output']['text']
        print("\n--- RESPONSE ---\n")
        print(output_text)

        citations = response.get('citations', [])
        if citations:
                print("\n--- SOURCE ---")
                for citation in citations:
                    for reference in citation.get('retrievedReferences', []):
                        content_text = reference['content']['text']
                        location = reference['location']['s3Location']['uri']
                        
                        # Extract first number after first '\r'
                        start_time = ""
                        first_r_index = content_text.find('\r')
                        if first_r_index != -1:
                            after_first_r = content_text[first_r_index + 1:]
                            # Find the first comma to get the start time
                            comma_index = after_first_r.find(',')
                            if comma_index != -1:
                                start_time = after_first_r[:comma_index].strip()
                                print(f"- [start_time]: {start_time}")
                        
                        content_preview = content_text[:50]
                        print(f"- [content]: {content_preview}...\n- [location]: {location}\n")
                        
                        # Write to CSV
                        writer.writerow({
                            'question_number': i,
                            'question': question,
                            'output_text': output_text,
                            'start_time': start_time,
                            'content_text': content_text,
                            'location': location
                        })
        else:
            # Write to CSV even if no citations
            writer.writerow({
                'question_number': i,
                'question': question,
                'output_text': output_text,
                'start_time': '',
                'content_text': '',
                'location': ''
            })

print(f"\nResults saved to {csv_filename}")