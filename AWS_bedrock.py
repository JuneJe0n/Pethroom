"""
Codes to test kb
"""

import boto3
import json
import os
from botocore.exceptions import ClientError
from credentials import aws_access_key_id, aws_secret_access_key, model_arn, knowledge_base_id

# --- Config ---
prompt = """
질문 : 강아지 중성화 수술 시기는 언제가 적절해?
"""

# --- Generate response ---
# Create Amazon Bedrock Runtime client
brt = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name="ap-northeast-2",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)


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
    exit(1)


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