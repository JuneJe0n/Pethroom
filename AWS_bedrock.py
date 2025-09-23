# Use the native inference API to send a text message to Amazon Titan Text G1 - Express.

import boto3
import json
import os
from botocore.exceptions import ClientError
from credentials import aws_access_key_id, aws_secret_access_key, model_arn, knowledge_base_id

# --- Config ---
prompt = """
KB에 있는 데이터는 csv 포맷입니다. csv에는 start, end, text의 3개의 열으로 구성되어있습니다. 
start, end는 숫자, text는 string 타입입니다. 주어진 질문에 대해서 답변을 할 때 start,end에 해당하는 숫자만 출력하세요. 숫자 외의 다른 문자는 출력하지 마세요.
---
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
                # Crop in case response is too long
                content_preview = reference['content']['text'][:100]
                location = reference['location']['s3Location']['uri']
                print(f"- [content]: {content_preview}...\n- [location]: {location}\n")