import boto3
import json

creative_prompt = """
Act as Shakespeare and create a verse on Generative AI
"""

ai_client = boto3.client(service_name="bedrock-runtime")

request_payload = {
    "prompt": "[INST]" + creative_prompt + "[/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}
request_body = json.dumps(request_payload)
ai_model = "meta.llama2-70b-chat-v1"
ai_response = ai_client.invoke_model(
    body=request_body,
    modelId=ai_model,
    accept="application/json",
    contentType="application/json"
)

ai_response_content = json.loads(ai_response.get("body").read())
generated_poem = ai_response_content['generation']
print(generated_poem)
