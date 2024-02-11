import boto3
import json

shakespeare_prompt = """
Act as Shakespeare and compose a verse on Generative AI
"""

ai_service = boto3.client(service_name="bedrock-runtime")

request_details = {
    "prompt": shakespeare_prompt,
    "maxTokens": 512,
    "temperature": 0.8,
    "topP": 0.8
}
request_json = json.dumps(request_details)
ai_model_identifier = "ai21.j2-mid-v1"
ai_response = ai_service.invoke_model(
    body=request_json,
    modelId=ai_model_identifier,
    accept="application/json",
    contentType="application/json",
)

ai_response_content = json.loads(ai_response.get("body").read())
generated_text = ai_response_content.get("completions")[0].get("data").get("text")
print(generated_text)
