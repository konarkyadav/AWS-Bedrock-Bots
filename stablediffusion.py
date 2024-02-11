import boto3
import json
import base64
import os

image_request = """
provide me an 4k hd image of mountains, also use a reddish sky a beautiful sunset and
cinematic display
"""
image_prompt=[{"text":image_request,"weight":1}]
ai_engine = boto3.client(service_name="bedrock-runtime")
image_config = {
    "text_prompts":image_prompt,
    "cfg_scale": 10,
    "seed": 0,
    "steps":50,
    "width":512,
    "height":512
}

request_body = json.dumps(image_config)
diffusion_model = "stability.stable-diffusion-xl-v0"
ai_response = ai_engine.invoke_model(
    body=request_body,
    modelId=diffusion_model,
    accept="application/json",
    contentType="application/json",
)

ai_response_data = json.loads(ai_response.get("body").read())
print(ai_response_data)
image_artifact = ai_response_data.get("artifacts")[0]
image_data_encoded = image_artifact.get("base64").encode("utf-8")
decoded_image_bytes = base64.b64decode(image_data_encoded)

# Save the generated image to a file in the specified directory.
image_output_dir = "generated_images"
os.makedirs(image_output_dir, exist_ok=True)
image_path = f"{image_output_dir}/beach_scene.png"
with open(image_path, "wb") as image_file:
    image_file.write(decoded_image_bytes)
