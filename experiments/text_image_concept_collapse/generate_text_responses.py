from google import genai
from google.genai import types

from pathlib import Path

import json

generation_request_path = Path("experiments/text_image_concept_collapse/text_response_batch.jsonl")
if not generation_request_path.is_file():
    text_prompts = [
        "Write a paragraph on a scene in nature.",
        "Write a paragraph about an animal.",
        "Write a paragraph about a plant.",
        "Write a paragraph describing a dinner meal.",
        "Write a paragraph describing a lunch meal.",
        "Write a paragraph describing a breakfast meal.",
        "Write a paragraph describing a snack.",
        "Write a paragraph describing a fantasy novel.",
        "Write a paragraph describing a sci-fi novel.",
        "Write a paragraph describing an indie video game.",
        "Write a paragraph on a historical moment.",
    ]
    
    def create_request_line(x: str, id : int):
        return {
            "key": f"req-{id}", 
                "request": {
                    "contents": [{
                        "parts": [{
                            "text": x
                        }],
                        "role": "user"
                    }],
                    "system_instruction": {
                        "parts": [{
                            "text": "Respond to the prompt directly."
                        }]
                    },
                    "generationConfig": {
                        "thinkingConfig": types.ThinkingConfigDict(
                            include_thoughts=True, 
                            thinking_level=types.ThinkingLevel.MEDIUM
                        )
                    }
                }
            }
    
    id = 0
    request_lines = []
    for tp in text_prompts:
        for i in range(20):  
            request_lines.append(create_request_line(tp, id))
            id += 1
    
    with open(generation_request_path, "w", encoding="utf-8") as f:
        for r in request_lines:
            f.write(json.dumps(r) + "\n")
            
    print("wrote generation reqeuest to file")
    

"""------------------------------------------------ SUBMITTING JOB CODE BELOW ---------------------------------------------"""

client = genai.Client()

uploaded_file = client.files.upload(
    file = 'experiments/text_image_concept_collapse/text_response_batch.jsonl',
    config = types.UploadFileConfig(
        display_name="text-image-collapse-experiment", 
        mime_type="jsonl"
    )
)

print(f"Uploaded file name : {uploaded_file.name}")

file_batch_job = client.batches.create(
    model="gemini-3.8-flash",
    src=uploaded_file.name,
    config={
        'display_name': "file-upload-job-1",
    },
)

print(f"Created batch job: {file_batch_job.name}")
    