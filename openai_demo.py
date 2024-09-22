import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
import base64

# Check if running inside a container
in_container = os.getenv('IN_CONTAINER', False)  # Default to 'False' if not found

# Access the secret from environment variables
if not in_container:
    load_dotenv()
    
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
# Path to your image
image_path = "images/kefalonia.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

print("METHOD 1")
# METHOD 1
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", 
               "content": [{"type": "text",
                            "text": "What’s in this image?"},{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                          ]}],
    stream=False,
)

stream=False
if stream:
  for chunk in stream:
      if chunk.choices[0].delta.content is not None:
          print(chunk.choices[0].delta.content, end="")
else:
   print(stream.choices[0].message.content)

print("METHOD 2")

# METHOD 2
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What’s in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())
