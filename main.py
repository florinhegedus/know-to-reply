# main.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import os
import base64


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Check if running inside a container
in_container = os.getenv('IN_CONTAINER', False)  # Default to 'False' if not found
if not in_container:
    load_dotenv()

# Set your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')  # Recommended to use environment variables
client = OpenAI(api_key=api_key)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
async def index():
    # Simple HTML form
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot Interface</title>
    </head>
    <body>
        <h1>Chat with GPT</h1>
        <form action="/chat" enctype="multipart/form-data" method="post">
            <textarea name="prompt" rows="4" cols="50" placeholder="Enter your prompt here"></textarea><br><br>
            <input type="file" name="file"><br><br>
            <input type="submit" value="Send">
        </form>
    </body>
    </html>
    """


@app.post("/chat")
async def chat(prompt: str = Form(...), file: UploadFile = File(None)):
    if file:
        # Read and process the image
        contents = await file.read()
        # Open the image using PIL and BytesIO
        image = Image.open(BytesIO(contents))

        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # You can change format if needed
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Append some instruction to the prompt
        prompt += " Respond based on the screenshots."

        # Prepare the image in the data URI format required for the OpenAI API
        image_data_uri = f"data:image/jpeg;base64,{base64_image}"

    # Call OpenAI API
    response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", 
                            "content": [{"type": "text",
                                            "text": prompt},{
                                            "type": "image_url",
                                            "image_url": {"url": image_data_uri}}
                                        ]}],
                    stream=False,
                )

    reply = response.choices[0].message.content

    # Return the response as HTML
    return HTMLResponse(f"<h2>Response:</h2><p>{reply}</p>")


app.mount("/static", StaticFiles(directory="static"), name="static")
