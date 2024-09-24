from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import os
import base64
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Check if running inside a container
in_container = os.getenv('IN_CONTAINER', False)  # Default to 'False' if not found
if not in_container:
    load_dotenv()

# Set your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')  # Recommended to use environment variables
client = OpenAI(api_key=api_key)

PROMPT_TEMPLATE = """
The purpose of this query is you to help me with a conversation.
Respond to the question based on the following rules:
- Only respond to questions related to getting help on conversations, specify if this is not the case
- Respond in the language of the provided screenshots.
- Enclose any examples or suggestions that the user might want to copy between `<copy>` `</copy>` tags.
- Do not use backticks
- Do not use double quotation marks
- Do not use code blocks
- Do not write a response twice

---

This is what the user wants: {question}
Remember to put the examples to be copied between `<copy>` `</copy>` tags.
"""



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(prompt: str = Form(...), file: UploadFile = File(None)):
    image_data_uri = None
    if file:
        # Read and process the image
        contents = await file.read()
        # Open the image using PIL and BytesIO
        image = Image.open(BytesIO(contents))

        # Convert image to RGB if it has an alpha channel (RGBA)
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # You can change format if needed
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Append some instruction to the prompt
        prompt += " Respond based on the screenshots."

        # Prepare the image in the data URI format required for the OpenAI API
        image_data_uri = f"data:image/jpeg;base64,{base64_image}"

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=prompt)

    # Prepare the messages
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if image_data_uri:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": image_data_uri}
        })

    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=False,
        )
        reply = response.choices[0].message.content
        logger.info(reply)
        return {"reply": reply}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

app.mount("/static", StaticFiles(directory="static"), name="static")
