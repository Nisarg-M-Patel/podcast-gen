from dotenv import load_dotenv
import os
from google import genai

# Load .env from current directory
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

print("Models supporting generateContent:\n")

for model in client.models.list():
    actions = model.supported_actions or []
    if "generateContent" in actions:
        print(model.name)
