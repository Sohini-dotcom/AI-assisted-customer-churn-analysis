from dotenv import load_dotenv       # reads the key from the .env file
from google import genai             # Google's official Gemini library

load_dotenv()
client = genai.Client()              # reads GEMINI_API_KEY automatically

response = client.models.generate_content(
    model="gemini-3.6-flash",        # free-tier eligible, fast and cheap
    contents="In two sentences, what is customer churn?",
)

print(response.text)
print("Input tokens:", response.usage_metadata.prompt_token_count)
print("Output tokens:", response.usage_metadata.candidates_token_count)
