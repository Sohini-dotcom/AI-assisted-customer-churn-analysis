import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

SENTIMENTS = {"positive", "neutral", "negative"}
TOPICS = {"price", "network_quality", "customer_support", "billing", "contract", "other"}

SYSTEM_PROMPT = """You analyze customer comments for a telecom company.
Reply with ONLY a JSON object. No explanation, no markdown, no extra text.
Use exactly these three keys:
  "sentiment": one of "positive", "neutral", "negative"
  "topic": one of "price", "network_quality", "customer_support", "billing", "contract", "other"
  "urgency": an integer from 1 to 5, where 1 = no action needed and 5 = the customer is about to leave"""


def parse_reply(text):
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    urgency = result.get("urgency")
    is_valid = (
        result.get("sentiment") in SENTIMENTS
        and result.get("topic") in TOPICS
        and isinstance(urgency, int)
        and 1 <= urgency <= 5
    )
    return result if is_valid else None


def classify(comment):
    for attempt in range(4):   # try up to 4 times total
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=comment,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                ),
            )
        except Exception as e:
            print(f"Server error on attempt {attempt + 1}, waiting and retrying... ({e})")
            time.sleep(5 * (attempt + 1))   # wait longer each retry: 5s, 10s, 15s...
            continue

        result = parse_reply(response.text)
        if result is not None:
            return result
        print("Invalid reply, retrying...")
    return None


if __name__ == "__main__":
    comments = [
        "I've been a customer for six years and the internet has never let me down. Very happy.",
        "My bill went up again this month and support keeps putting me on hold. I'm thinking of switching providers.",
        "The connection is fine, but I don't see why I'm still stuck on a month-to-month plan.",
    ]
    for comment in comments:
        print("Comment:", comment)
        print("Result: ", classify(comment))
        print()