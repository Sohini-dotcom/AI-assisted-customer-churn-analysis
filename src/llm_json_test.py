import json
from dotenv import load_dotenv     # reads the API key from the .env file
import anthropic                   # the library that talks to Claude

load_dotenv()
client = anthropic.Anthropic()

# The allowed answers. Closed lists keep the output clean and countable.
SENTIMENTS = {"positive", "neutral", "negative"}
TOPICS = {"price", "network_quality", "customer_support", "billing", "contract", "other"}

# The instructions Claude follows for every comment (the "system prompt").
SYSTEM_PROMPT = """You analyze customer comments for a telecom company.
Reply with ONLY a JSON object. No explanation, no markdown, no extra text.
Use exactly these three keys:
  "sentiment": one of "positive", "neutral", "negative"
  "topic": one of "price", "network_quality", "customer_support", "billing", "contract", "other"
  "urgency": an integer from 1 to 5, where 1 = no action needed and 5 = the customer is about to leave"""


def parse_reply(text):
    """Turn Claude's reply into a dictionary. Return None if it is not valid."""
    text = text.replace("```json", "").replace("```", "").strip()   # remove code fences if present
    try:
        result = json.loads(text)                                    # text -> dictionary
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
    """Ask Claude to classify one comment. Try twice, then give up."""
    for attempt in range(2):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",   # small, fast, cheap
            max_tokens=100,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": comment}],
        )
        result = parse_reply(response.content[0].text)
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