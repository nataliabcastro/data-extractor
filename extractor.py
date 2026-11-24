"""Calls the LLM with structured output and validates the result with Pydantic."""
import os
from datetime import date
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from schemas import Extraction

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Create a .env file in the project root (see .env.example).")
client = genai.Client(api_key=API_KEY)

BASE_PROMPT = """You are an assistant that extracts structured information from emails and meeting notes.

Reference date (today): {today} ({weekday})

Rules:
1. Extract ONLY what is in the text. Never invent names, dates or tasks.
2. Convert relative dates ("tomorrow", "next Friday", "on the 20th") to YYYY-MM-DD using the reference date.
3. A task is an action that someone committed to or was assigned to do. Suggestions without a decision are NOT tasks.
4. If the owner or the due date is unclear, use null.
5. Do not list the same person twice.
6. Keep names, descriptions and the summary in the same language as the original text.

TEXT:
\"\"\"
{text}
\"\"\"
"""


def extract(text: str, today: Optional[date] = None, max_attempts: int = 3) -> Extraction:
    """Extract people, dates and tasks from a text, retrying on validation errors."""
    today = today or date.today()
    prompt = BASE_PROMPT.format(today=today.isoformat(), weekday=today.strftime("%A"), text=text)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=Extraction.model_json_schema(),
        temperature=0,  # more predictable answers
    )

    last_error = None
    for attempt in range(1, max_attempts + 1):
        current_prompt = prompt
        if last_error:
            current_prompt += (
                "\n\nATTENTION: your previous answer failed validation with this error:\n"
                f"{last_error}\nFix it and answer again."
            )

        response = client.models.generate_content(
            model=MODEL, contents=current_prompt, config=config
        )

        try:
            return Extraction.model_validate_json(response.text)
        except ValidationError as error:
            last_error = str(error)
            if attempt < max_attempts:
                print(f"[attempt {attempt}] validation failed, retrying...")

    raise RuntimeError(f"Extraction failed after {max_attempts} attempts: {last_error}")