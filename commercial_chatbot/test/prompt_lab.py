# prompt_lab.py  - create the step 1 CLI script

import os 
import json 
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

"""
API key = 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
"""

# Load environment variables from .env
load_dotenv()

# Create client using OPENAI_API_KEY from environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

LOG_PATH = "chat_log.jsonl"

def log_event(event:dict) -> None:
    """Append a single JSON event to a JSON log file."""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def chat_once(user_text: str, system_prompt: str, temperature: float, model: str) -> str:
    """
    Send one user message to the model and return the assistant text.
    Step 1 is single-turn by design (no memory yet).
    """
    resp = client.chat.completions.create(
        model = model, 
        temperature = temperature,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
    )
    return resp.choices[0].message.content or ""

def main() -> None:
    system_prompt = (
        "You are a helpful commercial customer support assistant."
        "Be concise, accurate, and ask clarifying questions when needed."
    )

    model = "gpt-4.1-mini" # solid default; change if you want
    temperature = 0.3

    print("Step 1 CLI bot running.")
    print("Type your message. Type 'q' to quit.")
    print("Commands: /temp 0.7  /model gpt-4.1-mini  /sys <new system prompt>\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text: continue
        if user_text == "q":
            print("Bye")
            break
        # Command: Change temperature
        if user_text.startswith("/temp "):
            try:
                temperature = float(user_text.split(" ", 1)[1].strip())
                print(f"set temperature = {temperature})\n")
            except ValueError:
                print("(invalid temperature)\n")
            continue

        # Command: Change model
        if user_text.startswith("/model "):
            model = user_text.split(" ", 1)[1].strip()
            print(f"set model = {model}\n")
            continue    

        # Command: Change system prompt
        if user_text.startswith("/sys "):
            system_prompt = user_text.split(" ", 1)[1].strip()
            print(f"set system prompt = {system_prompt}\n")
            continue

        started = datetime.now(timezone.utc).isoformat() 

        try:
            assistant_text = chat_once(
                user_text=user_text, 
                system_prompt=system_prompt, 
                temperature=temperature,
                model=model
            )
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        ended = datetime.now(timezone.utc).isoformat() 

        print(f"\nBot: {assistant_text}\n")
        
        log_event(
            {
                "started_utc": started,
                "ended_utc": ended,
                "model": model,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "user_text": user_text,
                "assistant_text": assistant_text,
            }
        )

        
if __name__ == "__main__":
    main()






