# Prompt_lab_llama.py

import json 
from datetime import datetime, timezone
import requests

LOG_PATH = "chat_log.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"


def log_event(event:dict) -> None:
    """Append a single JSON event to a JSON log file."""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def chat_once(user_text: str, system_prompt: str, temperature: float, model: str) -> str:
    """
    Send one user message to the model and return the assistant text.
    Step 1 is single-turn by design (no memory yet).
    """
    payload = {
        "model": model, 
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "options": {
            "temperature": temperature,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "")


def main() -> None:
    system_prompt = (
        "You are a helpful commercial customer support assistant."
        "Be concise, accurate, and ask clarifying questions when needed."
    )

    model = "llama3.2:3b" # solid default; change if you want
    temperature = 0.3

    print("Step 1 CLI bot running (Ollama).")
    print("Type your message. Type 'q' to quit.")
    print("Commands: /temp 0.7  /model llama3.2:3b  /sys <new system prompt>\n")

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










