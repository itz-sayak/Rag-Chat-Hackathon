import os
import json
import requests
from dotenv import load_dotenv

def main():
    # Load .env if present so GROQ_MODEL/GROQ_API_KEY are picked up
    try:
        # Ensure .env values override any previously-set shell env vars
        load_dotenv(override=True)
    except Exception:
        pass
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
    if not api_key:
        print("GROQ_API_KEY not set")
        return 1
    bases = [
        os.getenv("GROQ_API_URL", "https://api.groq.com/openai"),
        "https://api.groq.com",
    ]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'pong'"},
        ],
        "max_tokens": 10,
        "stream": False,
    }
    last_err = None
    print(f"Using model: {model}")
    for base in bases:
        url = base.rstrip('/') + "/v1/chat/completions"
        print(f"Trying {url}...")
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            print("Status:", resp.status_code)
            print("Body:", resp.text[:500])
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"].get("content", "")
            print("OK, content:", msg)
            return 0
        except Exception as e:
            last_err = e
            print("Error:", e)
            continue
    print("Failed to connect to Groq:", last_err)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
