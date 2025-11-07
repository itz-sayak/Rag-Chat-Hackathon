import os
import json
import pytest
import requests


RUN_LIVE = os.getenv("RUN_GROQ_LIVE") == "1"


@pytest.mark.skipif(not RUN_LIVE, reason="Set RUN_GROQ_LIVE=1 to run live Groq connectivity test")
def test_groq_chat_completions_live():
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")
    assert api_key, "GROQ_API_KEY must be set for live test"
    assert model, "GROQ_MODEL must be set for live test"

    # Try OpenAI-compatible endpoint first, then fallback
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
    for base in bases:
        url = base.rstrip('/') + "/v1/chat/completions"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "")
            assert isinstance(content, str)
            return
        except Exception as e:
            last_err = e
            continue
    pytest.fail(f"Groq connectivity failed: {last_err}")
