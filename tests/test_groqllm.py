import os
import json
import pytest

from rag import GroqLLM


class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def test_groqllm_call(monkeypatch):
    dummy = {"choices": [{"text": "Hello from Groq"}]}

    def fake_post(url, json=None, headers=None, timeout=None, stream=False):
        return DummyResponse(dummy)

    monkeypatch.setattr("requests.post", fake_post)
    os.environ["GROQ_API_KEY"] = "fake"
    llm = GroqLLM(model_name="llama-4-maverick")
    out = llm._call("Say hi")
    assert "Hello from Groq" in out
