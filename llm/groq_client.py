import os
from groq import Groq


class GroqClient:

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set. Run: export GROQ_API_KEY=your_key")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def generate(self, prompt, temperature=0.1, max_tokens=512):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()