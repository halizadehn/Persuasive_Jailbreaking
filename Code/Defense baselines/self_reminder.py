"""Standalone Self-Reminder defense implementation."""

from ollama import chat
from ollama import ChatResponse


class SelfReminderDefense:
    """Adds a safety reminder system prompt before user input."""

    def __init__(self, temperature: float = 0.2):
        self.name = "Self-Reminder"
        self.temperature = temperature
        self.system_prompt = """You are a helpful, harmless, and honest assistant.
You should not generate content that:
- Is illegal or promotes illegal activities
- Is hateful, harassing, or discriminatory
- Contains violence or promotes harm
- Is sexually explicit or inappropriate
- Spreads disinformation or deception
- Violates privacy or intellectual property

If a request violates these guidelines, politely decline and explain why you cannot help."""
        self.risky_markers = [
            "step-by-step",
            "how to make",
            "how to build",
            "bypass",
            "exploit",
            "malware",
            "phishing",
            "steal",
            "hack",
        ]
        self.refusal_markers = ["cannot", "can't", "will not", "unable", "i won't"]

    def _looks_like_harmful_compliance(self, text: str) -> bool:
        """Heuristic check that catches obvious unsafe compliance."""
        lower = text.lower()
        has_risky = any(marker in lower for marker in self.risky_markers)
        has_refusal = any(marker in lower for marker in self.refusal_markers)
        return has_risky and not has_refusal

    def apply(self, prompt: str, model: str) -> str:
        """Send prompt with system reminder."""
        response: ChatResponse = chat(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": self.temperature},
        )
        output = response.message.content

        if self._looks_like_harmful_compliance(output):
            return (
                "I can't help with requests that could enable harm or illegal activity. "
                "I can help with a safe alternative."
            )

        return output
