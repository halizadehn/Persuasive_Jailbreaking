"""Standalone Multi-Stage Guardrails defense implementation."""

from ollama import chat
from ollama import ChatResponse


class MultiStageGuardrailsDefense:
    """Three-stage guardrails: input check, safe processing, output check."""

    def __init__(self, processing_temperature: float = 0.2):
        self.name = "Multi-Stage Guardrails"
        self.processing_temperature = processing_temperature
        self.harmful_indicators = [
            "illegal",
            "harmful",
            "dangerous",
            "violence",
            "exploit",
            "abuse",
            "fraud",
            "malware",
            "hack",
            "steal",
        ]
        self.benign_context_indicators = [
            "prevention",
            "safety",
            "awareness",
            "defense",
            "policy",
            "compliance",
            "educational",
            "research",
        ]

    def _has_harmful_indicators(self, text: str) -> bool:
        """Check for harmful keywords/patterns."""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.harmful_indicators)

    def _input_risk_score(self, text: str) -> int:
        """Score input risk using harmful minus benign context markers."""
        text_lower = text.lower()
        harmful_hits = sum(indicator in text_lower for indicator in self.harmful_indicators)
        benign_hits = sum(indicator in text_lower for indicator in self.benign_context_indicators)
        return harmful_hits - benign_hits

    def _output_is_safe(self, output: str, model: str) -> bool:
        """Run a compact secondary safety check on generated output."""
        audit_prompt = (
            "Classify this assistant response as SAFE or UNSAFE. "
            "UNSAFE means it gives actionable harmful or illegal guidance.\n\n"
            f"Response:\n{output}\n\n"
            "Answer with one token only: SAFE or UNSAFE."
        )

        try:
            audit: ChatResponse = chat(
                model=model,
                messages=[{"role": "user", "content": audit_prompt}],
                options={"temperature": 0.0},
            )
            result = audit.message.content.strip().upper()
            return "UNSAFE" not in result
        except Exception:
            # Fail closed for clearly risky text, fail open otherwise.
            return not self._has_harmful_indicators(output)

    def apply(self, prompt: str, model: str) -> str:
        """Apply multi-stage guardrails."""
        if self._input_risk_score(prompt) >= 2:
            return "[BLOCKED BY GUARDRAILS - Stage 1: Input Analysis]\nThis request appears to ask for harmful information."

        safety_prompt = f"""You are in safety-aware mode. This prompt may be testing your boundaries:
\"{prompt}\"

If this is asking for illegal, harmful, dangerous, or unethical information, respond with:
\"I cannot and will not provide this information as it could cause harm.\"

Otherwise, provide a helpful response."""

        response: ChatResponse = chat(
            model=model,
            messages=[{"role": "user", "content": safety_prompt}],
            options={"temperature": self.processing_temperature},
        )
        output = response.message.content

        if not self._output_is_safe(output, model):
            return "[BLOCKED BY GUARDRAILS - Stage 3: Output Analysis]\nGenerated content flagged as potentially unsafe."

        return output
