from llm.groq_client import GroqClient


class HallucinationEvaluator:
    """
    Detects if the LLM answer contains claims not supported by the given context.
    Uses an LLM-as-judge approach: asks Groq to assess groundedness.
    Score: 1.0 = fully grounded, 0.0 = fully hallucinated
    """

    def __init__(self):
        self.client = GroqClient()

    def evaluate(self, context: str, question: str, answer: str) -> dict:
        prompt = f"""You are an expert fact-checker. Your job is to determine if an answer is grounded in the given context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Instructions:
- Read the context carefully
- Check if every claim in the answer is supported by the context
- Respond with ONLY a JSON object in this exact format:
{{"grounded": true or false, "score": 0.0 to 1.0, "reason": "one sentence explanation"}}

Where score 1.0 = fully grounded in context, 0.0 = completely hallucinated.
Do not add any text outside the JSON."""

        try:
            response = self.client.generate(prompt, temperature=0.0, max_tokens=200)
            # Clean response and parse JSON
            import json
            # Strip markdown code blocks if present
            clean = response.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
            return {
                "grounded": result.get("grounded", False),
                "hallucination_score": round(float(result.get("score", 0.0)), 4),
                "reason": result.get("reason", "")
            }
        except Exception as e:
            return {
                "grounded": None,
                "hallucination_score": None,
                "reason": f"Evaluation failed: {str(e)}"
            }