from llm.groq_client import GroqClient


class CoherenceEvaluator:
    """
    Scores how well-structured, clear, and coherent the LLM answer is.
    Uses LLM-as-judge to score coherence from 0.0 to 1.0.
    """

    def __init__(self):
        self.client = GroqClient()

    def evaluate(self, answer: str) -> dict:
        prompt = f"""You are an expert writing evaluator. Rate the coherence and clarity of the following answer.

Answer:
{answer}

Instructions:
- A score of 1.0 means the answer is exceptionally clear, well-structured, and easy to understand
- A score of 0.5 means the answer is somewhat clear but could be improved
- A score of 0.0 means the answer is confusing, incoherent, or poorly written
- Consider: logical flow, sentence structure, clarity of expression, and completeness
- Respond with ONLY a JSON object in this exact format:
{{"score": 0.0 to 1.0, "reason": "one sentence explanation"}}

Do not add any text outside the JSON."""

        try:
            response = self.client.generate(prompt, temperature=0.0, max_tokens=150)
            import json
            clean = response.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
            return {
                "coherence_score": round(float(result.get("score", 0.0)), 4),
                "reason": result.get("reason", "")
            }
        except Exception as e:
            return {
                "coherence_score": None,
                "reason": f"Evaluation failed: {str(e)}"
            }