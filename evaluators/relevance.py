from llm.groq_client import GroqClient


class RelevanceEvaluator:
    """
    Scores how relevant the LLM answer is to the question asked.
    Uses LLM-as-judge to score relevance from 0.0 to 1.0.
    """

    def __init__(self):
        self.client = GroqClient()

    def evaluate(self, question: str, answer: str) -> dict:
        prompt = f"""You are an expert evaluator. Rate how relevant the given answer is to the question.

Question:
{question}

Answer:
{answer}

Instructions:
- A score of 1.0 means the answer directly and completely addresses the question
- A score of 0.5 means the answer is partially relevant
- A score of 0.0 means the answer is completely off-topic
- Respond with ONLY a JSON object in this exact format:
{{"score": 0.0 to 1.0, "reason": "one sentence explanation"}}

Do not add any text outside the JSON."""

        try:
            response = self.client.generate(prompt, temperature=0.0, max_tokens=150)
            import json
            clean = response.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
            return {
                "relevance_score": round(float(result.get("score", 0.0)), 4),
                "reason": result.get("reason", "")
            }
        except Exception as e:
            return {
                "relevance_score": None,
                "reason": f"Evaluation failed: {str(e)}"
            }