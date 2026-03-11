import os
import json
import warnings
import logging
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("FlagEmbedding").setLevel(logging.ERROR)

load_dotenv()  # loads GROQ_API_KEY from .env file

warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from llm.groq_client import GroqClient
from evaluators.accuracy import AccuracyEvaluator
from evaluators.hallucination import HallucinationEvaluator
from evaluators.relevance import RelevanceEvaluator
from evaluators.coherence import CoherenceEvaluator


def load_test_cases(path="data/test_cases.json"):
    with open(path, "r") as f:
        return json.load(f)


def run_evaluation():
    print("=" * 60)
    print("   LLM Evaluation Framework")
    print("=" * 60)

    # Load test cases
    test_cases = load_test_cases()
    print(f"\nLoaded {len(test_cases)} test cases\n")

    # Initialize components
    print("Initializing evaluators...")
    groq = GroqClient()
    accuracy_eval = AccuracyEvaluator()
    hallucination_eval = HallucinationEvaluator()
    relevance_eval = RelevanceEvaluator()
    coherence_eval = CoherenceEvaluator()
    print("Ready.\n")

    results = []

    for i, case in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] Evaluating: {case['question'][:60]}...")

        # Generate answer using Groq
        prompt = f"""Answer the following question using only the context provided.

Context: {case['context']}
Question: {case['question']}
Answer:"""

        actual_answer = groq.generate(prompt, temperature=0.1, max_tokens=256)

        # Run all evaluators
        accuracy = accuracy_eval.evaluate(case["expected_answer"], actual_answer)
        hallucination = hallucination_eval.evaluate(case["context"], case["question"], actual_answer)
        relevance = relevance_eval.evaluate(case["question"], actual_answer)
        coherence = coherence_eval.evaluate(actual_answer)

        # Compute overall score (average of all metrics)
        scores = [
            accuracy["accuracy_score"],
            hallucination["hallucination_score"] if hallucination["hallucination_score"] is not None else 0,
            relevance["relevance_score"] if relevance["relevance_score"] is not None else 0,
            coherence["coherence_score"] if coherence["coherence_score"] is not None else 0
        ]
        overall = round(sum(scores) / len(scores), 4)

        result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "actual_answer": actual_answer,
            "accuracy": accuracy,
            "hallucination": hallucination,
            "relevance": relevance,
            "coherence": coherence,
            "overall_score": overall
        }

        results.append(result)

        print(f"   Accuracy: {accuracy['accuracy_score']} | "
              f"Hallucination: {hallucination['hallucination_score']} | "
              f"Relevance: {relevance['relevance_score']} | "
              f"Coherence: {coherence['coherence_score']} | "
              f"Overall: {overall}")

    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/eval_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("   SUMMARY")
    print("=" * 60)
    avg_accuracy = sum(r["accuracy"]["accuracy_score"] for r in results) / len(results)
    avg_hallucination = sum(r["hallucination"]["hallucination_score"] or 0 for r in results) / len(results)
    avg_relevance = sum(r["relevance"]["relevance_score"] or 0 for r in results) / len(results)
    avg_coherence = sum(r["coherence"]["coherence_score"] or 0 for r in results) / len(results)
    avg_overall = sum(r["overall_score"] for r in results) / len(results)

    print(f"  Avg Accuracy:       {avg_accuracy:.4f}")
    print(f"  Avg Hallucination:  {avg_hallucination:.4f}")
    print(f"  Avg Relevance:      {avg_relevance:.4f}")
    print(f"  Avg Coherence:      {avg_coherence:.4f}")
    print(f"  Avg Overall Score:  {avg_overall:.4f}")
    print(f"\n  Results saved to: {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_evaluation()