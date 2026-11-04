import json
import time
from pathlib import Path
from gpt4all import GPT4All
from fractions import Fraction

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "qwen2-1_5b-instruct-q4_0.gguf"
SETTINGS = {"max_tokens": 512, "temp": 0.0, "top_k": 1}

question = input("Enter your reasoning question: ").strip()

if not question:
    raise SystemExit("A question is required.")

print("\nLoading the model...")
model = GPT4All(
    MODEL,
    model_path=str(MODEL_DIR),
    device="cpu",
    n_ctx=2048,
)


def generate(prompt):
    start = time.perf_counter()

    # Save the complete output produced by the model.
    raw_response = model.generate(prompt, **SETTINGS)

    # Remove any extra conversation after a role marker.
    cleaned_response = raw_response
    for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        cleaned_response = cleaned_response.split(marker, 1)[0]

    elapsed = round(time.perf_counter() - start, 2)

    return raw_response, cleaned_response.strip(), elapsed


# Ask the model for its first answer.
initial_prompt = (
    "<|im_start|>system\n"
    "Solve the problem carefully. End with FINAL: followed by your answer."
    "<|im_end|>\n"
    f"<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

print("Generating the initial answer...")
initial_raw, initial, initial_seconds = generate(initial_prompt)
print("\nINITIAL ANSWER:\n", initial)


# Ask the same model to review its initial answer.
revision_prompt = (
    initial_prompt
    + initial
    + "<|im_end|>\n"
    "<|im_start|>user\n"
    "Review your previous solution for reasoning or calculation errors. "
    "Provide your final solution, ending with FINAL: followed by your answer."
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

print("\nGenerating the revision...")
revised_raw, revised, revision_seconds = generate(revision_prompt)
print("\nREVISED ANSWER:\n", revised)


record = {
    "source": "local_model",
    "model": MODEL,
    "settings": SETTINGS,
    "question": question,
    "initial_response": initial,
    "initial_raw": initial_raw,
    "revised_response": revised,
    "revised_raw": revised_raw,
    "initial_seconds": initial_seconds,
    "revision_seconds": revision_seconds,
}

output = ROOT / "baseline_outputs.jsonl"

with output.open("a", encoding="utf-8") as file:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")

model.close()
print(f"\nSaved model outputs to: {output}")