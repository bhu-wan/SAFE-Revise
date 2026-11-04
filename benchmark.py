import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from gpt4all import GPT4All

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "gsm8k_train.jsonl"
MODEL = "qwen2-1_5b-instruct-q4_0.gguf"

SETTINGS = {"max_tokens": 128, "temp": 0.0, "top_k": 1}
NUM_QUESTIONS = 3

if not DATASET.exists():
    raise SystemExit("Missing gsm8k_train.jsonl in the SAFE-Revise folder.")

# Read the same first three questions each time.
with DATASET.open(encoding="utf-8") as file:
    cases = [json.loads(line) for _, line in zip(range(NUM_QUESTIONS), file)]

if len(cases) != NUM_QUESTIONS:
    raise SystemExit("The dataset has fewer questions than requested.")

run_id = (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + "-"
    + uuid.uuid4().hex[:8]
)

dataset_sha256 = hashlib.sha256(DATASET.read_bytes()).hexdigest()
output = ROOT / f"gsm8k_pilot_{run_id}.jsonl"

counts = {
    "preserved_correct": 0,
    "successful_correction": 0,
    "harmful_revision": 0,
    "unrepaired_error": 0,
    "unparsed": 0,
}


def clean_response(raw):
    for marker in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        raw = raw.split(marker, 1)[0]
    return raw.strip()


def final_number(response):
    # Accept only a final line in the requested format.
    lines = response.strip().splitlines()
    if not lines:
        return None

    pattern = r"FINAL\s*:\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\.?"
    match = re.fullmatch(pattern, lines[-1].strip(), flags=re.IGNORECASE)

    if not match:
        return None

    return Decimal(match.group(1).replace(",", ""))


def generate(model, prompt):
    start = time.perf_counter()
    raw = model.generate(prompt, **SETTINGS)
    elapsed = round(time.perf_counter() - start, 2)
    return raw, clean_response(raw), elapsed


print("Loading the local model...")
model = GPT4All(
    MODEL,
    model_path=str(ROOT / "models"),
    allow_download=False,
    device="cpu",
    n_ctx=2048,
)

try:
    # A new file is created for each run.
    with output.open("x", encoding="utf-8") as results:
        for index, case in enumerate(cases, start=1):
            question = case["question"]

            # The reference answer is used only for scoring.
            reference_text = case["answer"].rsplit("####", 1)[1].strip()
            reference = Decimal(reference_text.replace(",", ""))

            initial_prompt = (
                "<|im_start|>system\n"
                "Answer the question. On the last line write only FINAL: followed by "
                "the numerical answer, for example FINAL: 42."
                "<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            print(
                f"\nQuestion {index}/{NUM_QUESTIONS}: generating initial answer...",
                flush=True,
            )
            initial_raw, initial, initial_seconds = generate(model, initial_prompt)

            revision_prompt = (
                initial_prompt
                + initial
                + "<|im_end|>\n"
                "<|im_start|>user\n"
                "Review your previous answer. On the last line write only FINAL: "
                "followed by the numerical answer."
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            print("Generating revision...", flush=True)
            revised_raw, revised, revised_seconds = generate(model, revision_prompt)

            initial_number = final_number(initial)
            revised_number = final_number(revised)

            if initial_number is None or revised_number is None:
                outcome = "unparsed"
            elif initial_number == reference and revised_number == reference:
                outcome = "preserved_correct"
            elif initial_number != reference and revised_number == reference:
                outcome = "successful_correction"
            elif initial_number == reference and revised_number != reference:
                outcome = "harmful_revision"
            else:
                outcome = "unrepaired_error"

            counts[outcome] += 1

            record = {
                "run_id": run_id,
                "dataset": "GSM8K train",
                "dataset_sha256": dataset_sha256,
                "dataset_row": index,
                "model": MODEL,
                "settings": SETTINGS,
                "question": question,
                "reference": str(reference),
                "initial_prompt": initial_prompt,
                "revision_prompt": revision_prompt,
                "initial_raw": initial_raw,
                "initial_response": initial,
                "revised_raw": revised_raw,
                "revised_response": revised,
                "initial_parsed": (
                    str(initial_number) if initial_number is not None else None
                ),
                "revised_parsed": (
                    str(revised_number) if revised_number is not None else None
                ),
                "initial_seconds": initial_seconds,
                "revision_seconds": revised_seconds,
                "outcome": outcome,
            }

            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()

            print(
                f"Reference: {reference} | "
                f"Initial: {initial_number} | Revised: {revised_number}"
            )
            print(f"Outcome: {outcome}", flush=True)

finally:
    model.close()

print("\nPilot summary:")
for outcome, count in counts.items():
    print(f"  {outcome}: {count}")

print(f"Saved detailed results to: {output}")