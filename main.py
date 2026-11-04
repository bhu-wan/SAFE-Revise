import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


def ask_number(prompt):
    """Keep asking until the user enters a valid, finite number."""
    while True:
        text = input(prompt).strip()

        try:
            number = Decimal(text)
            if number.is_finite():
                return number
        except InvalidOperation:
            pass

        print("Enter a number such as 19, -3, or 2.5.")


# Collect one question and its answer pair.
question = input("Research question / math problem: ").strip()

while not question:
    question = input("Please enter a question: ").strip()

reference = ask_number("Verified reference answer: ")
initial = ask_number("Model's initial answer: ")
revised = ask_number("Model's revised answer: ")


# Score each answer against the reference.
initial_correct = initial == reference
revised_correct = revised == reference


# Classify the transition.
if initial_correct and revised_correct:
    outcome = "PRESERVED_CORRECT"
elif not initial_correct and revised_correct:
    outcome = "SUCCESSFUL_CORRECTION"
elif initial_correct and not revised_correct:
    outcome = "HARMFUL_REVISION"
else:
    outcome = "UNREPAIRED_ERROR"


# Save a structured record beside this Python file.
record = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "source": "manual_evaluator_check",
    "question": question,
    "reference_answer": str(reference),
    "initial_answer": str(initial),
    "revised_answer": str(revised),
    "initial_correct": initial_correct,
    "revised_correct": revised_correct,
    "outcome": outcome,
}

output_file = Path(__file__).with_name("results.jsonl")

with output_file.open("a", encoding="utf-8") as file:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")


print(f"\nInitial answer correct: {initial_correct}")
print(f"Revised answer correct: {revised_correct}")
print(f"Revision outcome: {outcome}")
print(f"Record saved to: {output_file}")