import hashlib
import random
import re


def _clean_gpqa_text(text: str | None) -> str:
    if text is None:
        return " "
    text = re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". "))
    return text.replace("  ", " ")


def process_gpqa_docs(dataset):
    def process_doc(doc: dict) -> dict:
        choices = [
            _clean_gpqa_text(doc["Incorrect Answer 1"]),
            _clean_gpqa_text(doc["Incorrect Answer 2"]),
            _clean_gpqa_text(doc["Incorrect Answer 3"]),
            _clean_gpqa_text(doc["Correct Answer"]),
        ]
        correct = _clean_gpqa_text(doc["Correct Answer"])
        seed = int.from_bytes(
            hashlib.sha256(doc["Question"].encode("utf-8")).digest()[:8], "big"
        )
        random.Random(seed).shuffle(choices)
        return {
            "question": doc["Question"],
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": choices,
            "answer": f"({chr(ord('A') + choices.index(correct))})",
        }

    return dataset.map(process_doc)


def gpqa_doc_to_text(doc: dict) -> str:
    return (
        "Example:\n"
        "Question: At standard atmospheric pressure, at what temperature "
        "does pure water freeze?\n"
        "Choices:\n"
        "(A) 100 degrees Celsius\n"
        "(B) -273.15 degrees Celsius\n"
        "(C) 50 degrees Celsius\n"
        "(D) 0 degrees Celsius\n\n"
        "Reasoning: At standard atmospheric pressure, the freezing point "
        "of pure water is 0 degrees Celsius.\n"
        "The answer is (D).\n\n"
        "Now answer the following question.\n"
        f"Question: {doc['question']}\n"
        "Choices:\n"
        f"(A) {doc['choice1']}\n"
        f"(B) {doc['choice2']}\n"
        f"(C) {doc['choice3']}\n"
        f"(D) {doc['choice4']}\n\n"
        "Please reason step by step. End your response with a line in the form "
        '"The answer is (<letter>)."'
    )
