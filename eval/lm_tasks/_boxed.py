"""Shared boxed-answer extraction and semantic verification for math tasks."""


BOXED_PREFIX = r"\boxed{"


def extract_boxed_contents(text: str) -> list[str]:
    contents = []
    search_from = 0
    while True:
        prefix_start = text.find(BOXED_PREFIX, search_from)
        if prefix_start < 0:
            return contents

        content_start = prefix_start + len(BOXED_PREFIX)
        depth = 1
        cursor = content_start
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1

        if depth == 0:
            contents.append(text[content_start : cursor - 1])
            search_from = cursor
        else:
            search_from = content_start


def last_boxed_content(text: str) -> str | None:
    contents = extract_boxed_contents(text)
    return contents[-1] if contents else None


def _math_verify(gold: str, target: str) -> bool:
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise RuntimeError(
            "MATH scoring requires lm-eval[math] (math_verify and ANTLR 4.11)"
        ) from exc
    return bool(verify(gold=parse(gold), target=parse(target)))


def _verify_boxed(gold: str, prediction: str | None) -> int:
    if prediction is None:
        return 0
    # math_verify does not reliably parse bare LaTeX such as \sqrt{17}.
    return int(
        _math_verify(
            rf"\boxed{{{gold}}}",
            rf"\boxed{{{prediction}}}",
        )
    )
