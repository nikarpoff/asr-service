"""
Text normalization of ASR-4: the same rule as the gateway intents (PLT-4).

The contract table tests/contract/normalize.cases.json pins it for both sides.
"""

import unicodedata


def normalize(text: str) -> str:
    """Lower case, ё as е, punctuation (P*) as spaces, symbols (S*) kept, single spaces, no outer ones."""
    text = text.lower().replace("ё", "е")
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    # split() takes any Unicode space: tabs, line breaks and no-break spaces
    return " ".join(text.split())
