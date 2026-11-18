"""
Normalization of ASR-4: every case of the contract table shared with the gateway (PLT-4).
"""

import json

from pathlib import Path

import pytest

from app.core.text import normalize

CASES = json.loads((Path(__file__).resolve().parent / "contract" / "normalize.cases.json").read_text())["cases"]


def test_contract_table_is_complete():
    names = [case["name"] for case in CASES]
    assert len(names) >= 20
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_normalize_contract_case(case):
    assert normalize(case["input"]) == case["expected"]


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_normalize_is_idempotent(case):
    assert normalize(normalize(case["input"])) == normalize(case["input"])
