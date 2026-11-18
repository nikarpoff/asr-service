"""
Reference examples of tests/contract pass the API schemas (SVC-12); the gateway keeps the same files.
"""

import json

from pathlib import Path
from typing import Type

import pytest

from pydantic import BaseModel, ValidationError

from app.api_v1.schemas import ErrorResponse, InfoResponse, RecognizeResponse
from bench.latency import BenchResult

CONTRACT = Path(__file__).resolve().parent / "contract"

SCHEMAS = {
    "info.response.json": InfoResponse,
    "recognize.response.json": RecognizeResponse,
    "recognize.silence.response.json": RecognizeResponse,
    "error.response.json": ErrorResponse,
    "bench.result.json": BenchResult,
}
CASE_TABLES = {"normalize.cases.json"}           # tables of cases, read by their own tests


def load(name: str) -> dict:
    return json.loads((CONTRACT / name).read_text(encoding="utf-8"))


def test_every_contract_file_is_checked():
    assert {path.name for path in CONTRACT.iterdir()} == set(SCHEMAS) | CASE_TABLES


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_example_matches_schema(name):
    schema: Type[BaseModel] = SCHEMAS[name]
    example = load(name)
    # extra="forbid": a field the schema does not know fails, and the dump gives the same JSON back
    assert schema.model_validate(example).model_dump(mode="json") == example


@pytest.mark.parametrize("schema", [RecognizeResponse, InfoResponse, ErrorResponse, BenchResult])
def test_schemas_forbid_unknown_fields(schema):
    assert schema.model_config.get("extra") == "forbid"


def test_recognize_examples_are_consistent():
    speech, silence = load("recognize.response.json"), load("recognize.silence.response.json")

    assert speech["speech"] is True and speech["text"]
    assert "inference_ms" in speech["timings"]

    # Without speech the model is not called: no text and no inference stage
    assert silence["speech"] is False and silence["text"] == silence["normalized"] == ""
    assert "inference_ms" not in silence["timings"]

    for example in (speech, silence):
        assert example["rtf"] == round(example["timings"]["total_ms"] / example["audio_duration_ms"], 3)


def test_wrong_example_fails():
    example = load("recognize.response.json")
    with pytest.raises(ValidationError):
        RecognizeResponse.model_validate({**example, "confidence": 0.9})
    with pytest.raises(ValidationError):
        RecognizeResponse.model_validate({key: value for key, value in example.items() if key != "speech"})
