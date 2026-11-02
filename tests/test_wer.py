"""
WER and CER of bench/wer.py: known values per condition on the toy manifest (ASR-7, EXP-4).
"""

import json

from pathlib import Path

import numpy as np
import pytest

from app.core.backends import build_backend
from app.core.config import get_asr_settings, Settings
from bench import wer

MANIFEST = Path(__file__).resolve().parent / "data" / "questions.jsonl"


@pytest.fixture
def fake_text(monkeypatch):
    """The fake backend recognizes the given text in any speech."""
    def configure(text: str):
        monkeypatch.setenv("FAKE_TEXT", text)
        get_asr_settings.cache_clear()
        return build_backend(Settings(_env_file=None, MODEL_BACKEND="fake"))

    yield configure
    get_asr_settings.cache_clear()


@pytest.mark.parametrize(
    "reference, hypothesis, distance",
    [("", "", 0), ("abc", "abc", 0), ("abc", "", 3), ("", "ab", 2), ("kitten", "sitting", 3), ("где", "дверь", 4)],
)
def test_edit_distance(reference, hypothesis, distance):
    assert wer.edit_distance(reference, hypothesis) == distance
    assert wer.edit_distance(reference.split(), hypothesis.split()) <= max(len(reference.split()), len(hypothesis.split()))


def test_score_item_normalizes_both_sides():
    # Case, ё and punctuation are not errors (EXP-4: after the ASR-4 normalization)
    item = wer.score_item("Где ёлка?", "где елка")
    assert (item["word_errors"], item["char_errors"]) == (0, 0)

    item = wer.score_item("где стоит дверь", "Где дверь!")
    assert (item["words"], item["word_errors"]) == (3, 1)
    assert (item["chars"], item["char_errors"]) == (15, 6)


def test_wer_by_condition_on_toy_manifest(fake_text):
    # Every speech is heard as "дверь": the errors are known by hand
    scores = wer.evaluate(fake_text("Дверь."), MANIFEST)

    hypotheses = {item["id"]: item["hypothesis"] for item in scores["items"]}
    assert hypotheses == {"q001": "дверь", "q002": "дверь", "q003": "дверь", "q004": "дверь"}

    quiet, noise = scores["conditions"]["quiet"], scores["conditions"]["street_noise"]
    # quiet: дверь/дверь - 0 and 0; стул/дверь - 1 word, 5 chars
    assert (quiet["n"], quiet["words"], quiet["word_errors"], quiet["wer"]) == (2, 2, 1, 0.5)
    assert (quiet["chars"], quiet["char_errors"], quiet["cer"]) == (9, 5, round(5 / 9, 4))
    # street_noise: где/дверь - 1 word, 4 chars; окно/дверь - 1 word, 5 chars
    assert (noise["n"], noise["words"], noise["word_errors"], noise["wer"]) == (2, 2, 2, 1.0)
    assert (noise["chars"], noise["char_errors"], noise["cer"]) == (7, 9, round(9 / 7, 4))

    overall = scores["overall"]
    assert (overall["wer"], overall["cer"]) == (0.75, 0.875)


def test_wer_silence_is_a_deletion(fake_text, tmp_path):
    # No speech: the model is not called and every reference word is an error
    silence = tmp_path / "silence.wav"
    silence.write_bytes(
        b"RIFF" + (36 + 32000).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
        + np.array([1, 1], dtype="<u2").tobytes() + (16000).to_bytes(4, "little") + (32000).to_bytes(4, "little")
        + np.array([2, 16], dtype="<u2").tobytes() + b"data" + (32000).to_bytes(4, "little") + bytes(32000)
    )
    manifest = tmp_path / "questions.jsonl"
    manifest.write_text(json.dumps({"id": "q1", "audio": "silence.wav", "text": "где дверь", "condition": "quiet"}) + "\n")

    scores = wer.evaluate(fake_text("где дверь"), manifest)
    assert scores["items"][0]["hypothesis"] == ""
    assert (scores["conditions"]["quiet"]["wer"], scores["conditions"]["quiet"]["cer"]) == (1.0, 1.0)
    # An absent condition has no rates
    assert scores["conditions"]["street_noise"] == {
        "n": 0, "words": 0, "word_errors": 0, "wer": None, "chars": 0, "char_errors": 0, "cer": None,
    }


def test_manifest_with_unknown_condition_fails(tmp_path):
    manifest = tmp_path / "questions.jsonl"
    manifest.write_text(json.dumps({"id": "q1", "audio": "a.wav", "text": "x", "condition": "rain"}) + "\n")
    with pytest.raises(ValueError, match="condition must be one of"):
        wer.read_manifest(manifest)


def test_wer_script_writes_json(fake_text, tmp_path, capsys):
    fake_text("дверь")
    out = tmp_path / "wer.json"
    assert wer.main(["--manifest", str(MANIFEST), "--backend", "fake", "--out", str(out)]) == 0

    result = json.loads(out.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert (result["service"], result["backend"], result["model"], result["device"]) == ("asr-service", "fake", "fake", "cpu")
    assert set(result["conditions"]) == {"quiet", "street_noise"}
    assert result["overall"]["wer"] == 0.75
