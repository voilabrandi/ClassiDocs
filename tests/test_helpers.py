#Import Libraries
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


from src.helpers import (
    write_fasttext,
    undersample_quantitative,
    ramp,
    normalize_doc_id,
    pick_by_pos,
)


label_qual = "qualitative"
label_quant = "quantitative"



# Test if FastText training file is formatted properly
def test_write_fasttext_writes_valid_lines(tmp_path):
    df = pd.DataFrame({
        "text": ["hello world", "machine learning paper"],
        "label": ["cs.ai", "cs.lg"],
    })

    out_path = tmp_path / "train.txt"
    write_fasttext(df, "text", "label", out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert lines[0].startswith("__label__cs.ai ")
    assert lines[1].startswith("__label__cs.lg ")

# Ensure rows with empty text or labels are excluded
def test_write_fasttext_skips_empty_text_or_label(tmp_path):
    df = pd.DataFrame({
        "text": ["valid text", "", "another valid"],
        "label": ["cs.ai", "cs.db", ""],
    })

    out_path = tmp_path / "train.txt"
    write_fasttext(df, "text", "label", out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 1
    assert lines[0].startswith("__label__cs.ai ")


# Test if quantitative class is reduced based on ratio
def test_undersample_quantitative_reduces_quant_class():
    df = pd.DataFrame({
        "doc_id": [1, 2, 3, 4, 5, 6, 7],
        "title": ["a"] * 7,
        "abstract": ["x"] * 7,
        "label": [
            "qualitative", "qualitative",
            "quantitative", "quantitative", "quantitative", "quantitative", "quantitative"
        ],
    })

    out = undersample_quantitative(
        df,
        qual_label="qualitative",
        quant_label="quantitative",
        ratio=1.5,
        random_state=42,
    )

    counts = out["label"].str.lower().value_counts()
    assert counts["qualitative"] == 2
    assert counts["quantitative"] == 3  # floor(2 * 1.5) = 3

# Test if function returns original dataset if one class is missing
def test_undersample_quantitative_returns_unchanged_if_class_missing():
    df = pd.DataFrame({
        "doc_id": [1, 2],
        "title": ["a", "b"],
        "abstract": ["x", "y"],
        "label": ["qualitative", "qualitative"],
    })

    out = undersample_quantitative(
        df,
        qual_label="qualitative",
        quant_label="quantitative",
        ratio=1.5,
        random_state=42,
    )

    assert len(out) == 2
    assert "label_norm" not in out.columns



# Test if ramp function transitions correctly from start to end threshold
def test_ramp_start_middle_end():
    assert ramp(1, 0.70, 0.92, 30) == 0.70

    mid = ramp(15, 0.70, 0.92, 30)
    assert 0.70 < mid < 0.92

    assert ramp(30, 0.70, 0.92, 30) == 0.92



#Ensure whitespace is removed from document IDs
def test_normalize_doc_id_strips_whitespace():
    df = pd.DataFrame({
        "doc_id": [" 1 ", "2 ", " 3"],
        "title": ["a", "b", "c"],
        "abstract": ["x", "y", "z"],
    })

    out = normalize_doc_id(df)
    assert list(out["doc_id"]) == ["1", "2", "3"]


## Verify valid indices are selected and duplicates/invalid values are ignored
def test_pick_by_pos_valid_and_deduplicated():
    df = pd.DataFrame({
        "doc_id": [1, 2, 3, 4],
        "title": ["a", "b", "c", "d"],
    })

    out = pick_by_pos(df, [1, 2, 2, 4, 10, 0, -1])

    assert list(out["doc_id"]) == [1, 2, 4]