from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def character_retention(reference: str, prediction: str) -> float:
    """Normalized OCR character retention based on Levenshtein distance."""

    if not reference:
        return float(not prediction)
    previous = list(range(len(prediction) + 1))
    for row, reference_character in enumerate(reference, 1):
        current = [row]
        for column, prediction_character in enumerate(prediction, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + int(reference_character != prediction_character),
                )
            )
        previous = current
    return max(0.0, 1.0 - previous[-1] / len(reference))


def summarize_ocr_records(records: Sequence[dict[str, str]]) -> dict[str, object]:
    rows = []
    for record in records:
        reference = str(record["reference_text"])
        row = {
            "id": str(record["id"]),
            "character_retention": character_retention(
                reference, str(record["prediction_text"])
            ),
        }
        if "stage_a_text" in record:
            row["stage_a_character_retention"] = character_retention(
                reference, str(record["stage_a_text"])
            )
            row["character_retention_gain"] = (
                row["character_retention"] - row["stage_a_character_retention"]
            )
        rows.append(row)
    result: dict[str, object] = {
        "samples": len(rows),
        "mean_character_retention": sum(
            float(row["character_retention"]) for row in rows
        )
        / max(len(rows), 1),
        "per_sample": rows,
    }
    with_stage_a = [row for row in rows if "stage_a_character_retention" in row]
    if with_stage_a:
        result["mean_stage_a_character_retention"] = sum(
            float(row["stage_a_character_retention"]) for row in with_stage_a
        ) / len(with_stage_a)
        result["mean_character_retention_gain"] = sum(
            float(row["character_retention_gain"]) for row in with_stage_a
        ) / len(with_stage_a)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = summarize_ocr_records(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
