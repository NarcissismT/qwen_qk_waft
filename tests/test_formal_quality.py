from qwen_qk_waft.formal_quality import character_retention, summarize_ocr_records


def test_ocr_character_retention_and_stage_a_gain() -> None:
    assert character_retention("document", "document") == 1.0
    assert character_retention("document", "docxment") == 0.875
    report = summarize_ocr_records(
        [
            {
                "id": "sample",
                "reference_text": "abcd",
                "stage_a_text": "abxx",
                "prediction_text": "abcx",
            }
        ]
    )
    assert report["mean_character_retention"] == 0.75
    assert report["mean_stage_a_character_retention"] == 0.5
    assert report["mean_character_retention_gain"] == 0.25
