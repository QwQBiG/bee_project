"""Contract tests for the September 2026 submission JSON format."""

from __future__ import annotations

import json

import pytest

from inference.submission_contract import (
    DETECTION_FIELDS,
    TRACKING_FIELDS,
    build_detection_result,
    build_tracking_result,
    frame_id_from_path,
    write_result,
)


def test_frame_id_uses_filename_numeric_suffix():
    assert frame_id_from_path("C:/Test/Inside/detection/images/im_0007.jpg") == 7
    assert frame_id_from_path("frame_1000.jpg") == 1000
    with pytest.raises(ValueError):
        frame_id_from_path("frame_zero.jpg")


def test_detection_result_is_sorted_rounded_and_complete():
    result = build_detection_result(
        "438137", "Outside-detection", 2, 412300,
        [
            {"frame_id": 2, "class_id": 0, "conf": 0.81234567,
             "bbox": [4.555, 5.555, 6.555, 7.555]},
            {"frame_id": 1, "class_id": 0, "conf": 0.7,
             "bbox": [1, 2, 3, 4]},
            {"frame_id": 1, "class_id": 0, "conf": 0.9,
             "bbox": [8, 9, 10, 11]},
        ],
    )
    assert result["fields"] == DETECTION_FIELDS
    assert result["num_records"] == len(result["detections"]) == 3
    assert [row[:3] for row in result["detections"]] == [
        [1, 0, 0.9], [1, 0, 0.7], [2, 0, 0.812346],
    ]
    assert result["detections"][2][3:7] == [4.55, 5.55, 6.55, 7.55]
    assert all(row[-1] == 1 for row in result["detections"])


def test_detection_caps_outside_at_200_per_frame():
    records = [
        {"frame_id": 1, "conf": index / 300, "bbox": [0, 0, 1, 1]}
        for index in range(250)
    ]
    result = build_detection_result(
        "438137", "Outside-detection", 1, 10, records)
    assert result["num_records"] == 200
    assert result["detections"][0][2] > result["detections"][-1][2]


def test_tracking_result_uses_fixed_mot_fields_and_order():
    result = build_tracking_result(
        "438137", "Inside-tracking", 2, 553100,
        [
            {"frame_id": 2, "track_id": 1, "bbox": [1, 2, 3, 4]},
            {"frame_id": 1, "track_id": 2, "bbox": [5, 6, 7, 8]},
            {"frame_id": 1, "track_id": 1, "bbox": [9, 10, 11, 12]},
        ],
    )
    assert result["fields"] == TRACKING_FIELDS
    assert result["num_tracks"] == 2
    assert result["num_records"] == 3
    assert [(row[0], row[1]) for row in result["tracks"]] == [
        (1, 1), (1, 2), (2, 1),
    ]
    assert all(row[6:] == [1, -1, -1, -1] for row in result["tracks"])


def test_write_result_is_single_line_utf8(tmp_path):
    payload = build_detection_result(
        "438137", "Inside-detection", 1, 1, [])
    output = write_result(tmp_path / "result.json", payload)
    raw = output.read_text(encoding="utf-8")
    assert "\n" not in raw and "\r" not in raw
    assert json.loads(raw) == payload


@pytest.mark.parametrize("team_id", ["12345", "1234567", "ABC123"])
def test_invalid_team_id_is_rejected(team_id):
    with pytest.raises(ValueError):
        build_detection_result(team_id, "Inside-detection", 1, 0, [])


def test_non_finite_detection_values_are_rejected():
    with pytest.raises(ValueError, match="conf"):
        build_detection_result(
            "438137", "Inside-detection", 1, 0,
            [{"frame_id": 1, "conf": float("nan"),
              "bbox": [0, 0, 1, 1]}],
        )
    with pytest.raises(ValueError, match="finite"):
        build_detection_result(
            "438137", "Inside-detection", 1, 0,
            [{"frame_id": 1, "conf": 0.5,
              "bbox": [0, 0, float("inf"), 1]}],
        )


def test_duplicate_track_id_in_one_frame_is_rejected():
    rows = [
        {"frame_id": 1, "track_id": 7, "bbox": [0, 0, 1, 1]},
        {"frame_id": 1, "track_id": 7, "bbox": [2, 2, 1, 1]},
    ]
    with pytest.raises(ValueError, match="at most once"):
        build_tracking_result("438137", "Inside-tracking", 1, 0, rows)
