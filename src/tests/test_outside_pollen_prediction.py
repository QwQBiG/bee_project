from types import SimpleNamespace

import numpy as np

from behavior.outside_pollen import OutsidePollenAnalyzer


def test_low_candidate_ratio_produces_low_confidence_warning_without_roi():
    analyzer = OutsidePollenAnalyzer({
        "min_samples": 1,
        "min_analyzable_tracks": 2,
        "min_pollen_ratio": 0.15,
    })
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    for track_id in (1, 2):
        analyzer.update(frame, [SimpleNamespace(
            track_id=track_id,
            bbox=[10.0 * track_id, 10.0, 12.0, 16.0],
            center=(10.0 * track_id + 6.0, 18.0),
        )], frame_id=1)

    report = analyzer.build_report()
    assessment = report["nutrition_assessment"]
    assert report["pollen_candidate_ratio"] == 0.0
    assert assessment["status"] == "warning"
    assert assessment["confidence"] == "low"
