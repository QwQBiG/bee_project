import unittest

from inference.processor import verified_pose_distribution


class AnalyzerStub:
    def __init__(self):
        self.calls = 0

    def analyze_pose_distribution(self, tracks):
        self.calls += 1
        return {"samples": len(tracks)}


class PoseReportingGuardTests(unittest.TestCase):
    def test_disabled_pose_returns_unknown_without_calling_legacy_analyzer(self):
        analyzer = AnalyzerStub()
        self.assertIsNone(verified_pose_distribution(analyzer, [object()], False))
        self.assertEqual(analyzer.calls, 0)

    def test_enabled_pose_uses_distribution_analyzer(self):
        analyzer = AnalyzerStub()
        self.assertEqual(
            verified_pose_distribution(analyzer, [object(), object()], True),
            {"samples": 2},
        )
        self.assertEqual(analyzer.calls, 1)


if __name__ == "__main__":
    unittest.main()
