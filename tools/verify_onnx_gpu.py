"""Verify actual CUDA node execution, not just provider availability."""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "output/gpu_verification"
    output.mkdir(parents=True, exist_ok=True)
    cache = root / ".runtime/cuda-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache))
    ort.preload_dlls()
    report = []
    for name in ("bee_inside_ft_v1.onnx", "hive_entrance_bee_yolov8n.onnx"):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.enable_profiling = True
        options.profile_file_prefix = str(output / name)
        session = ort.InferenceSession(str(root / "artifacts/models" / name),
                                       sess_options=options,
                                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("CUDA unavailable")
        entry = session.get_inputs()[0]
        outputs = session.run(None, {entry.name: np.zeros(entry.shape, np.float32)})
        assert all(np.isfinite(x).all() for x in outputs)
        trace = Path(session.end_profiling())
        events = json.loads(trace.read_text(encoding="utf-8"))
        counts = Counter(e.get("args", {}).get("provider") for e in events
                         if e.get("cat") == "Node" and e.get("args", {}).get("provider"))
        if not counts["CUDAExecutionProvider"]:
            raise RuntimeError("No CUDA nodes executed")
        report.append({"model": name, "node_execution_counts": dict(counts),
                       "trace": str(trace), "onnxruntime": ort.__version__})
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
