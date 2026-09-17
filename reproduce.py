"""Rebuild and check the numerical results without any parent workspace files."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SCRIPTS = (
    "verify_drift.py", "build_paper_assets.py", "build_measurement_assets.py",
    "build_feedback_assets.py", "build_joint_assets.py",
    "build_information_assets.py", "verify_restricted_optimality.py",
    "build_revision_assets.py", "verify_precision.py",
)


def compare_json(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            compare_json(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_json(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)), path
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=2e-13), (
            path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def main():
    started = time.perf_counter()
    for name in ("generated", "figures", "checks"):
        (ROOT / name).mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "input_manifest.json").read_text())
    for name, expected_hash in manifest["sha256"].items():
        actual_hash = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Input checksum mismatch: {name}"
    observations = json.loads((ROOT / "data/accounting_observations.json").read_text())["observations"]
    assert len(observations) == manifest["observations"]
    for row in observations:
        assert sum(row["components"].values()) == row["rd_total"], row["id"]
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    stages = []
    for script in SCRIPTS:
        tick = time.perf_counter()
        with (ROOT / "checks" / (Path(script).stem + ".log")).open("w") as log:
            subprocess.run([sys.executable, "-B", str(ROOT / script)],
                           cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        stages.append({"script": script, "seconds": round(time.perf_counter() - tick, 3)})
        print(f"PASS {script}", flush=True)
    matched = []
    for expected in sorted((ROOT / "expected").iterdir()):
        actual = ROOT / "generated" / expected.name
        if expected.suffix == ".json":
            compare_json(json.loads(actual.read_text()), json.loads(expected.read_text()), expected.name)
        else:
            assert actual.read_bytes() == expected.read_bytes(), f"Text mismatch: {expected.name}"
        matched.append(expected.name)
    for name in ("accounting_bound_region.pdf", "accounting_bound_region.png", "information_frontier.pdf", "information_frontier.png"):
        assert (ROOT / "figures" / name).stat().st_size > 1000, name
    report = {
        "release": manifest["release"], "status": "pass",
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "sympy", "mpmath")},
        "input_files_verified": len(manifest["sha256"]),
        "accounting_rows_checked": len(observations),
        "matched_outputs": matched, "stages": stages,
        "seconds": round(time.perf_counter() - started, 3),
        "json_comparison": {"relative_tolerance": 1e-11, "absolute_tolerance": 2e-13},
        "tex_comparison": "byte-identical",
        "scope": "Reproduction of necessary-condition resource bounds and stated examples; not a certificate of unrestricted planner optimality.",
    }
    (ROOT / "checks/reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
