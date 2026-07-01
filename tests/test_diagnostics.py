"""Environment diagnostics / doctor command."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audience_tracker import diagnostics as dx  # noqa: E402


def test_report_shape_and_never_raises():
    rep = dx.report()
    assert rep["python_ok"] is True  # tests run on >= 3.10
    for key in ("python_version", "platform", "modules", "cuda", "capabilities"):
        assert key in rep
    # Every capability's modules are represented in the module table.
    for cap, mods in dx.CAPABILITIES.items():
        assert cap in rep["capabilities"]
        for m in mods:
            assert m in rep["modules"]


def test_serve_capability_available_in_test_env():
    # The dev/test env has fastapi+uvicorn+websockets, so 'serve' is satisfied.
    # The heavier detect stack varies by machine; assert the report is internally
    # consistent rather than assuming a CPU-only CI environment.
    rep = dx.report()
    assert dx.capability_ok(rep, "serve") is True
    missing = dx.missing_modules(rep, "detect")
    if dx.capability_ok(rep, "detect"):
        assert missing == []
    else:
        assert missing
        assert all(rep["modules"][name]["available"] is False for name in missing)


def test_format_report_is_readable():
    text = dx.format_report(dx.report())
    assert "Python" in text and "Capabilities:" in text


def test_doctor_require_exit_codes():
    rep = dx.report()
    assert dx.main(["--require", "serve"]) == 0
    assert dx.main(["--require", "detect"]) == (
        0 if dx.capability_ok(rep, "detect") else 1
    )
    assert dx.main(["--require-cuda"]) == (
        0 if rep["cuda"]["cuda_available"] else 1
    )


if __name__ == "__main__":
    test_report_shape_and_never_raises()
    test_serve_capability_available_in_test_env()
    test_format_report_is_readable()
    test_doctor_require_exit_codes()
    print("ok")
