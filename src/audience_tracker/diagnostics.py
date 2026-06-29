"""Environment diagnostics — power the ``audience-tracker doctor`` command.

Checks which optional dependency stacks are installed and whether a CUDA GPU is
visible to PyTorch, then reports it in a human-friendly way. Used both directly
by operators and by the Windows installer to verify a setup and to gate its exit
code. Never imports the heavy stacks unless they're present, and never raises —
a missing dependency is data, not an error.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import sys
from typing import Optional

# capability -> the import names it requires.
CAPABILITIES: dict[str, list[str]] = {
    "serve": ["fastapi", "uvicorn", "websockets"],          # REST/WS API + mock
    "detect": ["numpy", "cv2", "ultralytics", "supervision", "torch", "torchvision"],
    "reid": ["torchreid"],                                   # OSNet x1.0
    "agent": ["numpy", "cv2", "websockets"],                 # venue Capture Agent
    "deploy": ["modal"],
}

# import name -> candidate distribution names (for version lookup / pip hints).
_DIST_NAMES: dict[str, list[str]] = {
    "cv2": ["opencv-python-headless", "opencv-python"],
    "torchreid": ["torchreid", "deep-person-reid"],
}

MIN_PYTHON = (3, 10)


def _can_import(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _version(import_name: str) -> Optional[str]:
    for dist in _DIST_NAMES.get(import_name, [import_name]):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            return None
    return None


def check_module(import_name: str) -> dict:
    available = _can_import(import_name)
    return {
        "name": import_name,
        "available": available,
        "version": _version(import_name) if available else None,
    }


def check_cuda() -> dict:
    """Report PyTorch's view of the GPU. Imports torch only if it's installed."""
    out = {"torch_installed": False, "cuda_available": False, "device_name": None, "torch_version": None}
    if not _can_import("torch"):
        return out
    try:
        import torch  # heavy, but the user explicitly asked to check

        out["torch_installed"] = True
        out["torch_version"] = getattr(torch, "__version__", None)
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - defensive
        out["error"] = str(exc)
    return out


def report() -> dict:
    modules = sorted({m for mods in CAPABILITIES.values() for m in mods})
    mod_reports = {m: check_module(m) for m in modules}
    capabilities = {
        cap: all(mod_reports[m]["available"] for m in mods)
        for cap, mods in CAPABILITIES.items()
    }
    py = sys.version_info
    return {
        "python_version": f"{py.major}.{py.minor}.{py.micro}",
        "python_ok": (py.major, py.minor) >= MIN_PYTHON,
        "platform": platform.platform(),
        "modules": mod_reports,
        "cuda": check_cuda(),
        "capabilities": capabilities,
    }


def capability_ok(rep: dict, capability: str) -> bool:
    return bool(rep.get("capabilities", {}).get(capability, False))


def missing_modules(rep: dict, capability: str) -> list[str]:
    return [m for m in CAPABILITIES.get(capability, []) if not rep["modules"][m]["available"]]


def format_report(rep: dict) -> str:
    lines: list[str] = []
    lines.append("Audience Tracker — environment check")
    lines.append("=" * 40)
    py_mark = "OK" if rep["python_ok"] else "!!"
    lines.append(f"[{py_mark}] Python {rep['python_version']}  ({rep['platform']})")
    if not rep["python_ok"]:
        lines.append(f"     needs Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")

    cuda = rep["cuda"]
    if not cuda["torch_installed"]:
        lines.append("[--] PyTorch not installed")
    elif cuda["cuda_available"]:
        lines.append(f"[OK] CUDA GPU: {cuda['device_name']}  (torch {cuda['torch_version']})")
    else:
        lines.append(f"[!!] PyTorch {cuda['torch_version']} installed but CUDA NOT available "
                     "(CPU-only wheel? install the CUDA build — see docs)")

    lines.append("")
    lines.append("Dependencies:")
    for name, m in rep["modules"].items():
        mark = "OK" if m["available"] else "--"
        ver = f" {m['version']}" if m["version"] else ""
        lines.append(f"  [{mark}] {name}{ver}")

    lines.append("")
    lines.append("Capabilities:")
    for cap, ok in rep["capabilities"].items():
        mark = "READY" if ok else "missing"
        detail = "" if ok else f" ({', '.join(missing_modules(rep, cap))})"
        lines.append(f"  {cap:<7} {mark}{detail}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="audience-tracker doctor")
    parser.add_argument(
        "--require",
        choices=sorted(CAPABILITIES),
        help="Exit non-zero unless this capability is fully available.",
    )
    parser.add_argument(
        "--require-cuda", action="store_true", help="Also require a CUDA GPU to be available."
    )
    args = parser.parse_args(argv)

    rep = report()
    print(format_report(rep))

    ok = True
    if args.require and not capability_ok(rep, args.require):
        print(f"\nFAIL: capability '{args.require}' not satisfied: "
              f"{', '.join(missing_modules(rep, args.require))}")
        ok = False
    if args.require_cuda and not rep["cuda"]["cuda_available"]:
        print("\nFAIL: a CUDA GPU was required but is not available.")
        ok = False
    return 0 if ok else 1
