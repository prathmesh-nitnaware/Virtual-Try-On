from __future__ import annotations

from pathlib import Path

try:
    from vton_pipeline import run_pipeline
    from utils.paths import get_vton_paths
except ModuleNotFoundError as exc:
    print("Pipeline Status: FAILED")
    print(f"Reason: Missing dependency: {exc}")
    raise SystemExit(1)


def main() -> None:
    paths = get_vton_paths()
    person = paths.samples / "person.jpg"
    garment = paths.samples / "garment.jpg"
    if not person.exists() or not garment.exists():
        print("Pipeline Status: FAILED")
        print(f"Reason: Missing samples at {person} and/or {garment}")
        return

    report = run_pipeline(person, garment, debug=True)
    print(f"Pipeline Status: {report.status}")
    if report.status != "SUCCESS":
        print(f"Reason: {report.reason}")
        return
    print("Debug outputs:")
    for k, v in report.outputs.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
