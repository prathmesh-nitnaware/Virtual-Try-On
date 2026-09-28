from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    vton_root = repo_root / "backend" / "cv" / "vton"
    target = vton_root / "debug_pipeline.py"
    spec = importlib.util.spec_from_file_location("vton_debug_pipeline", target)
    if spec is None or spec.loader is None:
        print(f"Could not load debug pipeline from {target}")
        return 1
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vton_debug_main = getattr(module, "main")
    vton_debug_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
