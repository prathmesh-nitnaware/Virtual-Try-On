from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VtonPaths:
    root: Path
    checkpoints: Path
    outputs: Path
    samples: Path
    cpvton_repo: Path
    schp_repo: Path
    hrnet_repo: Path
    runtime_data: Path


def get_vton_paths() -> VtonPaths:
    root = Path(__file__).resolve().parents[1]
    return VtonPaths(
        root=root,
        checkpoints=root / "checkpoints",
        outputs=root / "outputs",
        samples=root / "samples",
        cpvton_repo=root / "cpvton" / "cp-vton-plus",
        schp_repo=root / "parsers" / "Self-Correction-Human-Parsing",
        hrnet_repo=root / "pose" / "deep-high-resolution-net.pytorch",
        runtime_data=root / "cpvton" / "runtime_data",
    )
