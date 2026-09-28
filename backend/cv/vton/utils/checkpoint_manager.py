from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import VtonPaths


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path


def expected_checkpoints(paths: VtonPaths) -> list[CheckpointSpec]:
    return [
        CheckpointSpec("SCHP", paths.checkpoints / "human_parsing" / "exp-schp-201908301523-atr.pth"),
        CheckpointSpec("HRNet", paths.checkpoints / "pose" / "pose_hrnet_w32_256x192.pth"),
        CheckpointSpec("GMM", paths.checkpoints / "vton" / "gmm_final.pth"),
        CheckpointSpec("TOM", paths.checkpoints / "vton" / "tom_final.pth"),
    ]


def validate_checkpoints(paths: VtonPaths) -> tuple[bool, list[str], dict[str, Path]]:
    status_lines: list[str] = []
    resolved: dict[str, Path] = {}
    ok = True
    for spec in expected_checkpoints(paths):
        if spec.path.exists():
            status_lines.append(f"✓ {spec.name} Loaded ({spec.path})")
            resolved[spec.name] = spec.path
        else:
            status_lines.append(f"✗ Missing {spec.name} Checkpoint ({spec.path})")
            ok = False
    return ok, status_lines, resolved
