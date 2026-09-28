from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SchpResult:
    raw_mask_path: Path
    cp_mask_path: Path
    image_mask_path: Path
    label_stats: dict[int, int]


def _map_atr_to_cpvton(atr_mask: np.ndarray) -> np.ndarray:
    cp = np.zeros_like(atr_mask, dtype=np.uint8)
    cp[atr_mask == 1] = 1
    cp[atr_mask == 2] = 2
    cp[atr_mask == 4] = 5
    cp[atr_mask == 7] = 6
    cp[atr_mask == 5] = 12
    cp[atr_mask == 6] = 9
    cp[atr_mask == 11] = 13
    cp[atr_mask == 14] = 14
    cp[atr_mask == 15] = 15
    cp[atr_mask == 12] = 16
    cp[atr_mask == 13] = 17
    return cp


def run_schp_parsing(
    repo_dir: Path,
    checkpoint_path: Path,
    person_image: Path,
    output_dir: Path,
) -> SchpResult:
    input_dir = output_dir / "schp_input"
    raw_dir = output_dir / "schp_raw"
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(person_image, input_dir / person_image.name)

    cmd = [
        sys.executable,
        "simple_extractor.py",
        "--dataset",
        "atr",
        "--model-restore",
        str(checkpoint_path),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(raw_dir),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SCHP parsing failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    raw_mask_path = raw_dir / f"{person_image.stem}.png"
    if not raw_mask_path.exists():
        raise FileNotFoundError(f"SCHP did not produce mask: {raw_mask_path}")

    atr = np.array(Image.open(raw_mask_path).convert("P"), dtype=np.uint8)
    cp_mask = _map_atr_to_cpvton(atr)
    binary_mask = (cp_mask > 0).astype(np.uint8) * 255

    cp_mask_path = output_dir / "parsing_mask.png"
    image_mask_path = output_dir / "image_mask.png"
    Image.fromarray(cp_mask).save(cp_mask_path)
    Image.fromarray(binary_mask).save(image_mask_path)

    labels, counts = np.unique(cp_mask, return_counts=True)
    stats = {int(k): int(v) for k, v in zip(labels, counts)}

    return SchpResult(
        raw_mask_path=raw_mask_path,
        cp_mask_path=cp_mask_path,
        image_mask_path=image_mask_path,
        label_stats=stats,
    )
