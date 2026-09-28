from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CpvtonResult:
    warped_cloth_path: Path
    result_path: Path


def _run(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def prepare_cpvton_runtime_data(
    runtime_root: Path,
    person_image: Path,
    garment_image: Path,
    garment_mask: Path,
    parse_mask: Path,
    person_mask: Path,
    pose_json: Path,
) -> None:
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    (runtime_root / "test" / "image").mkdir(parents=True, exist_ok=True)
    (runtime_root / "test" / "cloth").mkdir(parents=True, exist_ok=True)
    (runtime_root / "test" / "cloth-mask").mkdir(parents=True, exist_ok=True)
    (runtime_root / "test" / "image-parse-new").mkdir(parents=True, exist_ok=True)
    (runtime_root / "test" / "image-mask").mkdir(parents=True, exist_ok=True)
    (runtime_root / "test" / "pose").mkdir(parents=True, exist_ok=True)

    person_name = "person.jpg"
    garment_name = "garment.jpg"
    parse_name = "person.png"
    pose_name = "person_keypoints.json"

    shutil.copy2(person_image, runtime_root / "test" / "image" / person_name)
    shutil.copy2(garment_image, runtime_root / "test" / "cloth" / garment_name)
    shutil.copy2(garment_mask, runtime_root / "test" / "cloth-mask" / garment_name)
    shutil.copy2(parse_mask, runtime_root / "test" / "image-parse-new" / parse_name)
    shutil.copy2(person_mask, runtime_root / "test" / "image-mask" / parse_name)
    shutil.copy2(pose_json, runtime_root / "test" / "pose" / pose_name)
    (runtime_root / "test_pairs.txt").write_text(f"{person_name} {garment_name}\n", encoding="utf-8")


def run_cpvton_inference(
    cpvton_repo: Path,
    runtime_data: Path,
    output_dir: Path,
    gmm_ckpt: Path,
    tom_ckpt: Path,
) -> CpvtonResult:
    result_root = cpvton_repo / "result" / "PIPELINE" / "test"

    _run(
        [
            sys.executable,
            "test.py",
            "--name",
            "PIPELINE",
            "--stage",
            "GMM",
            "--workers",
            "1",
            "--batch-size",
            "1",
            "--dataroot",
            str(runtime_data),
            "--datamode",
            "test",
            "--data_list",
            "test_pairs.txt",
            "--checkpoint",
            str(gmm_ckpt),
        ],
        cpvton_repo,
    )

    (runtime_data / "test" / "warp-cloth").mkdir(parents=True, exist_ok=True)
    (runtime_data / "test" / "warp-mask").mkdir(parents=True, exist_ok=True)
    shutil.copytree(result_root / "warp-cloth", runtime_data / "test" / "warp-cloth", dirs_exist_ok=True)
    shutil.copytree(result_root / "warp-mask", runtime_data / "test" / "warp-mask", dirs_exist_ok=True)

    _run(
        [
            sys.executable,
            "test.py",
            "--name",
            "PIPELINE",
            "--stage",
            "TOM",
            "--workers",
            "1",
            "--batch-size",
            "1",
            "--dataroot",
            str(runtime_data),
            "--datamode",
            "test",
            "--data_list",
            "test_pairs.txt",
            "--checkpoint",
            str(tom_ckpt),
        ],
        cpvton_repo,
    )

    warped_src = result_root / "warp-cloth" / "person.jpg"
    result_src = result_root / "try-on" / "person.jpg"
    if not warped_src.exists() or not result_src.exists():
        raise FileNotFoundError("CP-VTON+ did not produce expected output files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    warped_dst = output_dir / "warped_cloth.png"
    result_dst = output_dir / "result.png"
    shutil.copy2(warped_src, warped_dst)
    shutil.copy2(result_src, result_dst)
    return CpvtonResult(warped_cloth_path=warped_dst, result_path=result_dst)
