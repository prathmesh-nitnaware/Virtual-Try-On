from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from parsers.schp_parser import run_schp_parsing
from pose.hrnet_pose import run_hrnet_pose
from cpvton.cpvton_runner import prepare_cpvton_runtime_data, run_cpvton_inference
from utils.checkpoint_manager import validate_checkpoints
from utils.device import detect_best_device
from utils.garment_preprocess import preprocess_garment
from utils.paths import get_vton_paths


@dataclass
class PipelineReport:
    status: str
    reason: str
    runtime_seconds: float
    device: str
    models_loaded: list[str]
    outputs: dict[str, str]
    parsing_stats: dict[int, int]
    pose_detected_joints: int
    garment_alpha_coverage: float


def run_pipeline(person_image: Path, garment_image: Path, debug: bool = True) -> PipelineReport:
    t0 = time.time()
    paths = get_vton_paths()
    paths.outputs.mkdir(parents=True, exist_ok=True)

    ok, ckpt_lines, ckpts = validate_checkpoints(paths)
    for line in ckpt_lines:
        print(line)
    if not ok:
        return PipelineReport(
            status="FAILED",
            reason="Missing one or more checkpoints.",
            runtime_seconds=time.time() - t0,
            device="N/A",
            models_loaded=[],
            outputs={},
            parsing_stats={},
            pose_detected_joints=0,
            garment_alpha_coverage=0.0,
        )

    device_info = detect_best_device()
    print(f"Using device: {device_info.backend} | {device_info.name} | VRAM: {device_info.vram_gb} GB")

    parsing = run_schp_parsing(
        repo_dir=paths.schp_repo,
        checkpoint_path=ckpts["SCHP"],
        person_image=person_image,
        output_dir=paths.outputs,
    )
    print(f"Parsing labels: {parsing.label_stats}")

    hrnet_cfg = paths.hrnet_repo / "experiments" / "coco" / "hrnet" / "w32_256x192_adam_lr1e-3.yaml"
    pose = run_hrnet_pose(
        hrnet_repo=paths.hrnet_repo,
        config_path=hrnet_cfg,
        checkpoint_path=ckpts["HRNet"],
        person_image=person_image,
        output_dir=paths.outputs,
        device=device_info.device,
    )
    print(f"Detected joints: {pose.num_detected_joints}/17")

    garment = preprocess_garment(garment_image=garment_image, output_dir=paths.outputs)
    print(f"Garment coverage: {garment.alpha_coverage:.4f}")

    prepare_cpvton_runtime_data(
        runtime_root=paths.runtime_data,
        person_image=person_image,
        garment_image=garment.processed_path,
        garment_mask=garment.mask_path,
        parse_mask=parsing.cp_mask_path,
        person_mask=parsing.image_mask_path,
        pose_json=pose.keypoints_json_path,
    )
    cpvton = run_cpvton_inference(
        cpvton_repo=paths.cpvton_repo,
        runtime_data=paths.runtime_data,
        output_dir=paths.outputs,
        gmm_ckpt=ckpts["GMM"],
        tom_ckpt=ckpts["TOM"],
    )

    report = PipelineReport(
        status="SUCCESS",
        reason="Pipeline completed.",
        runtime_seconds=time.time() - t0,
        device=device_info.backend,
        models_loaded=["SCHP", "HRNet", "GMM", "TOM"],
        outputs={
            "parsing_mask": str(parsing.cp_mask_path),
            "pose_overlay": str(pose.overlay_path),
            "keypoints_json": str(pose.keypoints_json_path),
            "processed_garment": str(garment.processed_path),
            "warped_cloth": str(cpvton.warped_cloth_path),
            "result": str(cpvton.result_path),
        },
        parsing_stats=parsing.label_stats,
        pose_detected_joints=pose.num_detected_joints,
        garment_alpha_coverage=garment.alpha_coverage,
    )
    with open(paths.outputs / "pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    return report
