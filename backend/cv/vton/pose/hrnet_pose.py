from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


@dataclass(frozen=True)
class HrnetPoseResult:
    keypoints_json_path: Path
    overlay_path: Path
    num_detected_joints: int


def _load_hrnet_modules(hrnet_repo: Path):
    sys.path.insert(0, str(hrnet_repo / "lib"))
    from config import cfg, update_config  # type: ignore
    import models  # type: ignore
    from core.inference import get_final_preds  # type: ignore
    from utils.transforms import get_affine_transform  # type: ignore

    return cfg, update_config, models, get_final_preds, get_affine_transform


def run_hrnet_pose(
    hrnet_repo: Path,
    config_path: Path,
    checkpoint_path: Path,
    person_image: Path,
    output_dir: Path,
    device: torch.device,
) -> HrnetPoseResult:
    cfg, update_config, models, get_final_preds, get_affine_transform = _load_hrnet_modules(hrnet_repo)

    class Args:
        cfg = str(config_path)
        opts = []
        modelDir = ""
        logDir = ""
        dataDir = ""
        prevModelDir = ""

    update_config(cfg, Args)
    cfg.defrost()
    cfg.TEST.MODEL_FILE = str(checkpoint_path)
    cfg.freeze()

    pose_model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    pose_model.load_state_dict(state_dict, strict=False)
    pose_model.to(device)
    pose_model.eval()

    image_bgr = cv2.imread(str(person_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read person image: {person_image}")
    h, w = image_bgr.shape[:2]

    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    aspect_ratio = cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.IMAGE_SIZE[1]
    box_w = float(w)
    box_h = float(h)
    if box_w > aspect_ratio * box_h:
        box_h = box_w / aspect_ratio
    else:
        box_w = box_h * aspect_ratio
    scale = np.array([box_w / 200.0, box_h / 200.0], dtype=np.float32) * 1.25

    trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image_bgr,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR,
    )
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = tfm(Image.fromarray(cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)

    with torch.no_grad():
        output = pose_model(input_tensor)
    coords, maxvals = get_final_preds(cfg, output.cpu().numpy(), np.asarray([center]), np.asarray([scale]))
    joints = coords[0]
    confs = maxvals[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = image_bgr.copy()
    detected = 0
    kps: list[float] = []
    for (x, y), conf in zip(joints, confs):
        c = float(conf[0])
        kps.extend([float(x), float(y), c])
        if c > 0.05:
            detected += 1
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 0, 255), -1)

    overlay_path = output_dir / "pose_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    json_path = output_dir / "keypoints.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1.0, "people": [{"pose_keypoints": kps}]}, f, indent=2)

    return HrnetPoseResult(
        keypoints_json_path=json_path,
        overlay_path=overlay_path,
        num_detected_joints=detected,
    )
