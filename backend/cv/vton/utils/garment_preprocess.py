from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from rembg import remove
except Exception:  # pragma: no cover
    remove = None


@dataclass(frozen=True)
class GarmentPreprocessResult:
    processed_path: Path
    mask_path: Path
    width: int
    height: int
    alpha_coverage: float


def preprocess_garment(garment_image: Path, output_dir: Path, target_size: tuple[int, int] = (192, 256)) -> GarmentPreprocessResult:
    if remove is None:
        raise RuntimeError("rembg is not installed; garment preprocessing cannot run.")

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = garment_image.read_bytes()
    rgba_bytes = remove(raw_bytes)
    rgba = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")

    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Garment preprocessing produced empty alpha mask.")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cropped = rgba.crop((x0, y0, x1 + 1, y1 + 1))

    tw, th = target_size
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    c_w, c_h = cropped.size
    scale = min(tw / c_w, th / c_h)
    new_w = max(1, int(c_w * scale))
    new_h = max(1, int(c_h * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)
    ox = (tw - new_w) // 2
    oy = (th - new_h) // 2
    canvas.alpha_composite(resized, (ox, oy))

    alpha_mask = np.array(canvas.split()[-1], dtype=np.uint8)
    binary_mask = (alpha_mask >= 16).astype(np.uint8) * 255
    coverage = float((binary_mask > 0).sum()) / float(binary_mask.size)

    processed_path = output_dir / "processed_garment.png"
    mask_path = output_dir / "processed_garment_mask.png"
    canvas.save(processed_path)
    Image.fromarray(binary_mask).save(mask_path)

    return GarmentPreprocessResult(
        processed_path=processed_path,
        mask_path=mask_path,
        width=tw,
        height=th,
        alpha_coverage=coverage,
    )
