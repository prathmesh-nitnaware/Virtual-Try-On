from __future__ import annotations

import platform
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    vram_gb: float | None
    backend: str


def detect_best_device() -> DeviceInfo:
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024**3)
        return DeviceInfo(
            device=torch.device("cuda"),
            name=props.name,
            vram_gb=round(vram_gb, 2),
            backend="cuda",
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            device=torch.device("mps"),
            name=f"Apple Silicon ({platform.machine()})",
            vram_gb=None,
            backend="mps",
        )

    return DeviceInfo(
        device=torch.device("cpu"),
        name=platform.processor() or "CPU",
        vram_gb=None,
        backend="cpu",
    )
