# VTryon Phase 4A (Isolated Local AI Pipeline)

This phase adds an isolated local inference pipeline under `backend/cv/vton` for:

1. Human parsing (SCHP ATR)
2. Pose estimation (HRNet COCO)
3. Garment preprocessing
4. CP-VTON+ (GMM + TOM)

It does **not** modify the existing frontend AR demo or existing FastAPI routes.

## Isolation Choice

Phase 4A uses **Option A: separate Python virtual environment**:

- Minimal risk to current frontend/backend stack.
- No change required in existing `docker-compose.yml`.
- Easy local experimentation without breaking the current project runtime.

## Directory Structure

```text
backend/cv/vton/
├── parsers/
├── pose/
├── cpvton/
├── checkpoints/
├── samples/
├── outputs/
├── tests/
├── utils/
└── scripts/
```

## Setup

From repository root:

```powershell
powershell -ExecutionPolicy Bypass -File "backend/cv/vton/scripts/setup_vton_env.ps1"
```

## Checkpoint Setup

Place checkpoints at:

```text
backend/cv/vton/checkpoints/human_parsing/exp-schp-201908301523-atr.pth
backend/cv/vton/checkpoints/pose/pose_hrnet_w32_256x192.pth
backend/cv/vton/checkpoints/vton/gmm_final.pth
backend/cv/vton/checkpoints/vton/tom_final.pth
```

If any checkpoint is missing, the pipeline reports a hard failure.

## Samples

Place input images:

```text
backend/cv/vton/samples/person.jpg
backend/cv/vton/samples/garment.jpg
```

## Run

```powershell
python test_pipeline.py
```

Debug mode:

```powershell
python debug_pipeline.py
```

## Output Artifacts

Generated under `backend/cv/vton/outputs/`:

- `parsing_mask.png`
- `image_mask.png`
- `pose_overlay.png`
- `keypoints.json`
- `processed_garment.png`
- `processed_garment_mask.png`
- `warped_cloth.png`
- `result.png`
- `pipeline_report.json`

## Troubleshooting

- **Missing checkpoint**: verify file names and paths exactly.
- **SCHP failure**: verify parser checkpoint and that `simple_extractor.py` can execute.
- **HRNet failure**: verify config path and checkpoint compatibility.
- **CP-VTON+ failure**: verify GMM/TOM checkpoint files and generated runtime data under `backend/cv/vton/cpvton/runtime_data`.

## Known Limitations (Phase 4A)

- Focused on correctness, not speed.
- Pipeline depends on external pretrained checkpoints.
- Output quality depends strongly on checkpoint/data domain alignment.
