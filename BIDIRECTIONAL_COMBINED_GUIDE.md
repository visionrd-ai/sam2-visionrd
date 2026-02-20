# SAM2 Bidirectional Combined Video Segmentation Pipeline

## Overview
Complete pipeline that processes videos **both backward and forward** from an annotated frame, then combines them into a single coherent output with **zero frame duplication**.

## Architecture

```
Input Video (60 frames, annotated at frame 31)
        ↓
   ┌────┴────┐
   ↓         ↓
BACKWARD  FORWARD
(frames 1-31)  (frames 31-60)
   ↓         ↓
   └────┬────┘
        ↓
   COMBINED
   (60 frames total - annotated frame appears only once!)
```

## Key Features

✅ **Bidirectional Processing**
- Backward: Frame 1 → Annotated frame (reversed for SAM2)
- Forward: Annotated frame → Last frame

✅ **No Frame Duplication**
- Backward: 31 frames (includes annotated)
- Forward: 30 frames (starts from frame after annotated)
- Combined: 60 frames total (annotated appears once)

✅ **9 Output Videos per Direction**
- 4 main overlay videos:
  - Masks + Boxes
  - Boxes only
  - Masks overlaid on frames
  - Masks only (solid background)
- 5 per-object cropped videos (no mask overlay)

✅ **Smart Combining**
- Backward processed in reversed order (re-reversed for chronological)
- Forward processed in forward order
- Combined in correct chronological sequence

## Usage

### Basic Usage
```bash
python process_bidirectional_combined.py \
  --input-folder /path/to/video/and/json \
  --device cuda
```

### With Custom Output Location
```bash
python process_bidirectional_combined.py \
  --input-folder /path/to/video/and/json \
  --output-folder /custom/output/path \
  --device cuda
```

### Skip Processing
```bash
# Only backward
python process_bidirectional_combined.py \
  --input-folder /path \
  --skip-forward

# Only forward
python process_bidirectional_combined.py \
  --input-folder /path \
  --skip-backward
```

### Parameters
- `--input-folder` (required): Folder with video + JSON annotation
- `--output-folder`: Where to save combined videos (default: `output_bidirectional/`)
- `--processed-folder`: Intermediate processing folder (default: `processed_bidirectional/`)
- `--model-config`: SAM2 config (default: `configs/sam2.1/sam2.1_hiera_t.yaml`)
- `--checkpoint`: Model checkpoint (default: `checkpoints/sam2.1_hiera_tiny.pt`)
- `--fps`: Target FPS (default: 20)
- `--device`: `cuda`, `cpu`, or `mps` (default: `cuda`)
- `--skip-backward`: Skip backward processing
- `--skip-forward`: Skip forward processing

## Output Structure

```
output_bidirectional/{folder_name}/
├── output_backward/          (Backward pipeline output)
│   ├── *_backward_masks_and_boxes.mp4
│   ├── *_backward_boxes.mp4
│   ├── *_backward_masks_overlaid.mp4
│   ├── *_backward_masks_only.mp4
│   └── objects_cropped/
│       ├── *_object_6_backward_cropped.mp4
│       ├── *_object_7_backward_cropped.mp4
│       └── ...
│
├── output_forward/           (Forward pipeline output)
│   ├── *_processed_masks_and_boxes.mp4
│   ├── *_processed_boxes.mp4
│   ├── *_processed_masks_overlaid.mp4
│   ├── *_processed_masks_only.mp4
│   ├── objects/
│   │   └── *_object_*_isolated.mp4
│   └── masks_only/
│       └── *_mask_*.mp4
│
└── output_combined/          ✨ FINAL COMBINED VIDEOS ✨
    ├── *_combined_masks_and_boxes.mp4    (60 frames)
    ├── *_combined_boxes.mp4               (60 frames)
    ├── *_combined_masks_overlaid.mp4     (60 frames)
    ├── *_combined_masks_only.mp4         (60 frames)
    └── objects_cropped/
        ├── combined_object_6_cropped.mp4 (60 frames)
        ├── combined_object_7_cropped.mp4 (60 frames)
        ├── combined_object_8_cropped.mp4 (60 frames)
        ├── combined_object_9_cropped.mp4 (60 frames)
        └── combined_object_10_cropped.mp4 (60 frames)
```

## Frame Counting Logic

**Example: 60-frame video with annotated frame at frame 31**

| Direction | Frame Range | Count | Reason |
|-----------|------------|-------|--------|
| Backward  | 1-31       | 31    | Includes annotated frame |
| Forward   | 31-60      | 30    | Excludes annotated frame to avoid duplication |
| **Combined** | **1-60**   | **60**    | **Perfect match to original!** |

## Technical Details

### Backward Processing
1. Extract frames 0-30 from video
2. Reverse them (frame 30 → frame 0)
3. Create reversed video file
4. Run SAM2 propagation on reversed video
5. Write videos in reversed order
6. **Reverse final MP4 files** to chronological order

### Forward Processing
1. Extract frames 31-59 from video (frame 31 is the annotated frame)
2. Run SAM2 propagation forward
3. Write videos in forward order (chronological)

### Combining
1. Load all frames from backward video (31 total)
2. Skip last frame (this is the annotated frame, which will come from forward)
3. Load all frames from forward video (30 total)
4. Concatenate: [backward_frames[:-1]] + [forward_frames]
5. Result: 60 frames total with annotated frame appearing exactly once

## Per-Object Cropped Videos

Each per-object cropped video:
- ✅ Shows only the bounding box region
- ❌ Does NOT include mask overlay
- ✅ Covers entire 60-frame duration
- ✅ Maintains original FPS

## Performance

Typical execution times (RTX 3060, 60-frame video):
- Backward processing: ~15-20 seconds
- Forward processing: ~12-15 seconds
- Video combining: ~5-10 seconds
- **Total: ~30-45 seconds**

## Troubleshooting

### Error: Cannot find checkpoint
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/sam2.1_hiera_tiny.pt'
```
**Solution:** Make sure you run the script from the `sam2-visionrd/` directory, or provide absolute path to checkpoint.

### Error: No annotated frames found
**Solution:** Make sure your JSON file has at least one frame with bounding boxes in the `annotation.boxes` field.

### Missing forward output videos
**Solution:** The forward pipeline uses `processed_` prefix in filenames, not `forward_`. This is handled automatically by the combining logic.

### All zeros in per-object cropped videos
**Possible causes:**
- Mask didn't propagate properly (SAM2 failed to track)
- Check if backward/forward videos have annotations
- Try different SAM2 model size if too many failures

## Example Workflow

```bash
# 1. Navigate to sam2-visionrd directory
cd /home/multi-gpu/ai_res/Hira/sam2-visionrd

# 2. Run bidirectional pipeline
python process_bidirectional_combined.py \
  --input-folder DATA/test/test_video/ \
  --device cuda

# 3. Check outputs
ls -lh output_bidirectional/test_video/output_combined/

# 4. Verify frame counts
python -c "
import cv2
cap = cv2.VideoCapture('output_bidirectional/test_video/output_combined/*combined_masks_and_boxes.mp4')
print(f'Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')
"

# 5. Play video (if you have a player)
# vlc output_bidirectional/test_video/output_combined/*combined_masks_and_boxes.mp4
```

## Requirements

- Python 3.10+
- PyTorch 2.5.1+ with CUDA support
- OpenCV (cv2)
- NumPy
- Matplotlib
- Hydra
- SAM2 (included in sam2-visionrd/)

## References

- SAM2: https://github.com/facebookresearch/sam2
- Original implementation: `process_backward_only.py` and `process_forward_only.py`
