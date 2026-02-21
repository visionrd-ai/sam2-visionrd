# Unified Bidirectional SAM2 Pipeline - Folder Structure

## Overview
All output folders now follow a consistent, symmetrical naming scheme for `backward`, `forward`, and `combined` processing. When `--output-folder` is specified, outputs are organized by input folder name.

## Output Organization

### With `--output-folder` specified:
```
--output-folder /path/to/output
├── {input_folder_name}/
│   ├── output_backward/
│   ├── output_forward/
│   └── output_combined/
```

### Default (without `--output-folder`):
```
script_dir/
├── output_bidirectional/
│   └── {input_folder_name}/
│       ├── output_backward/
│       ├── output_forward/
│       └── output_combined/
```

---

## Complete Folder Structure (Each Direction)

### Each output folder (`output_backward/`, `output_forward/`, `output_combined/`) contains:

```
output_backward/              (or output_forward/ or output_combined/)
├── 4 main overlay videos:
│   ├── {video_name}_backward_masks_and_boxes.mp4         (both masks and bounding boxes)
│   ├── {video_name}_backward_boxes.mp4                    (boxes only)
│   ├── {video_name}_backward_masks_overlaid.mp4           (masks blended with frame)
│   └── {video_name}_backward_masks_only.mp4               (solid masks on black background)
│
├── masks/                    (Per-object mask videos - mask overlaid on frame)
│   ├── {video_name}_object_{label1}_masks.mp4     (mask overlaid with 60% transparency)
│   ├── {video_name}_object_{label2}_masks.mp4
│   └── ...
│
├── objects_cropped/          (Per-object cropped videos - bounding box region only)
│   ├── {video_name}_object_{label1}_backward_cropped.mp4
│   ├── {video_name}_object_{label2}_backward_cropped.mp4
│   └── ...
│
└── objects_isolated/         (Per-object uncropped videos - full frame, only object visible)
    ├── {video_name}_object_{label1}_backward_isolated.mp4
    ├── {video_name}_object_{label2}_backward_isolated.mp4
    └── ...
```

---

## Forward-Specific Naming

Forward videos use `_processed_` prefix instead of `_forward_` for consistency:

```
output_forward/
├── {video_name}_processed_masks_and_boxes.mp4
├── {video_name}_processed_boxes.mp4
├── {video_name}_processed_masks_overlaid.mp4
├── {video_name}_processed_masks_only.mp4
├── masks/
├── objects_cropped/
└── objects_isolated/
```

---

## Combined Output Naming

Combined videos use `_combined_` prefix:

```
output_combined/
├── {video_name}_combined_masks_and_boxes.mp4
├── {video_name}_combined_boxes.mp4
├── {video_name}_combined_masks_overlaid.mp4
├── {video_name}_combined_masks_only.mp4
├── masks/
├── objects_cropped/
└── objects_isolated/
```

---

## Example Command and Output Structure

### Command:
```bash
python process_folder_hira.py \
  --data-folder /home/multi-gpu/ai_res/Hira/sam2-visionrd/DATA \
  --output-folder /home/multi-gpu/ai_res/Hira/sam2-visionrd/output-bi \
  --script process_bidirectional_combined.py \
  --device cuda
```

### Output Structure:
```
/home/multi-gpu/ai_res/Hira/sam2-visionrd/output-bi/
├── video_folder_1/
│   ├── output_backward/
│   │   ├── (4 videos)
│   │   ├── masks/
│   │   ├── objects_cropped/
│   │   └── objects_isolated/
│   ├── output_forward/
│   │   ├── (4 videos)
│   │   ├── masks/
│   │   ├── objects_cropped/
│   │   └── objects_isolated/
│   └── output_combined/
│       ├── (4 videos)
│       ├── masks/
│       ├── objects_cropped/
│       └── objects_isolated/
├── video_folder_2/
│   ├── output_backward/
│   ├── output_forward/
│   └── output_combined/
└── ...
```

---

## Folder Contents Summary

### Main Videos (4 per direction)
- **masks_and_boxes**: Segmentation masks overlaid on frame + bounding boxes
- **boxes**: Bounding boxes only
- **masks_overlaid**: Masks blended with original frame (60% transparency)
- **masks_only**: Masks only (solid colors on black background)

### Masks Folder
- Per-object mask videos with masks overlaid on frames
- One video per object showing mask with 60% transparency blend
- Same length and frame rate as main videos
- Useful for reviewing segmentation accuracy per object

### objects_cropped Folder
- One video per object showing only the cropped region from bounding box
- Original frame content (not segmented)
- Resolution matches bounding box size

### objects_isolated Folder
- One video per object showing full frame
- Only object pixels visible (black background elsewhere)
- Resolution matches original video

---

## Key Features

✅ **Consistent Naming**: All directions follow same structure
✅ **Symmetrical Folders**: `masks`, `objects_cropped`, `objects_isolated` in each
✅ **Flexible Output**: Supports custom `--output-folder` with input folder organization
✅ **Complete Data**: 4 videos + 3 folders per direction per video
✅ **Total Output**: 3 directions × (4 videos + object videos per folder) per input video

---

## Usage Tips

### To run batch processing with custom output folder:
```bash
python process_folder_hira.py \
  --data-folder YOUR_DATA_FOLDER \
  --output-folder /custom/output/path \
  --script process_bidirectional_combined.py
```

### To check output after processing:
```bash
# List all processed folders
ls /custom/output/path/

# Check structure of first folder
tree /custom/output/path/$(ls /custom/output/path/ | head -1)/

# Count total videos
find /custom/output/path -name "*.mp4" | wc -l
```

### Edge Cases Handling
- **Annotated frame at position 1**: Only forward processing runs
- **Annotated frame at last position**: Only backward processing runs  
- **Annotated frame in middle**: Both directions run and combine
- **Output naming**: `_backward_only`, `_forward_only`, or `_combined` suffix automatically applied

---

## File Sizes (Approximate per folder)

- Each overlay video: 50-200 MB (depending on resolution and object complexity)
- Per-object cropped video: 5-50 MB
- Per-object isolated video: 20-100 MB
- Per-object mask video: 20-100 MB

**Total per direction**: ~400-600 MB for typical video

---

## Processing Pipeline Flow

```
Input Video + Annotation
        ↓
┌───────────────────────┐
│  BACKWARD PROCESSING  │
└───────────────────────┘
        ↓
   output_backward/
        ↓
┌───────────────────────┐
│  FORWARD PROCESSING   │
└───────────────────────┘
        ↓
   output_forward/
        ↓
┌───────────────────────┐
│   COMBINE OUTPUTS     │
│  (dedup annotated)    │
└───────────────────────┘
        ↓
   output_combined/
        ↓
    ✅ COMPLETE
```
