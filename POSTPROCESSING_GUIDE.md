# Post-Processing Pipeline for Isolated Objects

## Overview

The post-processing pipeline extracts tight bounding boxes from `objects_isolated` videos and creates high-quality cropped videos. This is useful for:

- **Detailed inspection** of object segmentation quality
- **High-resolution crops** of detected objects
- **Consistent framing** of objects for comparison
- **Storage optimization** with controlled sizing

## Features

### Automatic Bounding Box Detection
- Analyzes full video to find object bounding box
- Samples frames intelligently (not every frame)
- Uses percentile-based detection for robustness
- Accounts for object movement/size variations

### Smart Padding & Framing
- Adds configurable padding around objects (default 15%)
- Keeps objects centered in crop
- Respects frame boundaries
- Optional square bounding boxes

### Flexible Output Sizing
- Keep original crop size
- Resize to specific resolution (e.g., 512x512)
- Maintains aspect ratio through padding

### Batch Processing
- Processes all three directions (backward, forward, combined)
- Separate post-processed folder per direction
- Organized output structure

---

## Usage

### Option 1: Post-Process Existing Output

Process folders that have already been run through main processing:

```bash
python postprocess_objects.py \
  --output-folder /path/to/output \
  --target-size 512 512 \
  --square
```

### Option 2: Batch Processing with Automatic Post-Processing

Run main processing and post-processing in one command:

```bash
python batch_with_postprocessing.py \
  --data-folder /path/to/videos \
  --output-folder /path/to/output \
  --script process_bidirectional_combined.py \
  --device cuda \
  --target-size 512 512 \
  --square-bbox
```

### Option 3: Process Specific Subfolder

Post-process one specific folder:

```bash
python postprocess_objects.py \
  --output-folder /path/to/output \
  --folder-name video_001 \
  --target-size 512 512
```

---

## Arguments

### `postprocess_objects.py`

```
--output-folder PATH          Output folder containing output_backward/forward/combined
                             (REQUIRED)

--target-size W H            Resize cropped videos to W×H pixels
                             Default: Keep crop size

--square                     Make bounding boxes square (equal width/height)
                             Default: False

--padding-ratio RATIO        Padding as ratio of object size
                             Default: 0.15 (15%)

--folder-name NAME           Process only specific subfolder
                             Default: Process all subfolders
```

### `batch_with_postprocessing.py`

Main processing arguments:
```
--data-folder PATH           Folder with video subfolders (REQUIRED)
--output-folder PATH         Base output folder (optional)
--script FILE                Processing script (default: process_bidirectional_combined.py)
--device DEVICE              cuda/cpu/mps (default: cuda)
--fps FPS                    Output video FPS (default: 20)
--continue-on-error          Continue if individual videos fail
```

Post-processing arguments:
```
--skip-postprocessing        Skip post-processing step
--target-size W H            Resize cropped videos to W×H
--square-bbox                Make bounding boxes square
--padding-ratio RATIO        Padding as % of object size (default: 0.15)
```

---

## Output Structure

After post-processing, the folder structure becomes:

```
output_folder/
├── video_001/
│   ├── output_backward/
│   │   ├── overlay_backward.mp4
│   │   ├── objects_cropped/
│   │   │   ├── video_001_object_obj_1_cropped.mp4
│   │   │   └── ...
│   │   ├── objects_isolated/
│   │   │   ├── video_001_object_obj_1_isolated.mp4
│   │   │   └── ...
│   │   ├── objects_isolated_postprocessed/     ← NEW
│   │   │   ├── video_001_object_obj_1_isolated.mp4
│   │   │   └── ...
│   │   └── masks/
│   │
│   ├── output_forward/
│   │   ├── overlay_processed_forward.mp4
│   │   ├── objects_cropped/
│   │   ├── objects_isolated/
│   │   ├── objects_isolated_postprocessed/     ← NEW
│   │   └── masks/
│   │
│   └── output_combined/
│       ├── overlay_combined.mp4
│       ├── objects_cropped/
│       ├── objects_isolated/
│       ├── objects_isolated_postprocessed/     ← NEW
│       └── masks/
│
└── video_002/
    └── ...
```

**Note**: Post-processed videos have the same filenames but are placed in the `objects_isolated_postprocessed/` folder.

---

## Examples

### Example 1: Standard Post-Processing

After running main processing:

```bash
python postprocess_objects.py --output-folder /output/production
```

**Result**: Creates tight crops of each object at original resolution.

---

### Example 2: Uniform 512×512 Crops

For consistent model training input:

```bash
python postprocess_objects.py \
  --output-folder /output/training_data \
  --target-size 512 512 \
  --square
```

**Result**: All object videos are exactly 512×512 with objects centered.

---

### Example 3: Full Pipeline

Start to finish with post-processing:

```bash
python batch_with_postprocessing.py \
  --data-folder /data/videos \
  --output-folder /output/bidirectional \
  --script process_bidirectional_combined.py \
  --device cuda \
  --target-size 448 448 \
  --square-bbox \
  --padding-ratio 0.2 \
  --continue-on-error
```

**Timeline**:
1. Process all videos in `/data/videos`
2. Generate output in `/output/bidirectional/{video_name}/`
3. For each direction (backward/forward/combined):
   - Find tight bounding box of each object
   - Add 20% padding
   - Crop to square
   - Resize to 448×448
   - Save in `objects_isolated_postprocessed/`

---

## Technical Details

### Bounding Box Detection Algorithm

1. **Sampling**: Analyzes ~50 frames uniformly distributed throughout video
2. **Per-frame detection**: Finds contours of non-black pixels
3. **Aggregation**: Collects bounding rectangles from all sampled frames
4. **Robustness**: Uses 25th-75th percentile (IQR) for stable bbox

```
Frame 1: bbox = (10, 20, 150, 180)
Frame 50: bbox = (12, 22, 152, 182)
...
Frame 999: bbox = (8, 18, 148, 178)

Result: median bbox across all frames
```

### Padding Application

Adds padding as percentage of object size:
- 15% padding on 100px object = 15px on each side
- Respects frame boundaries (doesn't crop off frame)

### Resizing Strategy

If target size is specified:
1. Crop to bounding box + padding
2. Resize cropped region to target size
3. Write to output video at original fps

---

## Performance

### Speed
- Analysis phase: ~5-10 seconds per video (sampling only)
- Cropping & writing: ~2-5 seconds per 100 frames
- Overall: ~30-60 seconds per 5-minute video

### Storage
- Cropped videos typically 20-40% of original isolated video size
- 512×512 crops: ~10-20 MB per 100 frames
- Batch of 10 objects × 3 directions: ~300-600 MB

### Hardware
- GPU not required (CPU processing)
- Memory: < 500 MB per video
- Can be parallelized if needed

---

## Troubleshooting

### "No object found in video"

**Cause**: Video appears to be completely black or empty

**Solutions**:
- Check if objects_isolated folder is correctly generated
- Verify original video has proper segmentation
- Check for encoding issues

### Low quality / blurry crops

**Cause**: Target size is too large for object size

**Solutions**:
- Don't use `--target-size` if objects are small
- Use original crop size
- Increase `--padding-ratio` to capture more context

### Different crops for different objects

**Cause**: Normal - objects have different sizes

**Solutions**:
- Use `--square` to standardize framing
- Use `--target-size` with `--square` for uniform output
- Increase `--padding-ratio` for consistent context

### Out of memory errors

**Cause**: Frame size too large or buffer issues

**Solutions**:
- Reduce `--target-size`
- Process fewer videos at once
- Ensure system has sufficient RAM (typically not an issue)

---

## Integration with Training Pipelines

Use post-processed videos for:

### Object Detection Training
```bash
python postprocess_objects.py \
  --output-folder /data/isolated_objects \
  --target-size 640 640 \
  --square \
  --padding-ratio 0.1
```

### Fine-grained Classification
```bash
python postprocess_objects.py \
  --output-folder /data/classification \
  --target-size 512 512 \
  --padding-ratio 0.2
```

### Quality Control / Inspection
```bash
python postprocess_objects.py \
  --output-folder /data/inspect \
  --padding-ratio 0.3
```

---

## Command Reference

### Quick Start

Process all folders with 512×512 square crops:
```bash
cd /home/multi-gpu/ai_res/Hira/sam2-visionrd
python postprocess_objects.py \
  --output-folder output-bi \
  --target-size 512 512 \
  --square
```

### Integration into Existing Workflow

After normal batch processing:
```bash
# 1. Run main processing (as usual)
python process_folder_hira.py \
  --data-folder DATA11 \
  --script process_bidirectional_combined.py \
  --output-folder output-final

# 2. Post-process all output
python postprocess_objects.py \
  --output-folder output-final \
  --target-size 512 512
```

### One-Command Full Pipeline

```bash
python batch_with_postprocessing.py \
  --data-folder DATA11 \
  --output-folder output-final \
  --script process_bidirectional_combined.py \
  --device cuda \
  --target-size 512 512 \
  --square-bbox
```

---

## Related Documentation

- [UNIFIED_FOLDER_STRUCTURE.md](UNIFIED_FOLDER_STRUCTURE.md) - Output folder structure
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Main pipeline overview
- [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md) - All command examples
