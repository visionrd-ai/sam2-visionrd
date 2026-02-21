# Quick Command Reference - Unified Bidirectional Pipeline

## Basic Commands

### Run with default output location:
```bash
python process_folder_hira.py \
  --data-folder /path/to/DATA \
  --script process_bidirectional_combined.py \
  --device cuda
```
Output: `script_dir/output_bidirectional/{input_folder_name}/...`

### Run with custom output folder:
```bash
python process_folder_hira.py \
  --data-folder /path/to/DATA \
  --output-folder /custom/output \
  --script process_bidirectional_combined.py \
  --device cuda
```
Output: `/custom/output/{input_folder_name}/...`

---

## Real-World Example

### Process DATA11 with custom output:
```bash
python process_folder_hira.py \
  --data-folder /home/multi-gpu/ai_res/Hira/sam2-visionrd/DATA11 \
  --output-folder /home/multi-gpu/ai_res/Hira/sam2-visionrd/output-bi \
  --script process_bidirectional_combined.py \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --fps 20 \
  --device cuda \
  --continue-on-error
```

---

## Advanced Options

### Skip preprocessing/frame extraction:
```bash
python process_folder_hira.py \
  --data-folder /path/to/DATA \
  --output-folder /output \
  --script process_bidirectional_combined.py \
  --skip-preprocessing \
  --skip-frame-extraction
```

### Filter specific folders:
```bash
python process_folder_hira.py \
  --data-folder /path/to/DATA \
  --output-folder /output \
  --script process_bidirectional_combined.py \
  --filter "video_" \
  --exclude "test"
```

### Dry run (preview only):
```bash
python process_folder_hira.py \
  --data-folder /path/to/DATA \
  --output-folder /output \
  --script process_bidirectional_combined.py \
  --dry-run
```

### Process only first N folders:
```bash
python process_folder_hira.py \
  --data-folder /path/to/DATA \
  --output-folder /output \
  --script process_bidirectional_combined.py \
  --max-folders 5
```

---

## Output Structure After Processing

```
/output/path/
├── video_1/
│   ├── output_backward/
│   │   ├── video_1_backward_masks_and_boxes.mp4
│   │   ├── video_1_backward_boxes.mp4
│   │   ├── video_1_backward_masks_overlaid.mp4
│   │   ├── video_1_backward_masks_only.mp4
│   │   ├── masks/
│   │   │   ├── object_label1/
│   │   │   │   └── frame_XXXXX_mask.png
│   │   │   └── object_label2/
│   │   │       └── frame_XXXXX_mask.png
│   │   ├── objects_cropped/
│   │   │   ├── video_1_object_label1_backward_cropped.mp4
│   │   │   └── video_1_object_label2_backward_cropped.mp4
│   │   └── objects_isolated/
│   │       ├── video_1_object_label1_backward_isolated.mp4
│   │       └── video_1_object_label2_backward_isolated.mp4
│   ├── output_forward/
│   │   ├── video_1_processed_*.mp4 (4 videos)
│   │   ├── masks/
│   │   ├── objects_cropped/
│   │   └── objects_isolated/
│   └── output_combined/
│       ├── video_1_combined_*.mp4 (4 videos)
│       ├── masks/
│       ├── objects_cropped/
│       └── objects_isolated/
├── video_2/
│   ├── output_backward/
│   ├── output_forward/
│   └── output_combined/
└── ...
```

---

## Checking Results

### List all folders:
```bash
ls /output/path/
```

### Check structure of first folder:
```bash
tree /output/path/$(ls -1 /output/path | head -1)/ -L 3
```

### Count total videos:
```bash
find /output/path -name "*.mp4" | wc -l
```

### Count videos per direction per folder:
```bash
find /output/path -name "*_backward_masks_and_boxes.mp4" | wc -l    # Backward
find /output/path -name "*_processed_masks_and_boxes.mp4" | wc -l   # Forward
find /output/path -name "*_combined_masks_and_boxes.mp4" | wc -l    # Combined
```

### Check total size:
```bash
du -sh /output/path/
```

### Check individual folder size:
```bash
du -sh /output/path/video_1/
```

---

## Expected Output (Per Video)

### Total Files:
- 4 backward overlay videos
- 4 forward overlay videos
- 4 combined overlay videos
- Per-object cropped videos (backward + forward + combined)
- Per-object isolated videos (backward + forward + combined)
- Per-object mask videos (backward + forward + combined) - masks overlaid on frame

### Approximate Size per Direction:
- 4 overlay videos: 200-400 MB
- Object videos: 100-300 MB
- Mask videos: 100-200 MB
- **Total per direction: 400-900 MB**
- **Total for all 3 directions: 1.2-2.7 GB per input video**

---

## Troubleshooting

### No output folder created?
Check that `--output-folder` path exists or is writable:
```bash
mkdir -p /output/path
chmod 755 /output/path
```

### Wrong folder structure?
Make sure both `--input-folder` and `--output-folder` are specified:
```bash
# ❌ Wrong - only uses default structure
python process_folder_hira.py --data-folder /data

# ✅ Right - uses custom output with input folder organization
python process_folder_hira.py --data-folder /data --output-folder /output
```

### Processing failed?
Use `--continue-on-error` to skip failed folders:
```bash
python process_folder_hira.py \
  --data-folder /data \
  --output-folder /output \
  --script process_bidirectional_combined.py \
  --continue-on-error
```

Check the batch summary JSON:
```bash
cat batch_processing_summary_*.json | python -m json.tool
```

---

## Performance Tips

1. **Use GPU**: `--device cuda` (much faster than CPU)
2. **Batch processing**: Process multiple folders at once
3. **Skip preprocessing** if video already preprocessed: `--skip-preprocessing`
4. **Reduce FPS** if storage is limited: `--fps 10`
5. **Filter folders** to test before full batch: `--filter "test" --max-folders 1`

---

## Advanced Filtering

### Process only new videos:
```bash
python process_folder_hira.py \
  --data-folder /data \
  --output-folder /output \
  --filter "2025"  # Only folders with 2025 in name
```

### Exclude test videos:
```bash
python process_folder_hira.py \
  --data-folder /data \
  --output-folder /output \
  --exclude "test"  # Skip folders with 'test' in name
```

### Combine filters:
```bash
python process_folder_hira.py \
  --data-folder /data \
  --output-folder /output \
  --filter "2025"         # Include only 2025
  --exclude "broken"      # But skip 'broken'
  --max-folders 10        # Process max 10
```

---

## Notes

- `annotated_frame_no`: The frame number in the original video where annotation exists
- `backward`: Processes frames from annotated frame backward to frame 1
- `forward`: Processes frames from annotated frame forward to last frame
- `combined`: Merges backward and forward, removing duplicate (annotated frame)
- `masks/`: New! Individual mask images per object per frame for post-processing
- `objects_cropped/`: Videos cropped to bounding box area only
- `objects_isolated/`: Videos with full frame but only object visible

---

## Example Workflow

```bash
# 1. Dry run to preview
python process_folder_hira.py \
  --data-folder /data \
  --output-folder /output \
  --dry-run

# 2. Process first 5 folders
python process_folder_hira.py \
  --data-folder /data \
  --output-folder /output \
  --script process_bidirectional_combined.py \
  --max-folders 5 \
  --continue-on-error \
  --device cuda

# 3. Check results
ls /output/
find /output -name "*.mp4" | wc -l

# 4. Process remaining
python process_folder_hira.py \
  --data-folder /data \
  --output-folder /output \
  --script process_bidirectional_combined.py \
  --device cuda
```
