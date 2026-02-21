# Changes Made - Unified Folder Structure & Symmetric Outputs

## Summary
Updated all three main processing scripts to create consistent, symmetrical folder structures across backward, forward, and combined processing. When `--output-folder` is specified with `--input-folder`, outputs are organized as `output_folder/{input_folder_name}/...`

---

## Files Modified

### 1. `process_bidirectional_combined.py`
**Changes:**
- ✅ Updated `main()` to handle `--output-folder` correctly
- ✅ When `--output-folder` specified: uses `output_folder/{input_folder_name}/`
- ✅ Default: uses `output_bidirectional/{input_folder_name}/`
- ✅ Creates three subdirectories: `output_backward/`, `output_forward/`, `output_combined/`

**Key Logic:**
```python
if args.output_folder:
    output_folder_parent = os.path.abspath(args.output_folder)
    output_folder_main = os.path.join(output_folder_parent, input_folder_name)
else:
    output_folder_main = os.path.join(script_dir, "output_bidirectional", input_folder_name)
```

---

### 2. `process_backward_only.py`
**Changes:**
- ✅ Updated `main()` to handle output folder structure
- ✅ Added `masks/` folder creation with per-object subdirectories
- ✅ Updated `create_backward_videos()` to save individual mask PNG files
- ✅ Maintains `objects_cropped/` and `objects_isolated/` folders

**New Folders Created:**
- `masks/object_{label}/frame_XXXXX_mask.png` - Individual binary mask images
- `objects_cropped/` - Cropped videos (bounded box region only)
- `objects_isolated/` - Isolated videos (full frame, object only)

**Video Naming:**
- `{video_name}_backward_masks_and_boxes.mp4`
- `{video_name}_backward_boxes.mp4`
- `{video_name}_backward_masks_overlaid.mp4`
- `{video_name}_backward_masks_only.mp4`

---

### 3. `process_forward_only.py`
**Changes:**
- ✅ Updated `main()` to handle output folder structure  
- ✅ Default path changed from `output/` to `output_forward/`
- ✅ Updated `propagate_and_create_videos()` to create unified folder structure
- ✅ Added `masks/` folder with per-frame mask images
- ✅ Renamed `objects/` to `objects_isolated/` for consistency
- ✅ Renamed videos from `_processed_` prefix (kept for forward to distinguish)

**New Folders Created:**
- `masks/object_{label}/frame_XXXXX_mask.png` - Individual binary mask images
- `objects_isolated/` - Uncropped isolated videos (full frame)
- `objects_cropped/` - Cropped videos (bounded box region only)

**Video Naming:**
- `{video_name}_processed_masks_and_boxes.mp4`
- `{video_name}_processed_boxes.mp4`
- `{video_name}_processed_masks_overlaid.mp4`
- `{video_name}_processed_masks_only.mp4`

---

## Folder Structure Unification

### Before:
```
Backward:  output_backward/
           ├── {video}_backward_*.mp4
           ├── objects_cropped/
           └── objects_isolated/

Forward:   output/
           ├── {video}_*.mp4
           ├── objects/
           ├── objects_cropped/
           └── masks_only/

Combined:  output_bidirectional/
           ├── {video}_combined_*.mp4
           ├── objects_cropped/
           └── objects_isolated/
```

### After (UNIFIED):
```
Backward:  output_backward/
           ├── 4 videos ({video}_backward_*.mp4)
           ├── masks/                    ✨ NEW
           ├── objects_cropped/          ✨ Consistent
           └── objects_isolated/         ✨ Consistent

Forward:   output_forward/
           ├── 4 videos ({video}_processed_*.mp4)
           ├── masks/                    ✨ NEW
           ├── objects_cropped/          ✨ Consistent
           └── objects_isolated/         ✨ Consistent

Combined:  output_combined/
           ├── 4 videos ({video}_combined_*.mp4)
           ├── masks/                    ✨ NEW
           ├── objects_cropped/          ✨ Consistent
           └── objects_isolated/         ✨ Consistent
```

---

## Output Folder Organization

### Command:
```bash
python process_folder_hira.py \
  --data-folder /data \
  --output-folder /output/path \
  --script process_bidirectional_combined.py
```

### Result:
```
/output/path/
├── video_folder_1/
│   ├── output_backward/    (4 videos + 3 folders)
│   ├── output_forward/     (4 videos + 3 folders)
│   └── output_combined/    (4 videos + 3 folders)
├── video_folder_2/
│   ├── output_backward/    (4 videos + 3 folders)
│   ├── output_forward/     (4 videos + 3 folders)
│   └── output_combined/    (4 videos + 3 folders)
└── ...
```

---

## Masks Folder Details

Each direction now saves individual mask images:

```
masks/
├── object_label1/
│   ├── frame_00000_mask.png    (Binary: 0=background, 255=object)
│   ├── frame_00001_mask.png
│   └── ...
├── object_label2/
│   ├── frame_00000_mask.png
│   ├── frame_00001_mask.png
│   └── ...
└── ...
```

**Benefits:**
- ✅ Frame-by-frame mask inspection
- ✅ Mask post-processing if needed
- ✅ Training data generation
- ✅ Debugging and analysis

---

## Video Output Summary

### Per Direction (4 videos + per-object variants):
1. **masks_and_boxes.mp4** - Segmentation + bounding boxes
2. **boxes.mp4** - Bounding boxes only (no masks)
3. **masks_overlaid.mp4** - Masks blended (60% opacity)
4. **masks_only.mp4** - Masks only (solid colors)

Plus per-object videos:
- **objects_cropped/** - N videos (cropped to bounding box)
- **objects_isolated/** - N videos (full frame, object only)
- **masks/** - N object folders with individual mask PNGs

---

## Backward Compatibility

✅ All existing parameters still work
✅ Old scripts can be called as-is
✅ Only affects folder structure and naming
✅ Output folder argument is optional (defaults to script_dir)

---

## Testing Instructions

### Test with custom output folder:
```bash
python process_folder_hira.py \
  --data-folder /home/multi-gpu/ai_res/Hira/sam2-visionrd/DATA11 \
  --output-folder /home/multi-gpu/ai_res/Hira/sam2-visionrd/output-bi-new \
  --script process_bidirectional_combined.py \
  --device cuda
```

### Verify structure:
```bash
# Check folder created with input folder name
ls /home/multi-gpu/ai_res/Hira/sam2-visionrd/output-bi-new/

# Check subfolder structure
tree /home/multi-gpu/ai_res/Hira/sam2-visionrd/output-bi-new/$(ls -1 /home/multi-gpu/ai_res/Hira/sam2-visionrd/output-bi-new/ | head -1)/

# Count total videos
find /home/multi-gpu/ai_res/Hira/sam2-visionrd/output-bi-new -name "*.mp4" | wc -l
```

---

## Key Benefits

🎯 **Consistent Structure**: Same folder layout for all directions
🎯 **Symmetrical Naming**: Easy to understand and automate
🎯 **Organized Output**: Input folder names as top-level organization
🎯 **Complete Masks**: Individual mask PNGs for each frame per object
🎯 **Flexible Output**: Custom output location with `--output-folder`
🎯 **Backward Compatible**: Old commands still work

---

## Related Documentation

See `UNIFIED_FOLDER_STRUCTURE.md` for complete folder structure details and examples.
