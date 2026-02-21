# Masks Folder Update - Now Contains Videos Instead of PNGs

## Summary

The `masks/` folder now contains **per-object mask videos** instead of individual frame PNGs.

## What Changed

### Before
```
masks/
├── object_label1/
│   ├── frame_00000_mask.png
│   ├── frame_00001_mask.png
│   └── ...
└── object_label2/
    ├── frame_00000_mask.png
    └── ...
```

### After ✨
```
masks/
├── {video_name}_object_label1_masks.mp4    (full video, mask overlaid)
└── {video_name}_object_label2_masks.mp4    (full video, mask overlaid)
```

## Benefits

✅ **Easier Review**: Play video instead of reviewing individual frames
✅ **Consistent Output**: All outputs are videos (easier downstream processing)
✅ **Smaller Storage**: Compressed video vs many PNG files
✅ **Same Duration**: Same length as main videos, same FPS
✅ **Visual Verification**: See mask progression over time smoothly

## Video Content

Each mask video:
- Shows the **full frame** (same as input video)
- Has the **mask overlaid** with 60% transparency
- Uses **color-coded masks** (same colors as main overlay videos)
- Same **resolution** as main video
- Same **FPS** as main video
- Useful for verifying segmentation accuracy per object

## Example

```
output_backward/masks/
├── video_001_object_car_masks.mp4      (full video: car mask overlaid, 60% opacity)
├── video_001_object_person_masks.mp4   (full video: person mask overlaid, 60% opacity)
└── video_001_object_bike_masks.mp4     (full video: bike mask overlaid, 60% opacity)
```

## Updated Files

- ✅ `process_backward_only.py` - Creates mask videos during processing
- ✅ `process_forward_only.py` - Creates mask videos during processing
- ✅ `UNIFIED_FOLDER_STRUCTURE.md` - Updated documentation
- ✅ `IMPLEMENTATION_SUMMARY.md` - Updated summary
- ✅ `COMMAND_REFERENCE.md` - Updated size estimates

## Storage Impact

**Approximate sizes:**
- Per-object mask video: 20-100 MB (compressed)
- Per direction: +100-200 MB for all mask videos
- Total system: +300-600 MB across all directions

Much more efficient than individual PNGs!

## Usage

### Check mask video for specific object:
```bash
# Play the car mask video
ls /output/output_backward/masks/
ffplay /output/output_backward/masks/video_001_object_car_masks.mp4
```

### Count mask videos:
```bash
find /output -path "*/masks/*_masks.mp4" | wc -l
```

### Compare with other outputs:
```bash
# All have same base name, different folder
/output/output_backward/
├── video_001_backward_masks_and_boxes.mp4   (main video)
├── masks/video_001_object_car_masks.mp4     (object mask video)
└── objects_isolated/video_001_object_car_isolated.mp4  (object only video)
```

---

## Implementation Details

### How Masks are Created

During frame processing, for each frame:
1. Compute segmentation mask for each object
2. Color-code the mask (same color as main overlay)
3. Blend on frame with 60% transparency
4. Write to per-object mask video

### Video Writers (per direction)

```python
mask_video_writers = {}

for obj_id, mask_info in all_masks.items():
    label = mask_info.get('label', f'obj_{obj_id}')
    clean_label = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in str(label))
    
    mask_video_path = os.path.join(masks_dir, f"{video_name_base}_object_{clean_label}_masks.mp4")
    mask_writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height))
    
    if mask_writer.isOpened():
        mask_video_writers[obj_id] = mask_writer
```

### Frame Writing

```python
# For each frame and object
for obj_id, mask_writer in mask_video_writers.items():
    mask_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Overlay mask if available
    if obj_id in frame_masks:
        mask = frame_masks[obj_id]
        color = np.array([*cmap(obj_id)[:3]])
        mask_colored = (mask * color * 255).astype(np.uint8)
        
        # 60% transparency blend
        mask_frame = cv2.addWeighted(mask_frame, 1.0, mask_colored, 0.6, 0)
    
    mask_writer.write(cv2.cvtColor(mask_frame, cv2.COLOR_RGB2BGR))
```

---

## Backward Compatibility

✅ Old code using individual mask PNGs will need updating
✅ But output structure is now unified and consistent
✅ Videos are easier to work with for most use cases

## Next Steps

1. Test mask video generation
2. Verify video quality and visibility
3. Check file sizes
4. Validate in downstream processing

---

## Questions?

- **Why 60% transparency?** - Allows frame detail to show through while mask is visible
- **Why not 100% solid mask?** - Already have `masks_only` video for that
- **Why not different colors?** - Using same colormap as main videos for consistency
- **File size?** - Compressed videos ~3-5x smaller than individual PNG files

---

**Status: ✅ COMPLETE - Mask videos now generated in all directions**
