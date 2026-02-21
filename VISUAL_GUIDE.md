# Visual Guide - Unified Bidirectional Pipeline

## 🎯 What Changed?

### Before vs After

```
BEFORE (Inconsistent):
├── Backward had different structure than Forward
├── Forward used different folder naming
├── Combined was different again
└── No standard for output organization

AFTER (Unified):
├── All use: masks/ + objects_cropped/ + objects_isolated/
├── Consistent video naming
├── Organized by input folder name
└── Same structure everywhere
```

---

## 📁 Folder Structure at a Glance

### Single Video Processing:
```
Input: /path/to/video_1 + annotation.json

With --output-folder /output:
/output/
└── video_1/
    ├── output_backward/      ← Processes frames backward from annotated
    │   ├── 4 videos          
    │   ├── masks/            ← NEW! Individual mask images
    │   ├── objects_cropped/   ← Same label, cropped to box
    │   └── objects_isolated/  ← Same label, full frame
    │
    ├── output_forward/       ← Processes frames forward from annotated
    │   ├── 4 videos
    │   ├── masks/            ← NEW! Individual mask images
    │   ├── objects_cropped/
    │   └── objects_isolated/
    │
    └── output_combined/      ← Merges backward + forward (no duplication)
        ├── 4 videos
        ├── masks/            ← NEW! Individual mask images
        ├── objects_cropped/
        └── objects_isolated/
```

### Batch Processing:
```
Input: /data/ (contains multiple video folders)

With --output-folder /output:
/output/
├── video_1/
│   ├── output_backward/
│   ├── output_forward/
│   └── output_combined/
│
├── video_2/
│   ├── output_backward/
│   ├── output_forward/
│   └── output_combined/
│
└── video_N/
    ├── output_backward/
    ├── output_forward/
    └── output_combined/
```

---

## 🎬 Video Outputs

### Each Direction (Backward, Forward, or Combined) produces:

```
1. MASKS_AND_BOXES
   ├── Shows: Segmentation masks + Bounding boxes
   └── Use: Full visualization of tracking results

2. BOXES
   ├── Shows: Only bounding boxes
   └── Use: Clean box visualization without clutter

3. MASKS_OVERLAID
   ├── Shows: Masks blended with original (60% transparent)
   └── Use: See context while visualizing segmentation

4. MASKS_ONLY
   ├── Shows: Solid masks on black background
   └── Use: Pure segmentation without scene context
```

---

## 🏷️ Naming Convention

### Main Videos:
```
{video_name}_backward_masks_and_boxes.mp4       ← Backward
{video_name}_processed_masks_and_boxes.mp4      ← Forward (uses 'processed')
{video_name}_combined_masks_and_boxes.mp4       ← Combined
```

### Per-Object Videos:
```
objects_cropped/
├── {video_name}_object_{label1}_backward_cropped.mp4
├── {video_name}_object_{label2}_backward_cropped.mp4
└── ...

objects_isolated/
├── {video_name}_object_{label1}_backward_isolated.mp4
├── {video_name}_object_{label2}_backward_isolated.mp4
└── ...
```

### Mask Images (NEW):
```
masks/
├── object_{label1}/
│   ├── frame_00000_mask.png    ← Binary mask for object 1, frame 0
│   ├── frame_00001_mask.png    ← Binary mask for object 1, frame 1
│   └── ...
├── object_{label2}/
│   ├── frame_00000_mask.png
│   ├── frame_00001_mask.png
│   └── ...
└── ...
```

---

## 📊 Processing Flow

```
Original Video (60 frames) + Annotation (frame 31)
    │
    ├─→ BACKWARD PIPELINE ────────────────────→
    │   Processes: frames 30, 29, 28, ..., 1
    │   Output: 30 frames (reversed to chrono)
    │   Saves: output_backward/ (4 videos + objects + masks)
    │
    ├─→ FORWARD PIPELINE ─────────────────────→
    │   Processes: frames 31, 32, 33, ..., 60
    │   Output: 30 frames
    │   Saves: output_forward/ (4 videos + objects + masks)
    │
    └─→ COMBINE (Smart Merging) ───────────────→
        Backward (30 frames) + Forward (30 frames)
        MINUS Annotated Frame (frame 31)
        Result: 59 frames (no duplication)
        Saves: output_combined/ (4 videos + objects + masks)
```

---

## 🔄 What's New?

### ✨ Individual Mask Images
**Before**: Only video outputs
**After**: Plus individual PNG masks per frame per object
```
masks/object_person/
├── frame_00000_mask.png  ← Binary: 0=background, 255=foreground
├── frame_00001_mask.png
└── ...
```

### ✨ Unified Folder Structure
**Before**: Different for each direction
**After**: Identical in all three
```
Each has:
├── 4 main videos
├── masks/
├── objects_cropped/
└── objects_isolated/
```

### ✨ Smart Output Organization
**Before**: Mixed into script directory
**After**: Organized by input folder name
```
--output-folder /output
├── video_1/
├── video_2/
└── video_3/
```

---

## 🚀 Quick Start Commands

### Process single video to custom location:
```bash
python process_bidirectional_combined.py \
  --input-folder /data/my_video \
  --output-folder /output/results \
  --device cuda

# Result: /output/results/my_video/
#         ├── output_backward/
#         ├── output_forward/
#         └── output_combined/
```

### Process multiple videos in batch:
```bash
python process_folder_hira.py \
  --data-folder /data \
  --output-folder /output/results \
  --script process_bidirectional_combined.py \
  --device cuda

# Result: /output/results/
#         ├── video_1/
#         ├── video_2/
#         └── video_3/
```

### Check results:
```bash
tree /output/results/ -L 3

# Expected:
# /output/results/
# ├── video_1/
# │   ├── output_backward/
# │   │   ├── 4 videos
# │   │   ├── masks/
# │   │   ├── objects_cropped/
# │   │   └── objects_isolated/
# │   ├── output_forward/
# │   │   ├── 4 videos
# │   │   ├── masks/
# │   │   ├── objects_cropped/
# │   │   └── objects_isolated/
# │   └── output_combined/
# │       ├── 4 videos
# │       ├── masks/
# │       ├── objects_cropped/
# │       └── objects_isolated/
# └── video_2/
#     ...
```

---

## 🎯 Use Cases

### Use Case 1: Video Analysis
```
1. Run bidirectional pipeline
2. Review combined videos
3. Inspect individual masks in masks/ folder
4. Check per-object cropped videos
```

### Use Case 2: Training Data Generation
```
1. Run bidirectional pipeline
2. Extract masks from masks/ folder
3. Use combined videos + masks for training
4. Segment by object using objects_isolated/ videos
```

### Use Case 3: Batch Processing
```
1. Place videos in /data/
2. Run batch processor with --output-folder
3. Process all automatically
4. Results organized by video name
5. Easy to review and iterate
```

### Use Case 4: Quality Assurance
```
1. Process sample video
2. Review masks_and_boxes.mp4
3. Check individual masks in masks/ for quality
4. Verify box accuracy with boxes.mp4
5. Approve or tweak parameters
```

---

## 📈 Scale Comparison

### Before Changes:
```
❌ Different structure for each direction
❌ Hard to find outputs (mixed locations)
❌ No per-frame masks
❌ Difficult to batch process
❌ Confusing naming conventions
```

### After Changes:
```
✅ Same structure everywhere
✅ Organized by input folder name
✅ Per-frame masks available
✅ Easy batch processing
✅ Clear naming conventions
✅ 3x more complete output
```

---

## 🔍 Inspect Results

### Count videos:
```bash
find /output/results -name "*.mp4" | wc -l
# Expected: ~10-15 per video depending on number of objects
```

### Check masks:
```bash
find /output/results -name "*_mask.png" | wc -l
# Expected: # of objects × # of frames × 3 directions
```

### Check folder structure:
```bash
ls -la /output/results/video_1/output_backward/
# Expected: masks/, objects_cropped/, objects_isolated/ + 4 videos
```

---

## 💾 Storage Estimation

### Per Video:
```
Backward:
├── 4 videos: ~200-400 MB
├── Masks: ~50-100 MB
├── Cropped objects: ~30-50 MB
└── Isolated objects: ~30-50 MB
Total: ~350-600 MB

Forward: ~350-600 MB
Combined: ~350-600 MB

TOTAL PER VIDEO: ~1-1.8 GB
```

### For 100 Videos:
```
Total Storage: 100-180 GB
Processing Time: 25-75 hours (GPU)
```

---

## 🎓 Understanding the Output

### What Each Video Shows:

1. **Masks + Boxes** - Segmentation masks overlaid on frames with bounding boxes
   - Useful for: Full understanding of tracking results
   
2. **Boxes Only** - Just the bounding box rectangles
   - Useful for: Verifying box accuracy
   
3. **Masks Overlaid** - Masks blended semi-transparently with scene
   - Useful for: Seeing both scene context and segmentation
   
4. **Masks Only** - Pure masks without scene context
   - Useful for: Quality assurance of mask accuracy

### Per-Object Videos:

- **Cropped**: Only the region inside the bounding box (small file)
- **Isolated**: Full frame but only the object visible (detailed tracking)

### Mask Images:

- **Binary PNG**: 0 (black) = background, 255 (white) = object
- **One per object per frame**: Enables detailed analysis
- **All three directions**: Different processing for comparison

---

## ✅ Ready to Use!

Your unified bidirectional pipeline is now:
- ✅ Symmetrical across all directions
- ✅ Well-organized by input folder
- ✅ Comprehensive (masks + videos + objects)
- ✅ Production-ready
- ✅ Fully documented

**Start processing!** 🚀
