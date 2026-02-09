#!/usr/bin/env python3
"""
SAM2 Video Segmentation Pipeline
Processes annotated videos and generates multiple output videos with masks and bounding boxes.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# Add SAM2 to path if needed
# sys.path.append('/path/to/sam2')

from sam2.build_sam import build_sam2_video_predictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SAM2 Video Segmentation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-folder',
        type=str,
        required=True,
        help='Input folder containing video file and annotation JSON'
    )
    
    parser.add_argument(
        '--output-folder',
        type=str,
        default=None,
        help='Output folder for generated videos (default: input_folder/output_videos)'
    )
    
    parser.add_argument(
        '--processed-folder',
        type=str,
        default=None,
        help='Processed folder for intermediate files (default: input_folder/processed)'
    )
    
    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/sam2.1/sam2.1_hiera_t.yaml',
        help='SAM2 model configuration file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/sam2.1_hiera_tiny.pt',
        help='SAM2 model checkpoint file'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Target FPS for frame extraction and output videos'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to run inference on'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing step (use existing processed files)'
    )
    
    parser.add_argument(
        '--skip-frame-extraction',
        action='store_true',
        help='Skip frame extraction step (use existing frames)'
    )
    
    return parser.parse_args()


def setup_device(device_name):
    """Setup computation device."""
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif device_name == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
        print("Note: SAM2 is trained with CUDA and might give different results on MPS")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    
    return device


def find_input_files(input_folder):
    """Find video and JSON annotation files in input folder."""
    # Find JSON file
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    if len(json_files) == 0:
        raise ValueError(f"No JSON annotation file found in {input_folder}")
    elif len(json_files) > 1:
        print(f"⚠ Multiple JSON files found, using first: {json_files[0]}")
    
    annotation_file = os.path.join(input_folder, json_files[0])
    
    # Find video file
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(input_folder) 
                   if any(f.endswith(ext) for ext in video_extensions)]
    
    if len(video_files) == 0:
        raise ValueError(f"No video file found in {input_folder}")
    elif len(video_files) > 1:
        print(f"⚠ Multiple video files found, using first: {video_files[0]}")
    
    video_path = os.path.join(input_folder, video_files[0])
    
    print(f"✓ Found annotation: {os.path.basename(annotation_file)}")
    print(f"✓ Found video: {os.path.basename(video_path)}")
    
    return video_path, annotation_file


def preprocess_video(video_path, annotation_file, processed_dir):
    """
    Find first annotated frame, trim video, and renumber JSON frames.
    Returns: (processed_video_path, processed_json_path, video_properties)
    """
    print(f"\n{'='*70}")
    print(f"STEP 1: PREPROCESSING - Find annotated frame and trim video")
    print(f"{'='*70}")
    
    # Load JSON
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Find first frame with bounding boxes
    first_annotated_frame_no = None
    num_boxes = 0
    
    for frame_data in data["frames"]:
        boxes = frame_data.get("annotation", {}).get("boxes", [])
        if len(boxes) > 0:
            first_annotated_frame_no = frame_data["frameNo"]
            num_boxes = len(boxes)
            print(f"✓ Found first annotated frame: frameNo {first_annotated_frame_no}")
            print(f"  Number of objects: {num_boxes}")
            break
    
    if first_annotated_frame_no is None:
        raise ValueError("No annotated frames found in JSON!")
    
    # Load video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Original video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    
    # Calculate trim range
    trim_start_frame = first_annotated_frame_no - 1  # Convert to 0-indexed
    frames_to_extract = total_frames - trim_start_frame
    
    print(f"\n✂️  Trimming video:")
    print(f"  Start from frame: {trim_start_frame} (0-indexed)")
    print(f"  Frames to extract: {frames_to_extract}")
    
    # Create processed folder
    os.makedirs(processed_dir, exist_ok=True)
    
    # Trim video
    video_name_base = os.path.splitext(os.path.basename(video_path))[0]
    trimmed_video_path = os.path.join(processed_dir, f"{video_name_base}_processed.mp4")
    
    print(f"\n⏳ Trimming video from frame {trim_start_frame}...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(trimmed_video_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, trim_start_frame)
    frames_written = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        out_writer.write(frame)
        frames_written += 1
        
        if frames_written % 50 == 0:
            print(f"  Progress: {frames_written}/{frames_to_extract} frames", end='\r')
    
    cap.release()
    out_writer.release()
    
    print(f"\n✓ Trimmed video saved: {frames_written} frames")
    
    # Create processed JSON with renumbered frames
    print(f"\n📝 Creating processed JSON with renumbered frames...")
    
    processed_frames = []
    frame_offset = first_annotated_frame_no - 1
    
    for frame_data in data["frames"]:
        original_frame_no = frame_data["frameNo"]
        
        if original_frame_no >= first_annotated_frame_no:
            new_frame_no = original_frame_no - frame_offset
            
            new_frame_data = frame_data.copy()
            new_frame_data["frameNo"] = new_frame_no
            new_frame_data["originalFrameNo"] = original_frame_no
            
            processed_frames.append(new_frame_data)
    
    # Create processed JSON
    processed_json = {
        "taskId": data.get("taskId", ""),
        "campaignId": data.get("campaignId", ""),
        "videoName": f"{video_name_base}_processed",
        "metadata": {
            "fps": fps,
            "totalFrames": frames_written,
            "segmentStart": 1,
            "segmentEnd": frames_written,
            "originalFirstFrame": first_annotated_frame_no,
            "frameOffset": frame_offset,
            "processedDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sourceVideo": os.path.basename(video_path),
            "sourceJSON": os.path.basename(annotation_file)
        },
        "frames": processed_frames
    }
    
    processed_json_path = os.path.join(processed_dir, f"{video_name_base}_processed.json")
    with open(processed_json_path, 'w') as f:
        json.dump(processed_json, f, indent=2)
    
    print(f"✓ Processed JSON saved: {len(processed_frames)} frames")
    
    # Create annotations summary
    annotations_txt_path = os.path.join(processed_dir, "annotations.txt")
    num_annotated = sum(1 for f in processed_frames 
                       if len(f.get("annotation", {}).get("boxes", [])) > 0)
    
    with open(annotations_txt_path, 'w') as f:
        f.write(f"PROCESSED VIDEO ANNOTATIONS\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Source Video: {os.path.basename(video_path)}\n")
        f.write(f"Source JSON: {os.path.basename(annotation_file)}\n")
        f.write(f"Processed Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Video Properties:\n")
        f.write(f"  Resolution: {width}x{height}\n")
        f.write(f"  FPS: {fps}\n")
        f.write(f"  Total Frames: {frames_written}\n\n")
        f.write(f"Processing:\n")
        f.write(f"  Original first annotated frame: frameNo {first_annotated_frame_no}\n")
        f.write(f"  Now starts at: frameNo 1\n")
        f.write(f"  Frame offset: {frame_offset} frames removed\n\n")
        f.write(f"Annotations:\n")
        f.write(f"  Total annotated frames: {num_annotated}\n")
        f.write(f"  Objects in first frame: {num_boxes}\n\n")
    
    print(f"✓ Annotations summary saved")
    
    print(f"\n{'='*70}")
    print(f"✅ PREPROCESSING COMPLETE")
    print(f"{'='*70}\n")
    
    video_properties = {
        'width': width,
        'height': height,
        'fps': fps,
        'num_boxes': num_boxes
    }
    
    return trimmed_video_path, processed_json_path, video_properties


def extract_frames(video_path, processed_dir, target_fps=20):
    """Extract frames from video at target FPS."""
    print(f"\n{'='*70}")
    print(f"STEP 2: EXTRACTING FRAMES")
    print(f"{'='*70}")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(processed_dir, video_name, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_interval = int(original_fps / target_fps) if original_fps > target_fps else 1
    
    idx = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx % frame_interval == 0:
            cv2.imwrite(f"{frames_dir}/{frame_count:05d}.jpg", frame)
            frame_count += 1
        
        idx += 1
    
    cap.release()
    
    print(f"\nOriginal FPS: {original_fps}")
    print(f"Target FPS: {target_fps}")
    print(f"Total frames extracted: {frame_count}")
    print(f"Frames saved to: {frames_dir}")
    print(f"{'='*70}\n")
    
    return frames_dir


def load_frames(frames_dir, annotation_file):
    """Load frame names and find first annotated frame."""
    print(f"\n{'='*70}")
    print(f"STEP 3: LOADING FRAMES")
    print(f"{'='*70}")
    
    frame_names = [p for p in os.listdir(frames_dir)
                   if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    print(f"\nTotal frames loaded: {len(frame_names)}")
    
    # Load JSON
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Find first annotated frame
    START_FRAME = None
    num_boxes = 0
    
    for frame_data in data["frames"]:
        boxes = frame_data.get("annotation", {}).get("boxes", [])
        if len(boxes) > 0:
            START_FRAME = frame_data["frameNo"] - 1  # Convert to 0-indexed
            num_boxes = len(boxes)
            print(f"✓ First annotated frame: frame_idx {START_FRAME} (frameNo {frame_data['frameNo']} in JSON)")
            print(f"  Number of objects: {num_boxes}")
            break
    
    if START_FRAME is None:
        raise ValueError("No annotated frames found!")
    
    print(f"{'='*70}\n")
    
    return frame_names, data, START_FRAME


def initialize_sam2(model_cfg, checkpoint, device, frames_dir):
    """Initialize SAM2 model and inference state."""
    print(f"\n{'='*70}")
    print(f"STEP 4: INITIALIZING SAM2")
    print(f"{'='*70}")
    
    print(f"Model config: {model_cfg}")
    print(f"Checkpoint: {checkpoint}")
    
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    
    print(f"\nInitializing inference state...")
    inference_state = predictor.init_state(video_path=frames_dir)
    
    print(f"✓ SAM2 initialized")
    print(f"{'='*70}\n")
    
    return predictor, inference_state


def initialize_objects(predictor, inference_state, data, START_FRAME, width, height):
    """Initialize SAM2 with bounding boxes from annotations."""
    print(f"\n{'='*70}")
    print(f"STEP 5: INITIALIZING OBJECT MASKS")
    print(f"{'='*70}")
    
    # Extract boxes from annotated frame
    annotated_boxes = []
    for frame_data in data["frames"]:
        if frame_data["frameNo"] - 1 == START_FRAME:
            annotated_boxes = frame_data["annotation"]["boxes"]
            break
    
    print(f"Processing {len(annotated_boxes)} annotated boxes...")
    
    all_masks = {}
    
    for idx, box_data in enumerate(annotated_boxes):
        # Extract normalized coordinates from 4 points
        coords = box_data["coordinates"]
        
        x_coords = [coords[f"point{i}"]["x"] for i in range(1, 5)]
        y_coords = [coords[f"point{i}"]["y"] for i in range(1, 5)]
        
        # Convert to pixel coordinates
        x_min = int(min(x_coords) * width)
        y_min = int(min(y_coords) * height)
        x_max = int(max(x_coords) * width)
        y_max = int(max(y_coords) * height)
        
        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        
        obj_id = idx + 1
        label = str(box_data.get("label", f"object_{obj_id}"))
        
        print(f"  Object {obj_id} (label={label}): box=[{x_min}, {y_min}, {x_max}, {y_max}]")
        
        # Add box to SAM2
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=START_FRAME,
            obj_id=obj_id,
            box=box,
        )
        
        # Find correct mask index
        mask_idx = list(out_obj_ids).index(obj_id)
        
        all_masks[obj_id] = {
            'mask_logits': out_mask_logits[mask_idx],
            'mask': (out_mask_logits[mask_idx] > 0.0).cpu().numpy(),
            'box': box,
            'label': label
        }
    
    print(f"\n✓ Initialized {len(all_masks)} object masks")
    print(f"{'='*70}\n")
    
    return all_masks


def propagate_and_create_videos(predictor, inference_state, all_masks, frames_dir, 
                                frame_names, output_dir, video_name_base, 
                                width, height, fps):
    """Propagate masks and create all output videos."""
    print(f"\n{'='*70}")
    print(f"STEP 6: PROPAGATING MASKS AND CREATING VIDEOS")
    print(f"{'='*70}")
    
    # Create output directory structure
    objects_dir = os.path.join(output_dir, "objects")
    masks_only_dir = os.path.join(output_dir, "masks_only")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(objects_dir, exist_ok=True)
    os.makedirs(masks_only_dir, exist_ok=True)
    
    # Define output paths for 4 overlay videos
    overlay_both_path = os.path.join(output_dir, f"{video_name_base}_masks_and_boxes.mp4")
    overlay_boxes_path = os.path.join(output_dir, f"{video_name_base}_boxes.mp4")
    overlay_masks_blended_path = os.path.join(output_dir, f"{video_name_base}_masks_overlaid.mp4")
    overlay_masks_only_path = os.path.join(output_dir, f"{video_name_base}_masks_only.mp4")
    
    # Initialize video writers for 4 overlay videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_both = cv2.VideoWriter(overlay_both_path, fourcc, fps, (width, height))
    writer_boxes = cv2.VideoWriter(overlay_boxes_path, fourcc, fps, (width, height))
    writer_masks_blended = cv2.VideoWriter(overlay_masks_blended_path, fourcc, fps, (width, height))
    writer_masks_only = cv2.VideoWriter(overlay_masks_only_path, fourcc, fps, (width, height))
    
    if not (writer_both.isOpened() and writer_boxes.isOpened() and 
            writer_masks_blended.isOpened() and writer_masks_only.isOpened()):
        raise RuntimeError("Failed to initialize overlay video writers")
    
    # Initialize video writers for individual objects
    object_writers = {}
    object_video_paths = {}
    
    for obj_id, mask_info in all_masks.items():
        label = mask_info.get('label', f'obj_{obj_id}')
        clean_label = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in str(label))
        
        obj_video_path = os.path.join(objects_dir, f"{video_name_base}_object_{clean_label}_isolated.mp4")
        obj_writer = cv2.VideoWriter(obj_video_path, fourcc, fps, (width, height))
        
        if obj_writer.isOpened():
            object_writers[obj_id] = obj_writer
            object_video_paths[obj_id] = obj_video_path
    
    # Initialize video writers for mask-only videos
    mask_only_writers = {}
    mask_only_video_paths = {}
    
    for obj_id, mask_info in all_masks.items():
        label = mask_info.get('label', f'obj_{obj_id}')
        clean_label = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in str(label))
        
        mask_video_path = os.path.join(masks_only_dir, f"{video_name_base}_mask_{clean_label}.mp4")
        mask_writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height))
        
        if mask_writer.isOpened():
            mask_only_writers[obj_id] = mask_writer
            mask_only_video_paths[obj_id] = mask_video_path
    
    print(f"\n📹 Main Overlay Videos (4 total):")
    print(f"  1. Masks + Boxes: {os.path.basename(overlay_both_path)}")
    print(f"  2. Boxes Only: {os.path.basename(overlay_boxes_path)}")
    print(f"  3. Masks Overlaid: {os.path.basename(overlay_masks_blended_path)}")
    print(f"  4. Masks Only (solid): {os.path.basename(overlay_masks_only_path)}")
    
    print(f"\n📁 objects/ folder ({len(object_writers)} videos)")
    print(f"📁 masks_only/ folder ({len(mask_only_writers)} videos)")
    
    print(f"\n  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Objects: {len(all_masks)}")
    
    print(f"\n⏳ Propagating masks and writing videos...")
    
    # Get colormap
    cmap = plt.get_cmap("tab10")
    
    # Propagate and write all videos simultaneously
    frame_count = 0
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # Load original frame
        frame_path = os.path.join(frames_dir, frame_names[out_frame_idx])
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"\n⚠ Warning: Could not read frame {out_frame_idx}, skipping...")
            continue
        
        # Create 4 separate overlay frames
        overlay_both = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay_boxes = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay_masks_blended = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay_masks_only = np.zeros((height, width, 3), dtype=np.uint8)
        
        frame_masks = {}
        
        # Draw each object's mask and/or box
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            frame_masks[out_obj_id] = mask
            
            color = np.array([*cmap(out_obj_id)[:3]])
            
            mask_3channel = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            color_rgb = color.reshape(1, 1, 3)
            mask_colored = (mask_3channel * color_rgb * 255).astype(np.uint8)
            
            # Blend masks
            overlay_both = cv2.addWeighted(overlay_both, 1.0, mask_colored, 0.6, 0)
            overlay_masks_blended = cv2.addWeighted(overlay_masks_blended, 1.0, mask_colored, 0.6, 0)
            
            # Add solid mask
            overlay_masks_only = cv2.add(overlay_masks_only, mask_colored)
            
            # Draw bounding boxes
            if out_obj_id in all_masks:
                mask_info = all_masks[out_obj_id]
                box = mask_info['box']
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                label = mask_info.get('label', f'obj_{out_obj_id}')
                
                cv2.rectangle(overlay_both, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay_both, str(label), (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.rectangle(overlay_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay_boxes, str(label), (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write 4 overlay videos
        writer_both.write(cv2.cvtColor(overlay_both, cv2.COLOR_RGB2BGR))
        writer_boxes.write(cv2.cvtColor(overlay_boxes, cv2.COLOR_RGB2BGR))
        writer_masks_blended.write(cv2.cvtColor(overlay_masks_blended, cv2.COLOR_RGB2BGR))
        writer_masks_only.write(cv2.cvtColor(overlay_masks_only, cv2.COLOR_RGB2BGR))
        
        # Write individual object videos
        for obj_id, obj_writer in object_writers.items():
            isolated_frame = np.zeros_like(frame)
            
            if obj_id in frame_masks:
                mask = frame_masks[obj_id]
                mask_bool = mask.astype(bool)
                isolated_frame[mask_bool] = frame[mask_bool]
            
            obj_writer.write(isolated_frame)
        
        # Write mask-only videos
        for obj_id, mask_writer in mask_only_writers.items():
            mask_only_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            if obj_id in frame_masks:
                mask = frame_masks[obj_id]
                color = np.array([*cmap(obj_id)[:3]])
                
                mask_3channel = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                color_rgb = color.reshape(1, 1, 3)
                mask_colored = (mask_3channel * color_rgb * 255).astype(np.uint8)
                
                mask_only_frame = mask_colored
            
            mask_writer.write(cv2.cvtColor(mask_only_frame, cv2.COLOR_RGB2BGR))
        
        frame_count += 1
        
        if frame_count % 10 == 0:
            progress = frame_count / len(frame_names) * 100
            print(f"  Progress: {frame_count}/{len(frame_names)} frames ({progress:.1f}%)", end='\r')
    
    # Release all writers
    writer_both.release()
    writer_boxes.release()
    writer_masks_blended.release()
    writer_masks_only.release()
    
    for obj_writer in object_writers.values():
        obj_writer.release()
    
    for mask_writer in mask_only_writers.values():
        mask_writer.release()
    
    # Verify outputs
    print(f"\n\n{'='*70}")
    print(f"✅ VIDEO GENERATION COMPLETE")
    print(f"{'='*70}")
    
    print(f"\n📹 Main Overlay Videos (4 total):")
    videos_to_check = [
        ("Masks + Boxes", overlay_both_path),
        ("Boxes Only", overlay_boxes_path),
        ("Masks Overlaid", overlay_masks_blended_path),
        ("Masks Only (solid)", overlay_masks_only_path)
    ]
    
    for vid_name, vid_path in videos_to_check:
        exists = os.path.exists(vid_path)
        size = os.path.getsize(vid_path) / 1024 / 1024 if exists else 0
        status = '✓' if exists else '✗'
        print(f"  {vid_name}: {size:.2f} MB {status}")
    
    print(f"\n📁 objects/ folder: {len(object_video_paths)} videos")
    print(f"📁 masks_only/ folder: {len(mask_only_video_paths)} videos")
    
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Objects tracked: {len(all_masks)}")
    print(f"  Total videos: {4 + len(object_video_paths) + len(mask_only_video_paths)}")
    print(f"{'='*70}")
    print(f"\n💡 All videos saved to: {output_dir}")
    print(f"   - Main videos: {output_dir}")
    print(f"   - Object videos: {objects_dir}")
    print(f"   - Mask videos: {masks_only_dir}")
    print(f"{'='*70}\n")


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    # Setup paths
    input_folder = os.path.abspath(args.input_folder)
    
    # Get the base directory (where script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get input folder name for subdirectory
    input_folder_name = os.path.basename(input_folder.rstrip('/'))
    
    # Set default paths with input folder name as subdirectory
    if args.processed_folder:
        processed_folder = os.path.abspath(args.processed_folder)
    else:
        processed_folder = os.path.join(script_dir, "processed", input_folder_name)
    
    if args.output_folder:
        output_folder = os.path.abspath(args.output_folder)
    else:
        output_folder = os.path.join(script_dir, "output", input_folder_name)
    
    print(f"\n{'='*70}")
    print(f"SAM2 VIDEO SEGMENTATION PIPELINE")
    print(f"{'='*70}")
    print(f"Input folder: {input_folder}")
    print(f"Processed folder: {processed_folder}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*70}\n")
    
    # Setup device
    device = setup_device(args.device)
    
    # Find input files
    video_path, annotation_file = find_input_files(input_folder)
    
    # Step 1: Preprocessing
    if args.skip_preprocessing:
        print("\n⏭️  Skipping preprocessing (using existing processed files)")
        # Find processed files
        processed_files = [f for f in os.listdir(processed_folder) if f.endswith('_processed.mp4')]
        if not processed_files:
            raise ValueError(f"No processed video found in {processed_folder}")
        video_path = os.path.join(processed_folder, processed_files[0])
        
        json_files = [f for f in os.listdir(processed_folder) if f.endswith('_processed.json')]
        if not json_files:
            raise ValueError(f"No processed JSON found in {processed_folder}")
        annotation_file = os.path.join(processed_folder, json_files[0])
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        num_boxes = len(data["frames"][0].get("annotation", {}).get("boxes", []))
        
        video_properties = {'width': width, 'height': height, 'fps': fps, 'num_boxes': num_boxes}
    else:
        video_path, annotation_file, video_properties = preprocess_video(
            video_path, annotation_file, processed_folder
        )
    
    # Step 2: Frame extraction
    if args.skip_frame_extraction:
        print("\n⏭️  Skipping frame extraction (using existing frames)")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_dir = os.path.join(processed_folder, video_name, "frames")
        if not os.path.exists(frames_dir):
            raise ValueError(f"Frames directory not found: {frames_dir}")
    else:
        frames_dir = extract_frames(video_path, processed_folder, args.fps)
    
    # Step 3: Load frames
    frame_names, data, START_FRAME = load_frames(frames_dir, annotation_file)
    
    # Step 4: Initialize SAM2
    predictor, inference_state = initialize_sam2(
        args.model_config, args.checkpoint, device, frames_dir
    )
    
    # Step 5: Initialize objects
    all_masks = initialize_objects(
        predictor, inference_state, data, START_FRAME, 
        video_properties['width'], video_properties['height']
    )
    
    # Step 6: Propagate and create videos
    video_name_base = os.path.splitext(os.path.basename(video_path))[0]
    propagate_and_create_videos(
        predictor, inference_state, all_masks, frames_dir, frame_names,
        output_folder, video_name_base, 
        video_properties['width'], video_properties['height'], args.fps
    )
    
    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()