#!/usr/bin/env python3
"""
SAM2 Backward-Only Video Segmentation Pipeline
Tests backward propagation separately.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from sam2.build_sam import build_sam2_video_predictor


def reverse_video(input_video_path, output_video_path, fps):
    """
    Reverse a video file: read all frames, reverse order, write back.
    Simple approach: just read → reverse → write
    """
    print(f"  Reversing video: {os.path.basename(input_video_path)}...")
    
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read all frames into memory
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    print(f"    Loaded {len(frames)} frames, reversing...")
    
    # Reverse frames
    frames.reverse()
    
    # Write reversed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for i, frame in enumerate(frames):
        out_writer.write(frame)
        if (i + 1) % 10 == 0:
            print(f"    Writing reversed frame {i + 1}/{len(frames)}", end='\r')
    
    out_writer.release()
    
    print(f"    ✓ Reversed video saved: {os.path.basename(output_video_path)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SAM2 Backward-Only Video Segmentation Pipeline',
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
        help='Output folder for generated videos'
    )
    
    parser.add_argument(
        '--processed-folder',
        type=str,
        default=None,
        help='Processed folder for intermediate files'
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
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    
    return device


def find_input_files(input_folder):
    """Find video and JSON annotation files in input folder."""
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    if len(json_files) == 0:
        raise ValueError(f"No JSON annotation file found in {input_folder}")
    elif len(json_files) > 1:
        print(f"⚠ Multiple JSON files found, using first: {json_files[0]}")
    
    annotation_file = os.path.join(input_folder, json_files[0])
    
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


def preprocess_backward(video_path, annotation_file, processed_dir):
    """
    Preprocess video for BACKWARD direction only.
    Returns backward video path, JSON path, and video properties.
    """
    print(f"\n{'='*70}")
    print(f"STEP 1: BACKWARD PREPROCESSING")
    print(f"{'='*70}")
    
    # Load JSON
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Find annotated frame
    first_annotated_frame_no = None
    num_boxes = 0
    
    for frame_data in data["frames"]:
        boxes = frame_data.get("annotation", {}).get("boxes", [])
        if len(boxes) > 0:
            first_annotated_frame_no = frame_data["frameNo"]
            num_boxes = len(boxes)
            print(f"✓ Found annotated frame: frameNo {first_annotated_frame_no}")
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
    cap.release()
    
    print(f"\n📹 Original video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Annotated frame (1-indexed): {first_annotated_frame_no}")
    
    video_name_base = os.path.splitext(os.path.basename(video_path))[0]
    
    # ==================== BACKWARD DIRECTION ====================
    print(f"\n{'─'*70}")
    print(f"BACKWARD: From frame {first_annotated_frame_no} to frame 1 (reversed)")
    print(f"{'─'*70}")
    
    backward_dir = os.path.join(processed_dir, "preprocessed_backward")
    os.makedirs(backward_dir, exist_ok=True)
    
    # Extract frames 0 to first_annotated_frame_no in reverse
    bwd_frame_indices = list(range(0, first_annotated_frame_no))
    bwd_frame_indices.reverse()  # Now in reverse order
    
    trimmed_video_path_bwd = os.path.join(backward_dir, f"{video_name_base}_backward.mp4")
    
    print(f"Extracting and reversing frames 0-{first_annotated_frame_no-1}...")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(trimmed_video_path_bwd, fourcc, fps, (width, height))
    
    bwd_frames_written = 0
    for idx in bwd_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            out_writer.write(frame)
            bwd_frames_written += 1
    
    cap.release()
    out_writer.release()
    
    print(f"✓ Backward video: {bwd_frames_written} frames (reversed)")
    
    # Create backward JSON (frames renumbered, reversed, INCLUDING annotated frame)
    processed_frames_bwd = []
    
    # Get all frames UP TO AND INCLUDING annotated frame
    frames_up_to_annotated = [f for f in data["frames"] 
                              if f["frameNo"] <= first_annotated_frame_no]
    
    # Reverse them
    frames_up_to_annotated.reverse()
    
    for new_idx, frame_data in enumerate(frames_up_to_annotated):
        new_frame_data = frame_data.copy()
        new_frame_data["frameNo"] = new_idx + 1  # Renumber from 1
        new_frame_data["originalFrameNo"] = frame_data["frameNo"]
        processed_frames_bwd.append(new_frame_data)
    
    processed_json_bwd = {
        "taskId": data.get("taskId", ""),
        "campaignId": data.get("campaignId", ""),
        "videoName": f"{video_name_base}_backward",
        "metadata": {
            "fps": fps,
            "totalFrames": bwd_frames_written,
            "direction": "backward",
            "originalAnnotatedFrame": first_annotated_frame_no,
            "processedDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "frames": processed_frames_bwd
    }
    
    processed_json_path_bwd = os.path.join(backward_dir, f"{video_name_base}_backward.json")
    with open(processed_json_path_bwd, 'w') as f:
        json.dump(processed_json_bwd, f, indent=2)
    
    print(f"✓ Backward JSON: {len(processed_frames_bwd)} frames (reversed, INCLUDING annotated)")
    
    print(f"\n{'='*70}")
    print(f"✅ BACKWARD PREPROCESSING COMPLETE")
    print(f"{'='*70}\n")
    
    video_properties = {
        'width': width,
        'height': height,
        'fps': fps,
        'num_boxes': num_boxes,
        'annotated_frame_no': first_annotated_frame_no
    }
    
    backward_props = {
        'video_path': trimmed_video_path_bwd,
        'json_path': processed_json_path_bwd,
        'frames_count': bwd_frames_written,
        'direction': 'backward'
    }
    
    return backward_props, video_properties


def extract_frames(video_path, output_dir, direction, target_fps=20):
    """Extract frames from video at target FPS."""
    print(f"\nExtracting frames for {direction}...")
    
    frames_dir = os.path.join(output_dir, "frames")
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
    
    print(f"  {direction}: {frame_count} frames extracted")
    
    return frames_dir


def load_frames(frames_dir, annotation_file):
    """Load frame names and find first annotated frame."""
    frame_names = [p for p in os.listdir(frames_dir)
                   if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    START_FRAME = None
    num_boxes = 0
    
    for frame_data in data["frames"]:
        boxes = frame_data.get("annotation", {}).get("boxes", [])
        if len(boxes) > 0:
            START_FRAME = frame_data["frameNo"] - 1  # Convert to 0-indexed
            num_boxes = len(boxes)
            break
    
    if START_FRAME is None:
        # No annotated frames in this direction
        return frame_names, data, None
    
    return frame_names, data, START_FRAME


def initialize_objects(predictor, inference_state, data, START_FRAME, width, height):
    """Initialize SAM2 with bounding boxes from annotations."""
    if START_FRAME is None:
        return {}
    
    annotated_boxes = []
    for frame_data in data["frames"]:
        if frame_data["frameNo"] - 1 == START_FRAME:
            annotated_boxes = frame_data["annotation"]["boxes"]
            break
    
    all_masks = {}
    
    for idx, box_data in enumerate(annotated_boxes):
        coords = box_data["coordinates"]
        
        x_coords = [coords[f"point{i}"]["x"] for i in range(1, 5)]
        y_coords = [coords[f"point{i}"]["y"] for i in range(1, 5)]
        
        x_min = int(min(x_coords) * width)
        y_min = int(min(y_coords) * height)
        x_max = int(max(x_coords) * width)
        y_max = int(max(y_coords) * height)
        
        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        
        obj_id = idx + 1
        label = str(box_data.get("label", f"object_{obj_id}"))
        
        print(f"  Object {obj_id} (label={label}): box=[{x_min}, {y_min}, {x_max}, {y_max}]")
        
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=START_FRAME,
            obj_id=obj_id,
            box=box,
        )
        
        mask_idx = list(out_obj_ids).index(obj_id)
        
        all_masks[obj_id] = {
            'mask_logits': out_mask_logits[mask_idx],
            'mask': (out_mask_logits[mask_idx] > 0.0).cpu().numpy(),
            'box': box,
            'label': label
        }
    
    return all_masks


def create_backward_videos(all_masks_bwd, frames_dir_bwd, frame_names_bwd, data_bwd,
                          output_dir, video_name_base, width, height, fps, video_properties,
                          predictor_bwd, inference_state_bwd):
    """
    Create backward output videos with re-reversed frames to chronological order.
    """
    print(f"\n{'='*70}")
    print(f"STEP 3: CREATING BACKWARD OUTPUT VIDEOS")
    print(f"{'='*70}")
    
    # ==================== BACKWARD PROPAGATION ====================
    backward_masks_by_frame = {}
    
    if all_masks_bwd:
        print(f"\nPropagating masks (BACKWARD)...")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor_bwd.propagate_in_video(inference_state_bwd):
            backward_masks_by_frame[out_frame_idx] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                backward_masks_by_frame[out_frame_idx][out_obj_id] = mask
    
    print(f"  Backward: {len(backward_masks_by_frame)} frames with masks (will reverse videos at end)")
    
    # ==================== CREATE OVERLAY VIDEOS ====================
    os.makedirs(output_dir, exist_ok=True)
    
    overlay_both_path = os.path.join(output_dir, f"{video_name_base}_backward_masks_and_boxes.mp4")
    overlay_boxes_path = os.path.join(output_dir, f"{video_name_base}_backward_boxes.mp4")
    overlay_masks_blended_path = os.path.join(output_dir, f"{video_name_base}_backward_masks_overlaid.mp4")
    overlay_masks_only_path = os.path.join(output_dir, f"{video_name_base}_backward_masks_only.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    writer_both = cv2.VideoWriter(overlay_both_path, fourcc, fps, (width, height))
    writer_boxes = cv2.VideoWriter(overlay_boxes_path, fourcc, fps, (width, height))
    writer_masks_blended = cv2.VideoWriter(overlay_masks_blended_path, fourcc, fps, (width, height))
    writer_masks_only = cv2.VideoWriter(overlay_masks_only_path, fourcc, fps, (width, height))
    
    # ==================== CREATE PER-OBJECT CROPPED VIDEOS ====================
    objects_dir = os.path.join(output_dir, "objects_cropped")
    os.makedirs(objects_dir, exist_ok=True)
    
    object_writers = {}
    
    for obj_id, mask_info in all_masks_bwd.items():
        label = mask_info.get('label', f'obj_{obj_id}')
        clean_label = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in str(label))
        
        box = mask_info['box']
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        obj_video_path = os.path.join(objects_dir, f"{video_name_base}_object_{clean_label}_backward_cropped.mp4")
        obj_writer = cv2.VideoWriter(obj_video_path, fourcc, fps, (crop_width, crop_height))
        
        if obj_writer.isOpened():
            object_writers[obj_id] = {
                'writer': obj_writer,
                'box': box,
                'label': label,
                'path': obj_video_path
            }
    
    print(f"\n📹 Output videos:")
    print(f"  4 backward overlay videos (re-reversed to chronological order)")
    print(f"  {len(object_writers)} cropped per-object videos (no mask overlay)")
    
    # ==================== WRITE FRAMES ====================
    cmap = plt.get_cmap("tab10")
    
    print(f"\nWriting BACKWARD frames (in REVERSED order - will reverse videos at end)...")
    for frame_idx in range(len(frame_names_bwd)):
        frame_path = os.path.join(frames_dir_bwd, frame_names_bwd[frame_idx])
        frame = cv2.imread(frame_path)
        
        if frame is None:
            continue
        
        overlay_both = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay_boxes = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay_masks_blended = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        overlay_masks_only = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw masks (using DIRECT index - no mapping needed!)
        if frame_idx in backward_masks_by_frame:
            for obj_id, mask in backward_masks_by_frame[frame_idx].items():
                color = np.array([*cmap(obj_id)[:3]])
                mask_3channel = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                color_rgb = color.reshape(1, 1, 3)
                mask_colored = (mask_3channel * color_rgb * 255).astype(np.uint8)
                
                overlay_both = cv2.addWeighted(overlay_both, 1.0, mask_colored, 0.6, 0)
                overlay_masks_blended = cv2.addWeighted(overlay_masks_blended, 1.0, mask_colored, 0.6, 0)
                overlay_masks_only = cv2.add(overlay_masks_only, mask_colored)
        
        # Draw boxes
        for obj_id, mask_info in all_masks_bwd.items():
            box = mask_info['box']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            label = mask_info.get('label', f'obj_{obj_id}')
            
            cv2.rectangle(overlay_both, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay_both, str(label), (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.rectangle(overlay_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay_boxes, str(label), (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write overlay videos
        writer_both.write(cv2.cvtColor(overlay_both, cv2.COLOR_RGB2BGR))
        writer_boxes.write(cv2.cvtColor(overlay_boxes, cv2.COLOR_RGB2BGR))
        writer_masks_blended.write(cv2.cvtColor(overlay_masks_blended, cv2.COLOR_RGB2BGR))
        writer_masks_only.write(cv2.cvtColor(overlay_masks_only, cv2.COLOR_RGB2BGR))
        
        # Write cropped per-object videos
        for obj_id, obj_info in object_writers.items():
            obj_writer = obj_info['writer']
            box = obj_info['box']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            cropped_frame = frame[y1:y2, x1:x2]
            
            if cropped_frame.size > 0:
                obj_writer.write(cropped_frame)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Progress: {frame_idx + 1}/{len(frame_names_bwd)} frames", end='\r')
    
    # Release writers
    writer_both.release()
    writer_boxes.release()
    writer_masks_blended.release()
    writer_masks_only.release()
    
    for obj_info in object_writers.values():
        obj_info['writer'].release()
    
    # ==================== SUMMARY ====================
    print(f"\n\n{'='*70}")
    print(f"REVERSING VIDEOS TO CHRONOLOGICAL ORDER")
    print(f"{'='*70}")
    
    # Reverse all 4 overlay videos
    print(f"\nReversing overlay videos...")
    videos_to_reverse = [
        (overlay_both_path, overlay_both_path.replace('.mp4', '_chronological.mp4')),
        (overlay_boxes_path, overlay_boxes_path.replace('.mp4', '_chronological.mp4')),
        (overlay_masks_blended_path, overlay_masks_blended_path.replace('.mp4', '_chronological.mp4')),
        (overlay_masks_only_path, overlay_masks_only_path.replace('.mp4', '_chronological.mp4'))
    ]
    
    for reversed_path, chrono_path in videos_to_reverse:
        if os.path.exists(reversed_path):
            reverse_video(reversed_path, chrono_path, fps)
            # Replace original with chronological version
            os.remove(reversed_path)
            os.rename(chrono_path, reversed_path)
    
    # Reverse all cropped object videos
    print(f"\nReversing cropped object videos...")
    for obj_id, obj_info in object_writers.items():
        obj_path = obj_info['path']
        if os.path.exists(obj_path):
            # Get crop dimensions
            box = obj_info['box']
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crop_width = x2 - x1
            crop_height = y2 - y1
            
            temp_path = obj_path.replace('.mp4', '_temp.mp4')
            reverse_video(obj_path, temp_path, fps)
            os.remove(obj_path)
            os.rename(temp_path, obj_path)
    
    print(f"\n{'='*70}")
    print(f"✅ BACKWARD VIDEO GENERATION COMPLETE")
    print(f"{'='*70}")
    
    print(f"\n📹 Backward Overlay Videos (4 total - chronological order):")
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
        print(f"  {status} {vid_name}: {size:.2f} MB")
    
    print(f"\n📁 objects_cropped/ folder:")
    print(f"  {len(object_writers)} cropped per-object videos (no mask overlay)")
    for obj_id, obj_info in object_writers.items():
        exists = os.path.exists(obj_info['path'])
        size = os.path.getsize(obj_info['path']) / 1024 / 1024 if exists else 0
        status = '✓' if exists else '✗'
        print(f"    {status} {obj_info['label']}: {size:.2f} MB")
    
    total_videos = 4 + len(object_writers)
    print(f"\n📊 Total videos: {total_videos}")
    print(f"   - 4 backward overlay videos (in chronological order)")
    print(f"   - {len(object_writers)} cropped object videos")
    
    print(f"\n{'='*70}")
    print(f"💡 All backward videos saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    input_folder = os.path.abspath(args.input_folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder_name = os.path.basename(input_folder.rstrip('/'))
    
    if args.processed_folder:
        processed_folder = os.path.abspath(args.processed_folder)
    else:
        processed_folder = os.path.join(script_dir, "processed_backward_test", input_folder_name)
    
    if args.output_folder:
        output_folder = os.path.abspath(args.output_folder)
    else:
        output_folder = os.path.join(script_dir, "output_backward_test", input_folder_name)
    
    print(f"\n{'='*70}")
    print(f"SAM2 BACKWARD-ONLY VIDEO SEGMENTATION PIPELINE")
    print(f"{'='*70}")
    print(f"Input folder: {input_folder}")
    print(f"Processed folder: {processed_folder}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*70}\n")
    
    # Setup device
    device = setup_device(args.device)
    
    try:
        # Find input files
        video_path, annotation_file = find_input_files(input_folder)
        
        # Step 1: Backward preprocessing
        backward_props, video_properties = preprocess_backward(
            video_path, annotation_file, processed_folder
        )
        
        # Step 2: Extract frames
        direction_processed_dir = os.path.join(processed_folder, "preprocessed_backward")
        frames_dir_bwd = extract_frames(backward_props['video_path'], direction_processed_dir, 'backward', args.fps)
        
        # Step 3: Load frames
        print(f"\n{'='*70}")
        print(f"STEP 2: LOADING FRAMES & INITIALIZING SAM2")
        print(f"{'='*70}")
        
        frame_names_bwd, data_bwd, START_FRAME = load_frames(frames_dir_bwd, backward_props['json_path'])
        
        print(f"✓ Loaded {len(frame_names_bwd)} frames")
        print(f"✓ First annotated frame (re-reversed to chrono): frameNo {START_FRAME + 1} (0-indexed: {START_FRAME})")
        
        if START_FRAME is None:
            raise ValueError("No annotated frames found!")
        
        # Step 4: Initialize SAM2
        print(f"\n🔧 Initializing SAM2 for backward...")
        predictor_bwd = build_sam2_video_predictor(args.model_config, args.checkpoint, device=device)
        inference_state_bwd = predictor_bwd.init_state(video_path=frames_dir_bwd)
        
        print(f"🔧 Initializing objects...")
        all_masks_bwd = initialize_objects(predictor_bwd, inference_state_bwd, data_bwd, START_FRAME, 
                                          video_properties['width'], video_properties['height'])
        
        print(f"✓ SAM2 ready with {len(all_masks_bwd)} objects")
        
        # Step 5: Create videos
        video_name_base = os.path.splitext(os.path.basename(video_path))[0]
        create_backward_videos(
            all_masks_bwd, frames_dir_bwd, frame_names_bwd, data_bwd,
            output_folder, video_name_base,
            video_properties['width'], video_properties['height'], args.fps, video_properties,
            predictor_bwd, inference_state_bwd
        )
        
        print("\n🎉 Backward pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
