#!/usr/bin/env python3
"""
SAM2 Bidirectional Combined Video Segmentation Pipeline
Processes video both BACKWARD and FORWARD, then combines output videos.
Ensures annotated frame appears only once in final output.
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
import subprocess
import tempfile

from sam2.build_sam import build_sam2_video_predictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SAM2 Bidirectional Combined Video Segmentation Pipeline',
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
        help='Output folder for combined videos'
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
    
    parser.add_argument(
        '--skip-backward',
        action='store_true',
        help='Skip backward processing (only run forward)'
    )
    
    parser.add_argument(
        '--skip-forward',
        action='store_true',
        help='Skip forward processing (only run backward)'
    )
    
    return parser.parse_args()


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


def find_annotated_frame(annotation_file):
    """Find the first frame with annotations."""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    for frame_data in data["frames"]:
        boxes = frame_data.get("annotation", {}).get("boxes", [])
        if len(boxes) > 0:
            return frame_data["frameNo"]
    
    return None


def run_backward_pipeline(input_folder, processed_folder, output_folder_bwd, args):
    """Run the backward processing pipeline."""
    print(f"\n{'='*70}")
    print(f"RUNNING BACKWARD PIPELINE")
    print(f"{'='*70}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backward_script = os.path.join(script_dir, 'process_backward_only.py')
    
    cmd = [
        'python', backward_script,
        '--input-folder', input_folder,
        '--processed-folder', os.path.join(processed_folder, 'backward'),
        '--output-folder', output_folder_bwd,
        '--model-config', args.model_config,
        '--checkpoint', args.checkpoint,
        '--fps', str(args.fps),
        '--device', args.device
    ]
    
    # Run from script directory so relative paths work
    result = subprocess.run(cmd, capture_output=False, cwd=script_dir)
    
    if result.returncode != 0:
        raise RuntimeError(f"Backward pipeline failed with return code {result.returncode}")
    
    print(f"✅ Backward pipeline completed")


def run_forward_pipeline(input_folder, processed_folder, output_folder_fwd, args):
    """Run the forward processing pipeline."""
    print(f"\n{'='*70}")
    print(f"RUNNING FORWARD PIPELINE")
    print(f"{'='*70}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    forward_script = os.path.join(script_dir, 'process_forward_only.py')
    
    cmd = [
        'python', forward_script,
        '--input-folder', input_folder,
        '--processed-folder', os.path.join(processed_folder, 'forward'),
        '--output-folder', output_folder_fwd,
        '--model-config', args.model_config,
        '--checkpoint', args.checkpoint,
        '--fps', str(args.fps),
        '--device', args.device
    ]
    
    # Run from script directory so relative paths work
    result = subprocess.run(cmd, capture_output=False, cwd=script_dir)
    
    if result.returncode != 0:
        raise RuntimeError(f"Forward pipeline failed with return code {result.returncode}")
    
    print(f"✅ Forward pipeline completed")


def combine_videos(video_bwd_path, video_fwd_path, output_path, fps, skip_bwd_last_frame=True):
    """
    Combine two videos (backward + forward) into one.
    If skip_bwd_last_frame=True, skip the last frame of backward video to avoid annotated frame duplication.
    """
    print(f"  Combining: {os.path.basename(video_bwd_path)} + {os.path.basename(video_fwd_path)}")
    
    # Read backward video
    cap_bwd = cv2.VideoCapture(video_bwd_path)
    frames_bwd = []
    
    while True:
        ret, frame = cap_bwd.read()
        if not ret:
            break
        frames_bwd.append(frame)
    
    cap_bwd.release()
    
    # Read forward video
    cap_fwd = cv2.VideoCapture(video_fwd_path)
    frames_fwd = []
    
    while True:
        ret, frame = cap_fwd.read()
        if not ret:
            break
        frames_fwd.append(frame)
    
    cap_fwd.release()
    
    # Combine frames
    if skip_bwd_last_frame and len(frames_bwd) > 0:
        # Skip last frame of backward video (it's the annotated frame, which is in forward video as first frame)
        combined_frames = frames_bwd[:-1] + frames_fwd
        skip_reason = "(skipping last backward frame to avoid duplication)"
    else:
        combined_frames = frames_bwd + frames_fwd
        skip_reason = ""
    
    print(f"    Backward frames: {len(frames_bwd)}, Forward frames: {len(frames_fwd)}")
    print(f"    Combined: {len(combined_frames)} frames {skip_reason}")
    
    # Get video properties
    if len(combined_frames) > 0:
        height, width = combined_frames[0].shape[:2]
    else:
        raise ValueError("No frames to combine!")
    
    # Write combined video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i, frame in enumerate(combined_frames):
        out_writer.write(frame)
        if (i + 1) % 50 == 0:
            print(f"    Writing: {i + 1}/{len(combined_frames)} frames", end='\r')
    
    out_writer.release()
    print(f"    ✓ Combined video saved: {os.path.basename(output_path)}")


def combine_per_object_videos(objects_bwd_dir, objects_fwd_dir, objects_combined_dir, fps):
    """Combine per-object cropped videos from backward and forward."""
    print(f"\n  Combining per-object videos...")
    
    os.makedirs(objects_combined_dir, exist_ok=True)
    
    # Find all backward object videos
    bwd_videos = {}
    if os.path.exists(objects_bwd_dir):
        for f in os.listdir(objects_bwd_dir):
            if f.endswith('.mp4') and 'backward_cropped' in f:
                # Extract object label from filename
                # Format: {video_name}_object_{label}_backward_cropped.mp4
                parts = f.replace('_backward_cropped.mp4', '').split('_object_')
                if len(parts) == 2:
                    label = parts[1]
                    bwd_videos[label] = os.path.join(objects_bwd_dir, f)
    
    # Find all forward object videos
    fwd_videos = {}
    if os.path.exists(objects_fwd_dir):
        for f in os.listdir(objects_fwd_dir):
            if f.endswith('.mp4') and 'isolated' in f:
                # Extract object label from filename
                # Format: {video_name}_object_{label}_isolated.mp4
                parts = f.replace('_isolated.mp4', '').split('_object_')
                if len(parts) == 2:
                    label = parts[1]
                    fwd_videos[label] = os.path.join(objects_fwd_dir, f)
    
    # Combine matching pairs
    combined_count = 0
    all_labels = set(bwd_videos.keys()) | set(fwd_videos.keys())
    
    for label in sorted(all_labels):
        video_bwd = bwd_videos.get(label)
        video_fwd = fwd_videos.get(label)
        
        if video_bwd and video_fwd:
            # Both exist, combine them
            output_name = f"combined_object_{label}_cropped.mp4"
            output_path = os.path.join(objects_combined_dir, output_name)
            
            try:
                combine_videos(video_bwd, video_fwd, output_path, fps, skip_bwd_last_frame=True)
                combined_count += 1
            except Exception as e:
                print(f"    ⚠ Failed to combine {label}: {str(e)}")
        
        elif video_bwd and not video_fwd:
            # Only backward, copy it
            output_name = f"combined_object_{label}_backward_only.mp4"
            output_path = os.path.join(objects_combined_dir, output_name)
            import shutil
            shutil.copy(video_bwd, output_path)
            print(f"    ✓ Copied backward-only video: {output_name}")
            combined_count += 1
        
        elif video_fwd and not video_bwd:
            # Only forward, copy it
            output_name = f"combined_object_{label}_forward_only.mp4"
            output_path = os.path.join(objects_combined_dir, output_name)
            import shutil
            shutil.copy(video_fwd, output_path)
            print(f"    ✓ Copied forward-only video: {output_name}")
            combined_count += 1
    
    print(f"  ✓ Combined {combined_count} per-object videos")
    
    return combined_count


def combine_per_object_videos_v2(objects_bwd_dir, objects_fwd_dir, objects_combined_dir, fps):
    """Combine per-object cropped videos from backward and forward (handles different naming)."""
    print(f"\n  Combining per-object videos...")
    
    os.makedirs(objects_combined_dir, exist_ok=True)
    
    # Find all backward object videos
    bwd_videos = {}
    if objects_bwd_dir and os.path.exists(objects_bwd_dir):
        for f in os.listdir(objects_bwd_dir):
            if f.endswith('.mp4') and 'backward_cropped' in f:
                # Extract object label from filename
                # Format: {video_name}_object_{label}_backward_cropped.mp4
                parts = f.replace('_backward_cropped.mp4', '').split('_object_')
                if len(parts) == 2:
                    label = parts[1]
                    bwd_videos[label] = os.path.join(objects_bwd_dir, f)
    
    # Find all forward object videos
    # Forward uses 'isolated' naming in 'objects' folder
    fwd_videos = {}
    if objects_fwd_dir and os.path.exists(objects_fwd_dir):
        for f in os.listdir(objects_fwd_dir):
            if f.endswith('.mp4') and 'isolated' in f:
                # Extract object label from filename
                # Format: {video_name}_object_{label}_isolated.mp4
                parts = f.replace('_isolated.mp4', '').split('_object_')
                if len(parts) == 2:
                    label = parts[1]
                    fwd_videos[label] = os.path.join(objects_fwd_dir, f)
    
    # Combine matching pairs
    combined_count = 0
    all_labels = set(bwd_videos.keys()) | set(fwd_videos.keys())
    
    if not all_labels:
        print(f"  ℹ️  No per-object videos to combine")
        return combined_count
    
    for label in sorted(all_labels):
        video_bwd = bwd_videos.get(label)
        video_fwd = fwd_videos.get(label)
        
        if video_bwd and video_fwd:
            # Both exist, combine them
            output_name = f"combined_object_{label}_cropped.mp4"
            output_path = os.path.join(objects_combined_dir, output_name)
            
            try:
                combine_videos(video_bwd, video_fwd, output_path, fps, skip_bwd_last_frame=True)
                combined_count += 1
            except Exception as e:
                print(f"    ⚠ Failed to combine {label}: {str(e)}")
        
        elif video_bwd and not video_fwd:
            # Only backward, copy it
            output_name = f"combined_object_{label}_backward_only.mp4"
            output_path = os.path.join(objects_combined_dir, output_name)
            import shutil
            shutil.copy(video_bwd, output_path)
            print(f"    ✓ Copied backward-only: {output_name}")
            combined_count += 1
        
        elif video_fwd and not video_bwd:
            # Only forward, copy it
            output_name = f"combined_object_{label}_forward_only.mp4"
            output_path = os.path.join(objects_combined_dir, output_name)
            import shutil
            shutil.copy(video_fwd, output_path)
            print(f"    ✓ Copied forward-only: {output_name}")
            combined_count += 1
    
    print(f"  ✓ Combined {combined_count} per-object videos")
    
    return combined_count


def combine_object_videos_by_type(objects_bwd_dir, objects_fwd_dir, objects_combined_dir, fps, 
                                   output_type, bwd_suffix, fwd_suffix):
    """
    Combine per-object videos from backward and forward based on type.
    Extracts object labels using regex to handle complex filenames.
    """
    import re
    print(f"  Combining {output_type} object videos...")
    
    os.makedirs(objects_combined_dir, exist_ok=True)
    
    # Find all backward object videos
    bwd_videos = {}
    if objects_bwd_dir and os.path.exists(objects_bwd_dir):
        for f in os.listdir(objects_bwd_dir):
            if f.endswith('.mp4') and '_object_' in f:
                # Extract object label using regex: _object_([^_]+)_
                match = re.search(r'_object_([^_]+)_', f)
                if match:
                    label = match.group(1)
                    bwd_videos[label] = os.path.join(objects_bwd_dir, f)
    
    # Find all forward object videos
    fwd_videos = {}
    if objects_fwd_dir and os.path.exists(objects_fwd_dir):
        for f in os.listdir(objects_fwd_dir):
            if f.endswith('.mp4') and '_object_' in f:
                # Extract object label using regex: _object_([^_]+)_
                match = re.search(r'_object_([^_]+)_', f)
                if match:
                    label = match.group(1)
                    fwd_videos[label] = os.path.join(objects_fwd_dir, f)
    
    # Combine matching pairs
    combined_count = 0
    all_labels = set(bwd_videos.keys()) | set(fwd_videos.keys())
    
    if not all_labels:
        print(f"  ℹ️  No {output_type} videos to combine")
        return combined_count
    
    for label in sorted(all_labels):
        video_bwd = bwd_videos.get(label)
        video_fwd = fwd_videos.get(label)
        
        if video_bwd and video_fwd:
            # Both exist, combine them
            output_name = f"combined_object_{label}_{output_type}.mp4"
            output_path = os.path.join(objects_combined_dir, output_name)
            
            try:
                combine_videos(video_bwd, video_fwd, output_path, fps, skip_bwd_last_frame=True)
                combined_count += 1
            except Exception as e:
                print(f"    ⚠ Failed to combine {label}: {str(e)}")
        
        elif video_bwd and not video_fwd:
            # Only backward, copy it
            output_name = f"combined_object_{label}_{output_type}_backward_only.mp4"
            output_path = os.path.join(objects_combined_dir, output_name)
            import shutil
            shutil.copy(video_bwd, output_path)
            print(f"    ✓ Copied backward-only: {output_name}")
            combined_count += 1
        
        elif video_fwd and not video_bwd:
            # Only forward, copy it
            output_name = f"combined_object_{label}_{output_type}_forward_only.mp4"
            output_path = os.path.join(objects_combined_dir, output_name)
            import shutil
            shutil.copy(video_fwd, output_path)
            print(f"    ✓ Copied forward-only: {output_name}")
            combined_count += 1
    
    print(f"  ✓ Combined {combined_count} {output_type} videos")
    
    return combined_count


def combine_outputs(output_bwd_folder, output_fwd_folder, output_combined_folder, fps, video_name_base):
    """Combine all output videos from backward and forward pipelines."""
    print(f"\n{'='*70}")
    print(f"COMBINING OUTPUT VIDEOS")
    print(f"{'='*70}")
    
    os.makedirs(output_combined_folder, exist_ok=True)
    
    # Check which outputs exist
    bwd_exists = os.path.exists(output_bwd_folder)
    fwd_exists = os.path.exists(output_fwd_folder)
    
    print(f"\nChecking outputs:")
    print(f"  Backward folder: {'✓ Found' if bwd_exists else '✗ Not found'}")
    print(f"  Forward folder:  {'✓ Found' if fwd_exists else '✗ Not found'}")
    
    if not bwd_exists and not fwd_exists:
        print(f"\n⚠️  Warning: Neither backward nor forward output found!")
        print(f"   No videos to combine.")
        return
    
    # Video name suffixes for the 4 main overlay videos
    overlay_suffixes_bwd = [
        'masks_and_boxes',
        'boxes',
        'masks_overlaid',
        'masks_only'
    ]
    
    # Forward uses different naming: 'processed_' instead of 'forward_'
    overlay_suffixes_fwd = [
        'masks_and_boxes',
        'boxes',
        'masks_overlaid',
        'masks_only'
    ]

    print(f"\n📹 Processing overlay videos...")
    combined_overlay_count = 0
    
    for suffix_bwd, suffix_fwd in zip(overlay_suffixes_bwd, overlay_suffixes_fwd):
        video_bwd = os.path.join(output_bwd_folder, f"{video_name_base}_backward_{suffix_bwd}.mp4") if bwd_exists else None
        video_fwd = os.path.join(output_fwd_folder, f"{video_name_base}_processed_{suffix_fwd}.mp4") if fwd_exists else None
        
        bwd_file_exists = video_bwd and os.path.exists(video_bwd)
        fwd_file_exists = video_fwd and os.path.exists(video_fwd)
        
        if bwd_file_exists and fwd_file_exists:
            # Both exist, combine them
            output_name = f"{video_name_base}_combined_{suffix_bwd}.mp4"
            output_path = os.path.join(output_combined_folder, output_name)
            combine_videos(video_bwd, video_fwd, output_path, fps, skip_bwd_last_frame=True)
            combined_overlay_count += 1
        
        elif bwd_file_exists and not fwd_file_exists:
            # Only backward, copy it
            output_name = f"{video_name_base}_combined_{suffix_bwd}_backward_only.mp4"
            output_path = os.path.join(output_combined_folder, output_name)
            import shutil
            shutil.copy(video_bwd, output_path)
            print(f"  ✓ Copied backward-only: {output_name}")
            combined_overlay_count += 1
        
        elif fwd_file_exists and not bwd_file_exists:
            # Only forward, copy it
            output_name = f"{video_name_base}_combined_{suffix_fwd}_forward_only.mp4"
            output_path = os.path.join(output_combined_folder, output_name)
            import shutil
            shutil.copy(video_fwd, output_path)
            print(f"  ✓ Copied forward-only: {output_name}")
            combined_overlay_count += 1
    
    # Combine per-object cropped videos
    print(f"\n📁 Processing per-object cropped videos...")
    objects_bwd_cropped = os.path.join(output_bwd_folder, 'objects_cropped') if bwd_exists else None
    objects_fwd_cropped = os.path.join(output_fwd_folder, 'objects_cropped') if fwd_exists else None
    objects_combined_cropped = os.path.join(output_combined_folder, 'objects_cropped')
    
    combine_object_videos_by_type(objects_bwd_cropped, objects_fwd_cropped, objects_combined_cropped, 
                                    fps, 'cropped', 'backward_cropped', 'cropped')
    
    # Combine per-object isolated (uncropped) videos
    print(f"\n📁 Processing per-object isolated videos...")
    objects_bwd_isolated = os.path.join(output_bwd_folder, 'objects_isolated') if bwd_exists else None
    objects_fwd_isolated = os.path.join(output_fwd_folder, 'objects_isolated') if fwd_exists else None
    objects_combined_isolated = os.path.join(output_combined_folder, 'objects_isolated')
    
    combine_object_videos_by_type(objects_bwd_isolated, objects_fwd_isolated, objects_combined_isolated, 
                                    fps, 'isolated', 'backward_isolated', 'isolated')
    
    print(f"\n{'='*70}")
    print(f"✅ OUTPUT COMBINATION COMPLETE")
    print(f"{'='*70}")
    
    # Summary
    print(f"\n📊 Final Combined Output Structure:")
    print(f"  {output_combined_folder}/")
    print(f"    ├── {video_name_base}_combined_*.mp4  (4 overlay videos)")
    print(f"    ├── objects_cropped/")
    print(f"    │   └── combined_object_*.mp4  (per-object cropped)")
    print(f"    └── objects_isolated/")
    print(f"        └── combined_object_*.mp4  (per-object uncropped)")
    
    print(f"\n✅ Output saved to: {output_combined_folder}")


def find_object_bounding_box(frame):
    """Find tight bounding box of the object (non-black pixels)."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    x_min, y_min = frame.shape[1], frame.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    if x_min >= x_max or y_min >= y_max:
        return None
    
    return x_min, y_min, x_max, y_max


def get_average_bounding_box(video_path, sample_rate=10):
    """Analyze video to find consistent bounding box of object."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    bboxes = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            bbox = find_object_bounding_box(frame)
            if bbox:
                bboxes.append(bbox)
        
        frame_idx += 1
    
    cap.release()
    
    if not bboxes:
        return None
    
    x_mins = [b[0] for b in bboxes]
    y_mins = [b[1] for b in bboxes]
    x_maxs = [b[2] for b in bboxes]
    y_maxs = [b[3] for b in bboxes]
    
    avg_x_min = np.percentile(x_mins, 25)
    avg_y_min = np.percentile(y_mins, 25)
    avg_x_max = np.percentile(x_maxs, 75)
    avg_y_max = np.percentile(y_maxs, 75)
    
    return int(avg_x_min), int(avg_y_min), int(avg_x_max), int(avg_y_max)


def add_padding(bbox, frame_h, frame_w, padding_ratio=0.15):
    """Add padding around bounding box while keeping it within frame bounds."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)
    
    return x1, y1, x2, y2


def postprocess_isolated_objects(output_combined_folder, padding_ratio=0.15):
    """
    Post-process isolated object videos in output_combined folder.
    Creates high-resolution cropped versions with tight bounding boxes.
    """
    objects_isolated_dir = os.path.join(output_combined_folder, "objects_isolated")
    
    if not os.path.exists(objects_isolated_dir):
        print(f"  ⊘ objects_isolated folder not found")
        return False
    
    output_postprocessed_dir = os.path.join(output_combined_folder, "objects_isolated_postprocessed")
    os.makedirs(output_postprocessed_dir, exist_ok=True)
    
    # Find all isolated object videos
    video_files = [f for f in os.listdir(objects_isolated_dir) if 'isolated' in f and f.endswith('.mp4')]
    
    if not video_files:
        print(f"  ⊘ No isolated object videos found in: {objects_isolated_dir}")
        return True
    
    print(f"  🎬 Post-processing {len(video_files)} isolated object videos...")
    
    successful = 0
    skipped = 0
    failed = 0
    
    for video_file in sorted(video_files):
        input_path = os.path.join(objects_isolated_dir, video_file)
        
        # Clean up filename: remove direction suffixes like _backward_only, _forward_only
        clean_filename = video_file.replace('_backward_only', '').replace('_forward_only', '')
        output_path = os.path.join(output_postprocessed_dir, clean_filename)
        
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"     ⊘ Could not open: {video_file}")
                failed += 1
                continue
            
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print(f"     ⊘ Empty video: {video_file}")
                cap.release()
                skipped += 1
                continue
            
            # Find average bounding box
            bbox = get_average_bounding_box(input_path, sample_rate=max(1, total_frames // 30))
            
            if bbox is None:
                print(f"     ⊘ No object detected: {video_file}")
                cap.release()
                skipped += 1
                continue
            
            # Add padding
            bbox = add_padding(bbox, frame_h, frame_w, padding_ratio)
            x1, y1, x2, y2 = bbox
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            if crop_w <= 0 or crop_h <= 0:
                print(f"     ⊘ Invalid crop size: {video_file}")
                cap.release()
                skipped += 1
                continue
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))
            
            if not out_writer.isOpened():
                print(f"     ✗ Failed to create writer: {video_file}")
                cap.release()
                failed += 1
                continue
            
            # Process frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    out_writer.write(cropped)
                    frame_count += 1
            
            cap.release()
            out_writer.release()
            
            if frame_count > 0:
                successful += 1
                size_mb = os.path.getsize(output_path) / 1024 / 1024
                print(f"     ✓ {clean_filename} ({crop_w}x{crop_h}, {size_mb:.1f} MB)")
            else:
                print(f"     ✗ No frames written: {video_file}")
                failed += 1
            
        except Exception as e:
            print(f"     ✗ {video_file}: {str(e)}")
            failed += 1
    
    print(f"  📊 Results: {successful} processed, {skipped} skipped, {failed} failed")
    print(f"  ✅ Post-processing completed")
    return True


def get_bbox_from_frame(frame, threshold=10):
    """Get bounding box from a frame (find non-black regions)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.min(xs), np.min(ys), np.max(xs), np.max(ys)


def crop_and_center_on_canvas(frame, bbox, output_size=512):
    """
    Crop frame using bounding box and center on black canvas.
    Maintains aspect ratio.
    """
    if bbox is None:
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)
    
    x_min, y_min, x_max, y_max = bbox
    cropped = frame[y_min:y_max, x_min:x_max]
    
    if cropped.size == 0:
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)
    
    # Resize to fit in canvas while maintaining aspect ratio
    h, w = cropped.shape[:2]
    scale = min(output_size / h, output_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Center on black canvas
    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    x_offset = (output_size - new_w) // 2
    y_offset = (output_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def postprocess_isolated_objects_black_canvas(output_combined_folder, output_size=512, fps=20):
    """
    Post-process isolated object videos with black canvas centering.
    Creates fixed-size outputs with objects centered on black background.
    """
    objects_isolated_dir = os.path.join(output_combined_folder, "objects_isolated")
    
    if not os.path.exists(objects_isolated_dir):
        print(f"  ⊘ objects_isolated folder not found")
        return False
    
    output_postprocessed_dir = os.path.join(output_combined_folder, "objects_isolated_black_canvas")
    os.makedirs(output_postprocessed_dir, exist_ok=True)
    
    # Find all isolated object videos
    video_files = [f for f in os.listdir(objects_isolated_dir) if 'isolated' in f and f.endswith('.mp4')]
    
    if not video_files:
        print(f"  ⊘ No isolated object videos found in: {objects_isolated_dir}")
        return True
    
    print(f"  🎬 Post-processing {len(video_files)} videos with black canvas (centered)...")
    
    successful = 0
    failed = 0
    
    for video_file in sorted(video_files):
        input_path = os.path.join(objects_isolated_dir, video_file)
        
        # Clean up filename
        clean_filename = video_file.replace('_backward_only', '').replace('_forward_only', '')
        output_path = os.path.join(output_postprocessed_dir, clean_filename)
        
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"     ✗ Could not open: {video_file}")
                failed += 1
                continue
            
            fps_in = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print(f"     ⊘ Empty video: {video_file}")
                cap.release()
                failed += 1
                continue
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps_in, (output_size, output_size))
            
            if not out_writer.isOpened():
                print(f"     ✗ Failed to create writer: {video_file}")
                cap.release()
                failed += 1
                continue
            
            # Process frames
            frame_count = 0
            
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                bbox = get_bbox_from_frame(frame, threshold=10)
                centered_frame = crop_and_center_on_canvas(frame, bbox, output_size)
                out_writer.write(centered_frame)
                frame_count += 1
            
            cap.release()
            out_writer.release()
            
            if frame_count > 0:
                successful += 1
                size_mb = os.path.getsize(output_path) / 1024 / 1024
                print(f"     ✓ {clean_filename} ({output_size}x{output_size}, {size_mb:.1f} MB)")
            else:
                print(f"     ✗ No frames written: {video_file}")
                failed += 1
            
        except Exception as e:
            print(f"     ✗ {video_file}: {str(e)}")
            failed += 1
    
    print(f"  📊 Results: {successful} processed, {failed} failed")
    print(f"  ✅ Black canvas post-processing completed")
    return True


def main():
    """Main bidirectional combined pipeline execution."""
    args = parse_args()
    
    input_folder = os.path.abspath(args.input_folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder_name = os.path.basename(input_folder.rstrip('/'))
    
    if args.processed_folder:
        processed_folder = os.path.abspath(args.processed_folder)
    else:
        processed_folder = os.path.join(script_dir, "processed_bidirectional", input_folder_name)
    
    # Handle output folder structure
    if args.output_folder:
        # When --output-folder is specified, use: output_folder/{input_folder_name}/
        output_folder_parent = os.path.abspath(args.output_folder)
        output_folder_main = os.path.join(output_folder_parent, input_folder_name)
    else:
        # Default: script_dir/output_bidirectional/{input_folder_name}/
        output_folder_main = os.path.join(script_dir, "output_bidirectional", input_folder_name)
    
    output_folder_bwd = os.path.join(output_folder_main, "output_backward")
    output_folder_fwd = os.path.join(output_folder_main, "output_forward")
    output_folder_combined = os.path.join(output_folder_main, "output_combined")
    
    print(f"\n{'='*70}")
    print(f"SAM2 BIDIRECTIONAL COMBINED VIDEO SEGMENTATION PIPELINE")
    print(f"{'='*70}")
    print(f"Input folder: {input_folder}")
    print(f"Processed folder: {processed_folder}")
    print(f"Output folder: {output_folder_main}")
    print(f"{'='*70}\n")
    
    try:
        # Find input files
        video_path, annotation_file = find_input_files(input_folder)
        video_name_base = os.path.splitext(os.path.basename(video_path))[0]
        
        # Find annotated frame
        annotated_frame = find_annotated_frame(annotation_file)
        print(f"✓ Annotated frame: {annotated_frame}")
        print(f"  - Backward will process: frames 1 to {annotated_frame}")
        print(f"  - Forward will process: frames {annotated_frame} onward")
        print(f"  - Combined will have NO DUPLICATION (annotated frame appears once)")
        
        # Track which directions succeeded
        backward_success = False
        forward_success = False
        
        # Run backward pipeline with error handling
        if not args.skip_backward:
            try:
                print(f"\n{'='*70}")
                print(f"Attempting BACKWARD processing...")
                print(f"{'='*70}")
                run_backward_pipeline(input_folder, processed_folder, output_folder_bwd, args)
                backward_success = True
                print(f"✅ Backward processing succeeded")
            except Exception as e:
                print(f"\n⚠️  Backward processing failed: {str(e)}")
                print(f"   Continuing with forward processing...")
                backward_success = False
        
        # Run forward pipeline with error handling
        if not args.skip_forward:
            try:
                print(f"\n{'='*70}")
                print(f"Attempting FORWARD processing...")
                print(f"{'='*70}")
                run_forward_pipeline(input_folder, processed_folder, output_folder_fwd, args)
                forward_success = True
                print(f"✅ Forward processing succeeded")
            except Exception as e:
                print(f"\n⚠️  Forward processing failed: {str(e)}")
                print(f"   Continuing with backward processing...")
                forward_success = False
        
        # Check if at least one direction succeeded
        if not backward_success and not forward_success:
            raise RuntimeError("Both backward and forward processing failed! No output to combine.")
        
        # Combine outputs (handles cases where one side is missing)
        combine_outputs(output_folder_bwd, output_folder_fwd, output_folder_combined, args.fps, video_name_base)
        
        # Post-process isolated objects in combined output
        print(f"\n{'='*70}")
        print(f"POST-PROCESSING ISOLATED OBJECTS")
        print(f"{'='*70}")
        try:
            # Step 1: Tight crop with padding (high-res object focus)
            print(f"\n1️⃣ Tight crop post-processing:")
            postprocess_isolated_objects(output_folder_combined, padding_ratio=0.15)
            
            # Step 2: Black canvas centering (fixed size, centered on black)
            print(f"\n2️⃣ Black canvas post-processing:")
            postprocess_isolated_objects_black_canvas(output_folder_combined, output_size=512, fps=args.fps)
            
            print(f"\n✅ All post-processing completed")
        except Exception as e:
            print(f"⚠️  Post-processing had issues: {str(e)}")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETION SUMMARY")
        print(f"{'='*70}")
        print(f"Backward processing: {'✅ Success' if backward_success else '❌ Failed (skipped)'}")
        print(f"Forward processing:  {'✅ Success' if forward_success else '❌ Failed (skipped)'}")
        print(f"Output generation:   ✅ Success")
        print(f"{'='*70}\n")
        
        print("🎉 Pipeline completed successfully!")
        print(f"\n📂 Final output location: {output_folder_combined}")
        print(f"   ├── overlay_combined.mp4")
        print(f"   ├── overlay_combined_boxes.mp4")
        print(f"   ├── overlay_combined_masks_blended.mp4")
        print(f"   ├── overlay_combined_masks_only.mp4")
        print(f"   ├── objects_cropped/")
        print(f"   ├── objects_isolated/")
        print(f"   ├── objects_isolated_postprocessed/  (High-res tight crops)")
        print(f"   └── objects_isolated_black_canvas/   (512x512 centered on black)")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
