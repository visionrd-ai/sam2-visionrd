#!/usr/bin/env python3
"""
Post-processing script for object isolated videos.
Extracts tight bounding box of objects and creates high-quality cropped videos.
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime


def find_object_bounding_box(frame):
    """
    Find tight bounding box of the object (non-black pixels).
    
    Returns:
        (x1, y1, x2, y2) or None if no object found
    """
    # Convert to grayscale if color
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Find non-black pixels
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get bounding rectangle of all contours
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
    """
    Analyze video to find consistent bounding box of object.
    Samples every N frames to get average/stable bounding box.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    bboxes = []
    frame_idx = 0
    sample_idx = 0
    
    print(f"  Analyzing {total_frames} frames (sampling every {sample_rate} frames)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_idx % sample_rate == 0:
            bbox = find_object_bounding_box(frame)
            if bbox:
                bboxes.append(bbox)
            sample_idx += 1
        
        frame_idx += 1
    
    cap.release()
    
    if not bboxes:
        return None
    
    # Calculate average bounding box
    x_mins = [b[0] for b in bboxes]
    y_mins = [b[1] for b in bboxes]
    x_maxs = [b[2] for b in bboxes]
    y_maxs = [b[3] for b in bboxes]
    
    # Use min/max to ensure all objects fit (conservative approach)
    avg_x_min = np.percentile(x_mins, 25)  # 25th percentile (more conservative)
    avg_y_min = np.percentile(y_mins, 25)
    avg_x_max = np.percentile(x_maxs, 75)  # 75th percentile
    avg_y_max = np.percentile(y_maxs, 75)
    
    return int(avg_x_min), int(avg_y_min), int(avg_x_max), int(avg_y_max)


def add_padding(bbox, frame_h, frame_w, padding_ratio=0.15):
    """
    Add padding around bounding box while keeping it within frame bounds.
    
    Args:
        bbox: (x1, y1, x2, y2)
        frame_h, frame_w: Frame dimensions
        padding_ratio: Padding as ratio of bbox size (0.15 = 15%)
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # Calculate padding
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    
    # Apply padding with bounds check
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)
    
    return x1, y1, x2, y2


def make_square_bbox(bbox, frame_h, frame_w):
    """
    Convert bbox to square while keeping object centered.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # Use larger dimension
    side = max(w, h)
    
    # Center the box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # New corners
    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(frame_w, x1 + side)
    y2 = min(frame_h, y1 + side)
    
    # Adjust if went out of bounds
    if x2 - x1 < side:
        x1 = max(0, x2 - side)
    if y2 - y1 < side:
        y1 = max(0, y2 - side)
    
    return x1, y1, x2, y2


def postprocess_video(input_video_path, output_video_path, target_size=None, square=False, padding_ratio=0.15):
    """
    Post-process isolated object video:
    1. Find tight bounding box
    2. Add padding
    3. Crop frames
    4. Optionally resize to target size
    5. Save to output video
    
    Args:
        input_video_path: Path to objects_isolated video
        output_video_path: Path to save post-processed video
        target_size: Optional (width, height) to resize cropped video
        square: Whether to make bounding box square
        padding_ratio: Padding as ratio of bbox size
    """
    print(f"  Processing: {os.path.basename(input_video_path)}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"    ✗ Failed to open video")
        return False
    
    # Get video properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"    Input: {frame_w}x{frame_h}, {total_frames} frames, {fps:.1f} fps")
    
    # Find average bounding box (analyzing the video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    bbox = get_average_bounding_box(input_video_path, sample_rate=max(1, total_frames // 50))
    
    if bbox is None:
        print(f"    ✗ No object found in video")
        cap.release()
        return False
    
    print(f"    Detected bbox: {bbox}")
    
    # Add padding
    bbox = add_padding(bbox, frame_h, frame_w, padding_ratio)
    print(f"    After padding: {bbox}")
    
    # Make square if requested
    if square:
        bbox = make_square_bbox(bbox, frame_h, frame_w)
        print(f"    After squaring: {bbox}")
    
    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1
    
    # Determine output size
    if target_size:
        out_w, out_h = target_size
    else:
        out_w, out_h = crop_w, crop_h
    
    print(f"    Output: {out_w}x{out_h}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
    
    if not out_writer.isOpened():
        print(f"    ✗ Failed to create output video writer")
        cap.release()
        return False
    
    # Process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop frame
        cropped = frame[y1:y2, x1:x2]
        
        # Resize if needed
        if target_size and (crop_w != out_w or crop_h != out_h):
            cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        
        # Write frame
        out_writer.write(cropped)
        frame_count += 1
        
        if (frame_idx + 1) % 50 == 0:
            print(f"    Progress: {frame_idx + 1}/{total_frames} frames", end='\r')
    
    print(f"    ✓ Wrote {frame_count} frames")
    
    cap.release()
    out_writer.release()
    
    return True


def process_folder(folder_path, target_size=None, square=False, padding_ratio=0.15):
    """
    Post-process all objects_isolated videos in a folder.
    
    Args:
        folder_path: Path to output folder (contains output_backward, output_forward, output_combined)
        target_size: Optional (width, height) to resize cropped videos
        square: Whether to make bounding box square
        padding_ratio: Padding as ratio of bbox size
    """
    print(f"\n{'='*70}")
    print(f"POST-PROCESSING: {os.path.basename(folder_path)}")
    print(f"{'='*70}")
    
    # Find all objects_isolated folders
    directions = ['output_backward', 'output_forward', 'output_combined']
    total_videos = 0
    successful = 0
    
    for direction in directions:
        objects_isolated_dir = os.path.join(folder_path, direction, 'objects_isolated')
        
        if not os.path.exists(objects_isolated_dir):
            print(f"\n⊘ {direction}/objects_isolated/ not found")
            continue
        
        print(f"\n📁 Processing {direction}/objects_isolated/...")
        
        # Create post-processed output folder
        output_dir = os.path.join(folder_path, direction, 'objects_isolated_postprocessed')
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all isolated object videos
        video_files = [f for f in os.listdir(objects_isolated_dir) if f.endswith('_isolated.mp4')]
        
        if not video_files:
            print(f"  No object videos found")
            continue
        
        print(f"  Found {len(video_files)} object videos")
        
        # Process each video
        for video_file in sorted(video_files):
            input_path = os.path.join(objects_isolated_dir, video_file)
            
            # Create output path with same name
            output_path = os.path.join(output_dir, video_file)
            
            total_videos += 1
            
            try:
                success = postprocess_video(
                    input_path, output_path,
                    target_size=target_size,
                    square=square,
                    padding_ratio=padding_ratio
                )
                
                if success:
                    successful += 1
                    size_mb = os.path.getsize(output_path) / 1024 / 1024
                    print(f"    ✓ Output: {size_mb:.1f} MB")
                else:
                    print(f"    ✗ Failed to process")
            
            except Exception as e:
                print(f"    ✗ Error: {str(e)}")
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {successful}/{total_videos} videos processed successfully")
    print(f"Output folders created with '_postprocessed' suffix")
    print(f"{'='*70}\n")
    
    return successful == total_videos


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Post-process isolated object videos for better viewing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help='Output folder path (contains output_backward, output_forward, output_combined)'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        nargs=2,
        default=None,
        metavar=('WIDTH', 'HEIGHT'),
        help='Target resolution for cropped videos (e.g., 512 512)'
    )
    
    parser.add_argument(
        '--square',
        action='store_true',
        help='Make bounding box square (equal width and height)'
    )
    
    parser.add_argument(
        '--padding-ratio',
        type=float,
        default=0.15,
        help='Padding around object as ratio of object size (0.15 = 15%)'
    )
    
    parser.add_argument(
        '--folder-name',
        type=str,
        default=None,
        help='Process specific folder (if not set, processes all subfolders)'
    )
    
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    
    output_folder = os.path.abspath(args.output_folder)
    
    if not os.path.exists(output_folder):
        print(f"❌ Output folder not found: {output_folder}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"OBJECT VIDEO POST-PROCESSING PIPELINE")
    print(f"{'='*70}")
    print(f"Output folder: {output_folder}")
    if args.target_size:
        print(f"Target size: {args.target_size[0]}x{args.target_size[1]}")
    if args.square:
        print(f"Square bbox: Yes")
    print(f"Padding ratio: {args.padding_ratio:.0%}")
    print(f"{'='*70}")
    
    # If specific folder name provided, process only that
    if args.folder_name:
        folder_path = os.path.join(output_folder, args.folder_name)
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            sys.exit(1)
        
        success = process_folder(
            folder_path,
            target_size=tuple(args.target_size) if args.target_size else None,
            square=args.square,
            padding_ratio=args.padding_ratio
        )
        
        sys.exit(0 if success else 1)
    
    # Otherwise, process all subfolders
    subfolders = [d for d in os.listdir(output_folder) 
                  if os.path.isdir(os.path.join(output_folder, d)) 
                  and d.startswith('output_') == False]
    
    if not subfolders:
        print(f"⚠️  No subfolders found in {output_folder}")
        print(f"   Expected structure: {output_folder}/[folder_name]/output_backward/...")
        sys.exit(1)
    
    print(f"\nFound {len(subfolders)} folders to process:")
    for sf in sorted(subfolders)[:5]:
        print(f"  - {sf}")
    if len(subfolders) > 5:
        print(f"  ... and {len(subfolders) - 5} more")
    
    # Process each folder
    all_success = True
    for subfolder in sorted(subfolders):
        folder_path = os.path.join(output_folder, subfolder)
        success = process_folder(
            folder_path,
            target_size=tuple(args.target_size) if args.target_size else None,
            square=args.square,
            padding_ratio=args.padding_ratio
        )
        
        if not success:
            all_success = False
    
    print(f"\n{'='*70}")
    print(f"✅ POST-PROCESSING COMPLETE" if all_success else f"⚠️  COMPLETED WITH ERRORS")
    print(f"{'='*70}\n")
    
    print("📂 Post-processed videos saved to:")
    print(f"   - {output_folder}/[folder]/output_backward/objects_isolated_postprocessed/")
    print(f"   - {output_folder}/[folder]/output_forward/objects_isolated_postprocessed/")
    print(f"   - {output_folder}/[folder]/output_combined/objects_isolated_postprocessed/")
    
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
