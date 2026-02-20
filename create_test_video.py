#!/usr/bin/env python3
"""
Create a test video around annotated frame: 30 before + annotated frame + 29 after = 60 frames total.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create test video around annotated frame',
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
        help='Output folder for test video (default: input_folder/test_video)'
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
    """Find first annotated frame number from JSON."""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    first_annotated_frame_no = None
    num_boxes = 0
    total_frames = len(data.get("frames", []))
    
    for frame_data in data["frames"]:
        boxes = frame_data.get("annotation", {}).get("boxes", [])
        if len(boxes) > 0:
            first_annotated_frame_no = frame_data["frameNo"]
            num_boxes = len(boxes)
            break
    
    if first_annotated_frame_no is None:
        raise ValueError("No annotated frames found in JSON!")
    
    print(f"\n{'='*70}")
    print(f"ANNOTATION ANALYSIS")
    print(f"{'='*70}")
    print(f"✓ First annotated frame: frameNo {first_annotated_frame_no}")
    print(f"  Number of objects: {num_boxes}")
    print(f"  Total frames in JSON: {total_frames}")
    
    return first_annotated_frame_no, num_boxes


def create_test_video(video_path, annotation_file, output_folder):
    """
    Create test video with 60 frames:
    - 30 frames BEFORE annotated frame
    - 1 annotated frame
    - 29 frames AFTER annotated frame
    """
    print(f"\n{'='*70}")
    print(f"CREATING TEST VIDEO")
    print(f"{'='*70}")
    
    # Find annotated frame
    annotated_frame_no, num_boxes = find_annotated_frame(annotation_file)
    
    # Convert to 0-indexed
    annotated_frame_idx = annotated_frame_no - 1
    
    # Calculate frame range
    start_idx = max(0, annotated_frame_idx - 30)  # 30 frames before (or from start if less available)
    end_idx = annotated_frame_idx + 29  # 29 frames after
    
    frames_before_available = annotated_frame_idx - start_idx
    
    print(f"\n📹 Frame Range Calculation:")
    print(f"  Annotated frame (1-indexed): {annotated_frame_no}")
    print(f"  Annotated frame (0-indexed): {annotated_frame_idx}")
    print(f"  Frames before available: {frames_before_available} (requested: 30)")
    print(f"  Start frame (0-indexed): {start_idx}")
    print(f"  End frame (0-indexed): {end_idx}")
    print(f"  Total frames to extract: {end_idx - start_idx + 1}")
    
    # Open source video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📺 Source Video Properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Create test video
    video_name_base = os.path.splitext(os.path.basename(video_path))[0]
    test_video_path = os.path.join(output_folder, f"{video_name_base}_test_60frames.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))
    
    print(f"\n⏳ Extracting frames {start_idx} to {end_idx}...")
    
    frame_count = 0
    for frame_idx in range(start_idx, min(end_idx + 1, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"  ⚠ Failed to read frame {frame_idx}")
            break
        
        # Add marker for annotated frame
        if frame_idx == annotated_frame_idx:
            # Add red border to annotated frame
            cv2.rectangle(frame, (5, 5), (width-5, height-5), (0, 0, 255), 10)
            cv2.putText(frame, "ANNOTATED FRAME", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        out_writer.write(frame)
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"  Progress: {frame_count}/{min(end_idx + 1 - start_idx, total_frames - start_idx)} frames", end='\r')
    
    cap.release()
    out_writer.release()
    
    print(f"\n✓ Test video created: {frame_count} frames")
    print(f"  Output: {test_video_path}")
    
    # Create test JSON
    print(f"\n📝 Creating test JSON...")
    
    with open(annotation_file, 'r') as f:
        orig_data = json.load(f)
    
    # Extract relevant frames from original JSON
    test_frames = []
    frame_count_in_json = 1  # Start counting from 1
    
    for frame_data in orig_data.get("frames", []):
        orig_frame_no = frame_data["frameNo"]
        
        if start_idx < orig_frame_no <= end_idx + 1:
            new_frame_data = frame_data.copy()
            new_frame_data["frameNo"] = frame_count_in_json
            new_frame_data["originalFrameNo"] = orig_frame_no
            new_frame_data["isAnnotatedFrame"] = (orig_frame_no == annotated_frame_no)
            test_frames.append(new_frame_data)
            frame_count_in_json += 1
    
    test_json = {
        "taskId": orig_data.get("taskId", ""),
        "campaignId": orig_data.get("campaignId", ""),
        "videoName": f"{video_name_base}_test_60frames",
        "metadata": {
            "fps": fps,
            "totalFrames": frame_count,
            "createdDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sourceVideo": os.path.basename(video_path),
            "sourceJSON": os.path.basename(annotation_file),
            "annotatedFrameOriginal": annotated_frame_no,
            "frameRangeOriginal": f"{start_idx}-{end_idx}",
            "description": f"Test video: 30 frames before + annotated frame + 29 frames after"
        },
        "frames": test_frames
    }
    
    test_json_path = os.path.join(output_folder, f"{video_name_base}_test_60frames.json")
    with open(test_json_path, 'w') as f:
        json.dump(test_json, f, indent=2)
    
    print(f"✓ Test JSON created: {len(test_frames)} frames")
    print(f"  Output: {test_json_path}")
    
    # Summary report
    print(f"\n{'='*70}")
    print(f"✅ TEST VIDEO CREATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"  Total frames extracted: {frame_count}")
    print(f"  Annotated frame position: frame {annotated_frame_no} (1-indexed)")
    print(f"  In test video: frame 1-{frame_count}")
    print(f"  Annotated frame in test: frame {frames_before_available + 1}")
    print(f"  Objects in annotated frame: {num_boxes}")
    print(f"\n📁 Output files:")
    print(f"  Video: {test_video_path}")
    print(f"  JSON:  {test_json_path}")
    print(f"\n💡 Test video is ready for validation!")
    print(f"   The annotated frame is marked with RED BORDER")
    print(f"{'='*70}\n")
    
    return test_video_path, test_json_path, frame_count, annotated_frame_no


def main():
    """Main execution."""
    args = parse_args()
    
    input_folder = os.path.abspath(args.input_folder)
    
    if args.output_folder:
        output_folder = os.path.abspath(args.output_folder)
    else:
        output_folder = os.path.join(input_folder, "test_video")
    
    print(f"\n{'='*70}")
    print(f"TEST VIDEO CREATOR")
    print(f"{'='*70}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*70}\n")
    
    try:
        # Find input files
        video_path, annotation_file = find_input_files(input_folder)
        
        # Create test video
        test_video_path, test_json_path, frame_count, annotated_frame_no = create_test_video(
            video_path, annotation_file, output_folder
        )
        
        print("\n✨ Test video is ready! You can now run the bidirectional pipeline on this test case.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
