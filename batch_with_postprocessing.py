#!/usr/bin/env python3
"""
Extended batch processing script with automatic post-processing.
Processes folders and then applies post-processing to isolated objects.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def run_main_processing(data_folder, output_folder, script, args_dict):
    """Run the main processing script."""
    print(f"\n{'='*70}")
    print(f"STEP 1: RUNNING MAIN PROCESSING")
    print(f"{'='*70}")
    
    # Build command
    cmd = [
        sys.executable, script,
        '--data-folder', data_folder,
    ]
    
    # Add all additional arguments
    for key, value in args_dict.items():
        if key in ['data_folder', 'output_folder', 'script', 'script_path']:
            continue
        
        if value is None:
            continue
        
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key.replace("_", "-")}')
        elif isinstance(value, list):
            cmd.extend([f'--{key.replace("_", "-")}'] + [str(v) for v in value])
        else:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    if output_folder:
        cmd.extend(['--output-folder', output_folder])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def run_postprocessing(output_folder, folder_name=None, target_size=None, square=False, padding_ratio=0.15):
    """Run post-processing on the output folder."""
    print(f"\n{'='*70}")
    print(f"STEP 2: RUNNING POST-PROCESSING")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable, 'postprocess_objects.py',
        '--output-folder', output_folder,
        '--padding-ratio', str(padding_ratio),
    ]
    
    if target_size:
        cmd.extend(['--target-size'] + [str(s) for s in target_size])
    
    if square:
        cmd.append('--square')
    
    if folder_name:
        cmd.extend(['--folder-name', folder_name])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch process folders with automatic post-processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main processing arguments
    parser.add_argument(
        '--data-folder',
        type=str,
        required=True,
        help='Folder containing video subfolders to process'
    )
    
    parser.add_argument(
        '--output-folder',
        type=str,
        default=None,
        help='Base output folder (optional, defaults based on script)'
    )
    
    parser.add_argument(
        '--script',
        type=str,
        default='process_bidirectional_combined.py',
        help='Processing script to use'
    )
    
    # SAM2 arguments
    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/sam2.1/sam2.1_hiera_t.yaml',
        help='SAM2 model config path'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/sam2.1_hiera_tiny.pt',
        help='SAM2 checkpoint path'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to run on'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Output video FPS'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue processing even if a video fails'
    )
    
    # Post-processing arguments
    parser.add_argument(
        '--skip-postprocessing',
        action='store_true',
        help='Skip post-processing step'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        nargs=2,
        default=None,
        metavar=('WIDTH', 'HEIGHT'),
        help='Target resolution for post-processed videos (e.g., 512 512)'
    )
    
    parser.add_argument(
        '--square-bbox',
        action='store_true',
        help='Make bounding boxes square in post-processing'
    )
    
    parser.add_argument(
        '--padding-ratio',
        type=float,
        default=0.15,
        help='Padding ratio for post-processed crops (0.15 = 15%)'
    )
    
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING WITH POST-PROCESSING")
    print(f"{'='*70}")
    print(f"Data folder: {args.data_folder}")
    print(f"Output folder: {args.output_folder or '(default)'}")
    print(f"Processing script: {args.script}")
    if not args.skip_postprocessing:
        print(f"Post-processing: ENABLED")
        if args.target_size:
            print(f"  - Target size: {args.target_size[0]}x{args.target_size[1]}")
        if args.square_bbox:
            print(f"  - Square bbox: YES")
        print(f"  - Padding ratio: {args.padding_ratio:.0%}")
    else:
        print(f"Post-processing: DISABLED")
    print(f"{'='*70}\n")
    
    # Build arguments for main processing
    args_dict = vars(args)
    
    # Run main processing
    main_success = run_main_processing(
        args.data_folder,
        args.output_folder,
        args.script,
        args_dict
    )
    
    if not main_success:
        print("\n❌ Main processing failed")
        sys.exit(1)
    
    # Run post-processing if not skipped
    if args.skip_postprocessing:
        print(f"\n⊘ Post-processing skipped")
        sys.exit(0)
    
    # Determine output folder for post-processing
    if args.output_folder:
        postprocess_folder = args.output_folder
    else:
        # Use default path from the script
        postprocess_folder = os.path.join(
            os.path.dirname(args.script),
            'output_bidirectional'
        )
    
    postprocess_success = run_postprocessing(
        postprocess_folder,
        target_size=args.target_size,
        square=args.square_bbox,
        padding_ratio=args.padding_ratio
    )
    
    if not postprocess_success:
        print("\n⚠️  Post-processing had errors")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"✅ BATCH PROCESSING AND POST-PROCESSING COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
