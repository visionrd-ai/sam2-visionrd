#!/usr/bin/env python3
"""
SAM2 Batch Processing Script
Processes all subfolders in a parent directory using process_Sam2_hira_final.py
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch process multiple folders with SAM2 Video Segmentation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-folder',
        type=str,
        required=True,
        help='Parent folder containing multiple input folders to process'
    )
    
    parser.add_argument(
        '--script',
        type=str,
        default='process_bidirectional_combined.py',
        help='Processing script to use (relative to script directory or absolute path)'
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
        '--output-folder',
        type=str,
        default=None,
        help='Custom output folder path (if not specified, uses default location per script)'
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
    
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Only process folders containing this string in their name'
    )
    
    parser.add_argument(
        '--exclude',
        type=str,
        default=None,
        help='Exclude folders containing this string in their name'
    )
    
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue processing other folders if one fails'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List folders that would be processed without actually processing them'
    )
    
    parser.add_argument(
        '--max-folders',
        type=int,
        default=None,
        help='Maximum number of folders to process (for testing)'
    )
    
    return parser.parse_args()


def find_input_folders(data_folder, filter_str=None, exclude_str=None):
    """
    Find all valid input folders in the parent directory.
    A valid folder contains both a video file and a JSON annotation file.
    """
    valid_folders = []
    
    # Check if data_folder exists
    if not os.path.exists(data_folder):
        raise ValueError(f"Data folder does not exist: {data_folder}")
    
    # List all subdirectories
    all_items = os.listdir(data_folder)
    subdirs = [item for item in all_items 
               if os.path.isdir(os.path.join(data_folder, item))]
    
    print(f"\n{'='*70}")
    print(f"SCANNING FOLDERS IN: {data_folder}")
    print(f"{'='*70}")
    print(f"Found {len(subdirs)} subdirectories")
    
    # Check each subdirectory for video and JSON files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(data_folder, subdir)
        
        # Apply filter if specified
        if filter_str and filter_str not in subdir:
            continue
        
        # Apply exclude filter if specified
        if exclude_str and exclude_str in subdir:
            continue
        
        # Check for video and JSON files
        files = os.listdir(subdir_path)
        
        has_video = any(f.endswith(tuple(video_extensions)) for f in files)
        has_json = any(f.endswith('.json') for f in files)
        
        if has_video and has_json:
            valid_folders.append(subdir_path)
            print(f"  ✓ {subdir}")
        else:
            missing = []
            if not has_video:
                missing.append("video")
            if not has_json:
                missing.append("JSON")
            print(f"  ✗ {subdir} (missing: {', '.join(missing)})")
    
    print(f"\n{'='*70}")
    print(f"Found {len(valid_folders)} valid folders to process")
    print(f"{'='*70}\n")
    
    return valid_folders


def process_folder(folder_path, args, script_path):
    """
    Process a single folder using the specified processing script.
    Returns: (success: bool, error_message: str or None)
    """
    folder_name = os.path.basename(folder_path)
    
    print(f"\n{'='*70}")
    print(f"PROCESSING FOLDER: {folder_name}")
    print(f"{'='*70}")
    print(f"Path: {folder_path}")
    print(f"Script: {os.path.basename(script_path)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        script_path,
        '--input-folder', folder_path,
        '--model-config', args.model_config,
        '--checkpoint', args.checkpoint,
        '--fps', str(args.fps),
        '--device', args.device
    ]
    
    # Add output folder if specified
    if args.output_folder:
        cmd.extend(['--output-folder', args.output_folder])
    if args.skip_preprocessing:
        if '--skip-preprocessing' not in str(cmd):  # Check if script supports it
            cmd.append('--skip-preprocessing')
    
    if args.skip_frame_extraction:
        if '--skip-frame-extraction' not in str(cmd):  # Check if script supports it
            cmd.append('--skip-frame-extraction')
    
    # Run the process
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCESSFULLY COMPLETED: {folder_name}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        return True, None
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Process exited with code {e.returncode}"
        
        print(f"\n{'='*70}")
        print(f"❌ FAILED: {folder_name}")
        print(f"Error: {error_msg}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        return False, error_msg
        
    except Exception as e:
        error_msg = str(e)
        
        print(f"\n{'='*70}")
        print(f"❌ FAILED: {folder_name}")
        print(f"Error: {error_msg}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        return False, error_msg


def save_summary(results, output_file):
    """Save processing summary to JSON file."""
    summary = {
        "processing_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_folders": len(results),
        "successful": sum(1 for r in results if r['success']),
        "failed": sum(1 for r in results if not r['success']),
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Summary saved to: {output_file}")


def main():
    """Main batch processing execution."""
    args = parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve processing script path (relative to script_dir or absolute)
    if os.path.isabs(args.script):
        # Absolute path provided
        processing_script = args.script
    else:
        # Relative to script directory
        processing_script = os.path.join(script_dir, args.script)
    
    # Check if processing script exists
    if not os.path.exists(processing_script):
        print(f"❌ Error: Processing script not found: {processing_script}")
        print(f"   Tried: {processing_script}")
        sys.exit(1)
    
    # Get absolute path to data folder
    data_folder = os.path.abspath(args.data_folder)
    
    print(f"\n{'='*70}")
    print(f"SAM2 BATCH PROCESSING")
    print(f"{'='*70}")
    print(f"Data folder: {data_folder}")
    print(f"Processing script: {processing_script}")
    print(f"Script name: {os.path.basename(processing_script)}")
    print(f"Model config: {args.model_config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"FPS: {args.fps}")
    if args.output_folder:
        print(f"Output folder: {args.output_folder}")
    if args.filter:
        print(f"Filter: {args.filter}")
    if args.exclude:
        print(f"Exclude: {args.exclude}")
    if args.max_folders:
        print(f"Max folders: {args.max_folders}")
    print(f"{'='*70}")
    
    # Find all valid input folders
    try:
        folders_to_process = find_input_folders(
            data_folder, 
            filter_str=args.filter,
            exclude_str=args.exclude
        )
    except Exception as e:
        print(f"❌ Error finding input folders: {e}")
        sys.exit(1)
    
    if not folders_to_process:
        print("❌ No valid folders found to process!")
        sys.exit(1)
    
    # Limit number of folders if specified
    if args.max_folders:
        folders_to_process = folders_to_process[:args.max_folders]
        print(f"📊 Limited to first {args.max_folders} folders\n")
    
    # Dry run mode
    if args.dry_run:
        print(f"\n{'='*70}")
        print(f"DRY RUN MODE - Would process these folders:")
        print(f"{'='*70}")
        for i, folder in enumerate(folders_to_process, 1):
            print(f"  {i}. {os.path.basename(folder)}")
        print(f"{'='*70}\n")
        print("Run without --dry-run to actually process these folders.")
        sys.exit(0)
    
    # Process each folder
    results = []
    start_time = datetime.now()
    
    for i, folder_path in enumerate(folders_to_process, 1):
        folder_name = os.path.basename(folder_path)
        
        print(f"\n{'='*70}")
        print(f"PROGRESS: Processing folder {i}/{len(folders_to_process)}")
        print(f"{'='*70}\n")
        
        success, error = process_folder(folder_path, args, processing_script)
        
        results.append({
            'folder': folder_name,
            'path': folder_path,
            'success': success,
            'error': error,
            'index': i
        })
        
        # Stop if error and not continuing on error
        if not success and not args.continue_on_error:
            print(f"\n⚠️  Stopping batch processing due to error in folder: {folder_name}")
            print(f"   Use --continue-on-error to process remaining folders despite errors.")
            break
    
    # Calculate total time
    end_time = datetime.now()
    total_time = end_time - start_time
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time}")
    print(f"\nFolders processed: {len(results)}/{len(folders_to_process)}")
    print(f"✅ Successful: {sum(1 for r in results if r['success'])}")
    print(f"❌ Failed: {sum(1 for r in results if not r['success'])}")
    
    # List failed folders
    failed_folders = [r for r in results if not r['success']]
    if failed_folders:
        print(f"\n{'='*70}")
        print(f"FAILED FOLDERS:")
        print(f"{'='*70}")
        for result in failed_folders:
            print(f"  ❌ {result['folder']}")
            if result['error']:
                print(f"     Error: {result['error']}")
    
    # List successful folders
    successful_folders = [r for r in results if r['success']]
    if successful_folders:
        print(f"\n{'='*70}")
        print(f"SUCCESSFUL FOLDERS:")
        print(f"{'='*70}")
        for result in successful_folders:
            print(f"  ✅ {result['folder']}")
    
    print(f"{'='*70}\n")
    
    # Save summary to file
    summary_file = os.path.join(script_dir, f"batch_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_summary(results, summary_file)
    
    # Exit with appropriate code
    if any(not r['success'] for r in results):
        sys.exit(1)
    else:
        print("\n🎉 All folders processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
