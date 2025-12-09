"""
MELD Dataset Processor
"""

import csv
import json
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Configuration
BASE_DIR = Path(__file__).parent.resolve()

SPLITS = {
    'eval': {
        'csv': 'dev_sent_emo.csv',
        'source_folder': 'dev_splits_complete',
    },
    'test': {
        'csv': 'test_sent_emo.csv',
        'source_folder': 'output_repeated_splits_test',
    },
    'train': {
        'csv': 'train_sent_emo.csv',
        'source_folder': 'train_splits',
    },
}

OUTPUT_DIR = BASE_DIR / 'output'


def read_csv_data(csv_path: Path) -> list[dict]:
    """Read CSV file and return list of dictionaries."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def get_video_filename(dialogue_id: str, utterance_id: str) -> str:
    """Generate video filename from dialogue and utterance IDs."""
    return f"dia{dialogue_id}_utt{utterance_id}.mp4"


def extract_audio_ffmpeg(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video using ffmpeg, converting to 16kHz mono MP3."""
    try:
        # Create parent directory if it doesn't exist
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'libmp3lame',  # MP3 codec
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-q:a', '2',  # Quality setting
            '-y',  # Overwrite output
            str(audio_path)
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Timeout extracting audio from {video_path}")
        return False
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False


def process_split(split_name: str, config: dict, max_workers: int = 8) -> None:
    """Process a single split (eval, test, or train)."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"{'='*60}")
    
    csv_path = BASE_DIR / config['csv']
    source_folder = BASE_DIR / config['source_folder']
    output_folder = OUTPUT_DIR / split_name
    audio_folder = output_folder / 'audio'
    meta_path = output_folder / 'meta.jsonl'
    
    # Create output directories
    audio_folder.mkdir(parents=True, exist_ok=True)
    
    # Read CSV data
    print(f"Reading CSV: {csv_path}")
    csv_data = read_csv_data(csv_path)
    print(f"Found {len(csv_data)} entries in CSV")
    
    # Prepare processing tasks
    tasks = []
    metadata_entries = []
    
    for idx, row in enumerate(csv_data):
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        video_filename = get_video_filename(dialogue_id, utterance_id)
        video_path = source_folder / video_filename
        
        # Generate numbered audio filename (e.g., 00000.mp3)
        audio_filename = f"{idx:05d}.mp3"
        audio_path = audio_folder / audio_filename
        
        # Create metadata entry
        metadata = {
            'id': f"{idx:05d}",
            'audio': f"audio/{audio_filename}",
            'Utterance': row['Utterance'],
            'Emotion': row['Emotion'],
            'Dialogue_ID': dialogue_id,
            'Utterance_ID': utterance_id,
        }
        
        if video_path.exists():
            tasks.append((video_path, audio_path, idx))
            metadata_entries.append(metadata)
        else:
            print(f"Warning: Video not found: {video_path}")
    
    print(f"Found {len(tasks)} valid video files to process")
    
    # Process videos with parallel extraction
    successful = 0
    failed = 0
    failed_indices = set()
    
    print(f"Extracting audio files (using {max_workers} workers)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_audio_ffmpeg, video_path, audio_path): (video_path, audio_path, idx)
            for video_path, audio_path, idx in tasks
        }
        
        with tqdm(total=len(futures), desc=f"Processing {split_name}") as pbar:
            for future in as_completed(futures):
                video_path, audio_path, idx = futures[future]
                try:
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                        failed_indices.add(idx)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    failed += 1
                    failed_indices.add(idx)
                pbar.update(1)
    
    # Filter metadata to only include successful extractions
    final_metadata = [m for i, m in enumerate(metadata_entries) if i not in failed_indices]
    
    # Write metadata file
    print(f"Writing metadata to {meta_path}")
    with open(meta_path, 'w', encoding='utf-8') as f:
        for entry in final_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n{split_name} Summary:")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Metadata entries: {len(final_metadata)}")


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def main():
    print("MELD Dataset Processor")
    print("="*60)
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg: brew install ffmpeg")
        return
    
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name, config in SPLITS.items():
        process_split(split_name, config)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)

if __name__ == '__main__':
    main()