#!/usr/bin/env python3

"""
Script to scan directory for .tar files and unpack them into organized structure:
audio/language/emotion/
"""
from typing import Tuple, Any
from concurrent import futures
import glob
import json
import os
import subprocess
import sys
import tarfile
import tempfile


def get_emotion_from_filename(tar_filename):
    """Extract the 4th word (emotion) from the tar filename."""
    basename = os.path.splitext(tar_filename)[0]
    parts = basename.split('_')
    if len(parts) >= 4:
        return parts[3]  # 4th word (0-indexed: 3)
    return "unknown"


def get_language_from_filename(tar_filename):
    """Extract the language (1st word) from the tar filename."""
    basename = os.path.splitext(tar_filename)[0]
    parts = basename.split('_')
    if len(parts) >= 1:
        return parts[0]
    return "unknown"


def convert_to_wav_16khz(input_path, output_path):
    """Convert audio file to 16kHz WAV using ffmpeg."""
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',      # mono
        '-y',            # overwrite output
        '-threads', '1',
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e.stderr.decode()}")
        return False


def get_transcript_from_json(json_path):
    """Extract transcript from the JSON metadata file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Try flash_transcription first, then look in annotation
            if 'flash_transcription' in data:
                return data['flash_transcription']
            elif 'annotation' in data:
                # Parse the XML-like annotation to get transcription
                annotation = data['annotation']
                start = annotation.find('<transcription_start>')
                end = annotation.find('</transcription_start>')
                if start != -1 and end != -1:
                    return annotation[start + len('<transcription_start>'):end]
            return ""
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return ""


def process_tar_file(args: Tuple[Any]):
    """Process a single tar file and organize its contents."""
    tar_path, output_base_dir = args
    tar_filename = os.path.basename(tar_path)
    language = get_language_from_filename(tar_filename)
    emotion = get_emotion_from_filename(tar_filename)
    
    print(f"Processing: {tar_filename}")
    print(f"  Language: {language}, Emotion: {emotion}")
    
    # Create output directory structure
    output_dir = os.path.join(output_base_dir, 'audio', language, emotion)
    os.makedirs(output_dir, exist_ok=True)
    
    manifest_entries = []
    
    # Create a temporary directory to extract files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract tar file
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(temp_dir)
        
        # Find all audio files (mp3, wav, etc.)
        audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
        
        # Walk through extracted files
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_lower = file.lower()
                file_ext = os.path.splitext(file_lower)[1]
                
                if file_ext in audio_extensions:
                    audio_path = os.path.join(root, file)
                    base_name = os.path.splitext(file)[0]
                    
                    # Look for corresponding JSON file
                    json_path = os.path.join(root, base_name + '.json')
                    transcript = ""
                    if os.path.exists(json_path):
                        transcript = get_transcript_from_json(json_path)
                    
                    # Output WAV filename (preserve original base name)
                    wav_filename = base_name + '.wav'
                    wav_output_path = os.path.join(output_dir, wav_filename)
                    
                    # Convert to 16kHz WAV
                    if convert_to_wav_16khz(audio_path, wav_output_path):
                        # Create relative path for manifest
                        relative_path = f"audio/{language}/{emotion}/{wav_filename}"
                        
                        manifest_entry = {
                            "file": relative_path,
                            "transcript": transcript,
                            "label": emotion,
                            "language": language
                        }
                        manifest_entries.append(manifest_entry)
                        print(f"  Converted: {file} -> {wav_filename}")
    
    # Write manifest.jsonl
    manifest_path = os.path.join(output_dir, 'manifest.jsonl')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"  Created manifest with {len(manifest_entries)} entries: {manifest_path}")
    return len(manifest_entries)


def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Scanning directory: {script_dir}")
    print("-" * 50)
    
    # Find all .tar files in the directory
    tar_files = glob.glob(os.path.join(script_dir, "*.tar"))
    
    if not tar_files:
        sys.exit("No .tar files found in the directory.")
    
    print(f"Found {len(tar_files)} tar file(s)")
    print("-" * 50)
    
    with futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        total_files = sum(executor.map(process_tar_file, ((p, script_dir) for p in tar_files)))
         
    print("-" * 50)
    print(f"Processing complete! Total audio files converted: {total_files}")


if __name__ == "__main__":
    main()
