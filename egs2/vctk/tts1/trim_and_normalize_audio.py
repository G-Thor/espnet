#!/usr/bin/env python3
"""
Audio Processor Script

This script processes audio files in a directory by:
1. Trimming leading and trailing silence
2. Normalizing the amplitude to a target level
3. Saving the processed files to a new directory

Usage:
    python trim_and_normalize_audio.py --input_dir /path/to/input --output_dir /path/to/output [--threshold -40] [--target_level -3]

Requirements:
    pip install librosa soundfile numpy tqdm argparse
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

def trim_silence(audio, sr, threshold_db=-40, frame_length=2048, hop_length=512):
    """
    Trim leading and trailing silence from an audio file.
    
    Parameters:
    - audio: Audio time series
    - sr: Sample rate
    - threshold_db: Threshold (in decibels) below reference to consider as silence
    - frame_length: Length of each frame in samples
    - hop_length: Number of samples between successive frames
    
    Returns:
    - trimmed_audio: Audio with silence trimmed
    """
    
    # Get indices of non-silent frames
    non_silent = librosa.effects.split(audio, top_db=-threshold_db, frame_length=frame_length, hop_length=hop_length)
    
    # Return empty array if no non-silent frames found
    if len(non_silent) == 0:
        return np.array([])
    
    # Get start and end indices
    start, end = non_silent[0][0], non_silent[-1][1]

    # Return the trimmed audio + 50ms of silence on each side 
    return audio[max(0,start-sr//5):min(len(audio),end+sr//5)]

def normalize_audio(audio, target_level_db=-3.0):
    """
    Normalize the amplitude of an audio file to a target level.
    
    Parameters:
    - audio: Audio time series
    - target_level_db: Target level in dB
    
    Returns:
    - normalized_audio: Normalized audio
    """
    # Handle empty audio
    if len(audio) == 0 or np.all(audio == 0):
        return audio
    
    # Calculate current peak amplitude
    max_amplitude = np.max(np.abs(audio))
    
    # Avoid division by zero
    if max_amplitude == 0:
        return audio
    
    # Calculate target peak amplitude
    target_amplitude = 10 ** (target_level_db / 20.0)
    
    # Calculate gain to apply
    gain = target_amplitude / max_amplitude
    
    # Apply gain
    normalized_audio = audio * gain
    
    return normalized_audio

def process_audio(input_path, output_path, trim_threshold_db=-40, target_level_db=-3):
    """
    Process an audio file by trimming silence and normalizing amplitude.
    
    Parameters:
    - input_path: Path to input audio file
    - output_path: Path to save processed audio file
    - trim_threshold_db: Threshold for silence detection in dB
    - target_level_db: Target level for normalization in dB
    """
    try:
        # Load audio file
        audio, sr = librosa.load(input_path, sr=None, mono=True)
        
        # Trim silence
        audio_trimmed = trim_silence(audio, sr, threshold_db=trim_threshold_db)
        
        # Normalize amplitude
        audio_normalized = normalize_audio(audio_trimmed, target_level_db=target_level_db)
        
        # Save processed audio
        sf.write(output_path, audio_normalized, sr)
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process audio files by trimming silence and normalizing amplitude')
    parser.add_argument('--input_dir', required=True, help='Directory containing input audio files')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed audio files')
    parser.add_argument('--threshold', type=float, default=-40, help='Threshold (in dB) for silence detection (default: -40)')
    parser.add_argument('--target_level', type=float, default=-3, help='Target level (in dB) for normalization (default: -3)')
    parser.add_argument('--extensions', default='.wav,.mp3,.flac', help='Comma-separated list of audio file extensions to process (default: .wav,.mp3,.flac)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of audio files to process
    extensions = args.extensions.split(',')
    audio_files = []
    
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"No audio files found in {args.input_dir} with extensions {extensions}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file
    success_count = 0
    
    for input_file in tqdm(audio_files, desc="Processing audio files"):
        # Determine output path
        rel_path = os.path.relpath(input_file, args.input_dir)
        output_file = os.path.join(args.output_dir, rel_path)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Process the audio file
        success = process_audio(
            input_file, 
            output_file, 
            trim_threshold_db=args.threshold, 
            target_level_db=args.target_level
        )
        
        if success:
            success_count += 1
    
    print(f"Successfully processed {success_count} out of {len(audio_files)} files")
    print(f"Processed files saved to {args.output_dir}")

if __name__ == "__main__":
    main()