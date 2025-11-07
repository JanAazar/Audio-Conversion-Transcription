#!/usr/bin/env python3
"""
Script to merge audio files for a range of conversations.
Converts .m4a files to mono and merges them into stereo conversation.wav.
"""

import os
import soundfile as sf
import librosa
import numpy as np
import sys
from pathlib import Path

def convert_to_mono(audio_file, target_sr=48000):
    """
    Convert audio file to mono.
    
    Args:
        audio_file: Path to input audio file
        target_sr: Target sample rate in Hz
    """
    audio_data, sample_rate = librosa.load(audio_file, sr=target_sr, mono=False)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=0)
    
    return audio_data, sample_rate

def process_audio(first_file, second_file, folder_path):
    """Process and merge two audio files (same as app.py process_audio function)."""
    
    print(f"  Processing: {os.path.basename(first_file)} + {os.path.basename(second_file)}")
    
    # Convert both files to mono
    print("    Converting files to mono...")
    first_data, first_sr = convert_to_mono(first_file)
    second_data, second_sr = convert_to_mono(second_file)
    
    # Save mono files
    first_mono_path = os.path.join(folder_path, "first_speaker.wav")
    second_mono_path = os.path.join(folder_path, "second_speaker.wav")
    
    sf.write(first_mono_path, first_data, first_sr, format='WAV', subtype='PCM_16')
    sf.write(second_mono_path, second_data, second_sr, format='WAV', subtype='PCM_16')
    print("    ✓ Saved mono speaker files")
    
    # Ensure both are the same sample rate
    if first_sr != second_sr:
        print(f"    Aligning sample rates ({first_sr} Hz vs {second_sr} Hz)...")
        if first_sr > second_sr:
            second_data = np.interp(
                np.linspace(0, len(second_data), int(len(second_data) * first_sr / second_sr)),
                np.arange(len(second_data)),
                second_data
            )
            sample_rate = first_sr
        else:
            first_data = np.interp(
                np.linspace(0, len(first_data), int(len(first_data) * second_sr / first_sr)),
                np.arange(len(first_data)),
                first_data
            )
            sample_rate = second_sr
    else:
        sample_rate = first_sr
    
    # Ensure both are the same duration (pad shorter one with zeros)
    print("    Aligning audio durations...")
    max_len = max(len(first_data), len(second_data))
    if len(first_data) < max_len:
        first_data = np.pad(first_data, (0, max_len - len(first_data)), 'constant')
    if len(second_data) < max_len:
        second_data = np.pad(second_data, (0, max_len - len(second_data)), 'constant')
    
    # Combine into stereo
    print("    Creating stereo output...")
    stereo = np.column_stack([first_data, second_data])
    
    # Save conversation file
    conversation_path = os.path.join(folder_path, "conversation.wav")
    sf.write(conversation_path, stereo, sample_rate, format='WAV', subtype='PCM_16')
    
    duration = len(first_data) / sample_rate
    print(f"    ✓ Saved conversation.wav ({duration:.2f}s, {sample_rate} Hz, stereo)")
    
    return conversation_path

def parse_range_args(default_start=63, default_end=73):
    """Parse optional start and end arguments from command line."""
    if len(sys.argv) == 1:
        return default_start, default_end
    if len(sys.argv) == 2:
        try:
            start = int(sys.argv[1])
            return start, start
        except ValueError:
            raise ValueError("Start conversation number must be an integer")
    if len(sys.argv) >= 3:
        try:
            start = int(sys.argv[1])
            end = int(sys.argv[2])
            if start > end:
                raise ValueError("Start conversation number must be less than or equal to end number")
            return start, end
        except ValueError:
            raise ValueError("Start and end conversation numbers must be integers")


def main():
    """Main function to process a range of conversations."""
    base_folder = "Recording"
    
    # Process conversations in provided range (defaults to 63-73)
    start_num, end_num = parse_range_args()
    
    print(f"Merging conversations {start_num} to {end_num}...")
    print("=" * 60)
    
    successful = 0
    skipped = 0
    failed = 0
    
    for conv_num in range(start_num, end_num + 1):
        folder_name = f"Conversation_{conv_num}"
        folder_path = os.path.join(base_folder, folder_name)
        
        print(f"\n[{conv_num}/{end_num}] Processing {folder_name}...")
        
        if not os.path.exists(folder_path):
            print(f"  ⚠️  Folder does not exist, skipping")
            skipped += 1
            continue
        
        # Find .m4a files in the folder
        m4a_files = list(Path(folder_path).glob("*.m4a"))
        
        if len(m4a_files) < 2:
            print(f"  ⚠️  Found {len(m4a_files)} .m4a file(s), need 2. Skipping")
            skipped += 1
            continue
        
        # Use first two .m4a files found
        first_file = str(m4a_files[0])
        second_file = str(m4a_files[1])
        
        # Check if conversation.wav already exists
        conversation_path = os.path.join(folder_path, "conversation.wav")
        if os.path.exists(conversation_path):
            print(f"  ℹ️  conversation.wav already exists, skipping")
            skipped += 1
            continue
        
        try:
            # Process and merge the audio files
            result = process_audio(first_file, second_file, folder_path)
            print(f"  ✅ Successfully merged {folder_name}")
            successful += 1
        except Exception as e:
            print(f"  ❌ Error processing {folder_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"  ✅ Successful: {successful}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Total processed: {successful + skipped + failed}")

if __name__ == "__main__":
    main()

