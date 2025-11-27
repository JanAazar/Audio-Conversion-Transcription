#!/usr/bin/env python3
"""
Script to merge audio files for a range of conversations and then clean up.
Converts .m4a files to mono and merges them into stereo conversation.wav.
Then deletes all files except first_speaker.wav, second_speaker.wav, and conversation.wav.
"""

import os
import soundfile as sf
import librosa
import numpy as np
import argparse
from pathlib import Path

def convert_to_mono(audio_file, target_sr=48000):
    """
    Convert audio file to mono.
    
    Args:
        audio_file: Path to input audio file
        target_sr: Target sample rate in Hz
    """
    audio_data, sample_rate = librosa.load(audio_file, sr=target_sr, mono=False)
    
    if audio_data.ndim > 1:
        channel_rms = np.sqrt(np.mean(audio_data**2, axis=1))
        best_channel = int(np.argmax(channel_rms))
        
        averaged = np.mean(audio_data, axis=0)
        averaged_rms = np.sqrt(np.mean(averaged**2))
        best_rms = channel_rms[best_channel] if channel_rms.size else 0.0
        
        if best_rms > 0 and (averaged_rms < 0.1 * best_rms):
            print(
                f"    ⚠️  Detected unbalanced stereo in {os.path.basename(audio_file)}. "
                f"Using channel {best_channel + 1} only."
            )
            audio_data = audio_data[best_channel]
        else:
            audio_data = averaged
    
    return audio_data.astype(np.float32), sample_rate

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

def cleanup_conversation_folder(folder_path):
    """Delete all files in folder except the three specified .wav files."""
    if not os.path.exists(folder_path):
        print(f"  ⚠️  Folder does not exist: {folder_path}")
        return False
    
    files_to_keep = {"first_speaker.wav", "second_speaker.wav", "conversation.wav"}
    deleted_count = 0
    
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        # Skip directories
        if os.path.isdir(item_path):
            continue
        
        # Delete if not in the keep list
        if item not in files_to_keep:
            try:
                os.remove(item_path)
                print(f"    🗑️  Deleted: {item}")
                deleted_count += 1
            except Exception as e:
                print(f"    ❌ Error deleting {item}: {e}")
                return False
    
    if deleted_count > 0:
        print(f"  ✅ Cleaned up {folder_path} ({deleted_count} file(s) deleted)")
    else:
        print(f"  ℹ️  No files to delete in {folder_path}")
    
    return True

def main():
    """Main function to merge and then clean up a range of conversations."""
    parser = argparse.ArgumentParser(
        description="Merge audio files for conversations and then clean up folders by deleting all files except first_speaker.wav, second_speaker.wav, and conversation.wav"
    )
    parser.add_argument(
        "start",
        type=int,
        help="Starting conversation number (e.g., 63)"
    )
    parser.add_argument(
        "end",
        type=int,
        help="Ending conversation number (inclusive, e.g., 73)"
    )
    parser.add_argument(
        "--exclude",
        type=int,
        nargs="+",
        default=[],
        help="Conversation numbers to exclude (e.g., --exclude 65 66)"
    )
    
    args = parser.parse_args()
    
    base_folder = "Recording/Spanish-Regular"
    
    # Conversations to process
    conversation_numbers = list(range(args.start, args.end + 1))
    
    # Remove excluded conversations
    for excluded in args.exclude:
        if excluded in conversation_numbers:
            conversation_numbers.remove(excluded)
    
    if not conversation_numbers:
        print("No conversations to process after exclusions.")
        return
    
    exclude_text = f" (excluding {', '.join(map(str, args.exclude))})" if args.exclude else ""
    
    # ========== STEP 1: MERGE AUDIO FILES ==========
    print(f"Merging conversations: {conversation_numbers[0]} to {conversation_numbers[-1]}{exclude_text}")
    print("=" * 60)
    
    merge_successful = 0
    merge_skipped = 0
    merge_failed = 0
    
    for conv_num in conversation_numbers:
        folder_name = f"Conversation_{conv_num}"
        folder_path = os.path.join(base_folder, folder_name)
        
        print(f"\n[{conv_num}/{conversation_numbers[-1]}] Processing {folder_name}...")
        
        if not os.path.exists(folder_path):
            print(f"  ⚠️  Folder does not exist, skipping")
            merge_skipped += 1
            continue
        
        # Find .m4a files in the folder
        m4a_files = list(Path(folder_path).glob("*.m4a"))
        
        if len(m4a_files) < 2:
            print(f"  ⚠️  Found {len(m4a_files)} .m4a file(s), need 2. Skipping")
            merge_skipped += 1
            continue
        
        # Use first two .m4a files found
        first_file = str(m4a_files[0])
        second_file = str(m4a_files[1])
        
        # Check if conversation.wav already exists
        conversation_path = os.path.join(folder_path, "conversation.wav")
        if os.path.exists(conversation_path):
            print(f"  ℹ️  conversation.wav already exists, skipping merge")
            merge_skipped += 1
        else:
            try:
                # Process and merge the audio files
                result = process_audio(first_file, second_file, folder_path)
                print(f"  ✅ Successfully merged {folder_name}")
                merge_successful += 1
            except Exception as e:
                print(f"  ❌ Error processing {folder_name}: {e}")
                import traceback
                traceback.print_exc()
                merge_failed += 1
                continue
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"  ✅ Successful: {merge_successful}")
    print(f"  ⏭️  Skipped: {merge_skipped}")
    print(f"  ❌ Failed: {merge_failed}")
    print(f"  📊 Total processed: {merge_successful + merge_skipped + merge_failed}")
    
    # ========== STEP 2: CLEANUP FOLDERS ==========
    print("\n" + "=" * 60)
    print(f"Cleaning up conversations: {conversation_numbers[0]} to {conversation_numbers[-1]}{exclude_text}")
    print("=" * 60)
    
    cleanup_successful = 0
    cleanup_failed = 0
    
    for conv_num in conversation_numbers:
        folder_name = f"Conversation_{conv_num}"
        folder_path = os.path.join(base_folder, folder_name)
        
        print(f"\n[{conv_num}] Cleaning up {folder_name}...")
        
        if cleanup_conversation_folder(folder_path):
            cleanup_successful += 1
        else:
            cleanup_failed += 1
    
    print("\n" + "=" * 60)
    print("Cleanup complete!")
    print(f"  ✅ Successful: {cleanup_successful}")
    print(f"  ❌ Failed: {cleanup_failed}")
    print(f"  📊 Total processed: {cleanup_successful + cleanup_failed}")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 60)
    print("All operations complete!")
    print(f"  Merge - ✅: {merge_successful}, ⏭️: {merge_skipped}, ❌: {merge_failed}")
    print(f"  Cleanup - ✅: {cleanup_successful}, ❌: {cleanup_failed}")

if __name__ == "__main__":
    main()

