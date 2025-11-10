import os
from pathlib import Path
import soundfile as sf

def get_total_audio_duration():
    """Calculate total duration of all conversation.wav files in Recording subfolders."""
    recording_dir = Path("Recording")
    
    if not recording_dir.exists():
        print(f"Error: {recording_dir} directory not found!")
        return
    
    total_duration = 0.0
    file_count = 0
    files_processed = []
    
    # Loop through all subdirectories
    for subfolder in recording_dir.iterdir():
        if subfolder.is_dir():
            conversation_file = subfolder / "conversation.wav"
            
            if conversation_file.exists():
                try:
                    # Get audio info using soundfile
                    with sf.SoundFile(str(conversation_file)) as f:
                        duration = len(f) / f.samplerate
                        total_duration += duration
                        file_count += 1
                        files_processed.append({
                            'folder': subfolder.name,
                            'duration': duration
                        })
                        print(f"✓ {subfolder.name}/conversation.wav: {duration:.2f} seconds ({duration/60:.2f} minutes)")
                except Exception as e:
                    print(f"✗ Error reading {conversation_file}: {e}")
            else:
                print(f"⚠ No conversation.wav found in {subfolder.name}")
    
    print("\n" + "="*60)
    print(f"SUMMARY:")
    print(f"  Total files processed: {file_count}")
    print(f"  Total duration: {total_duration:.2f} seconds")
    print(f"  Total duration: {total_duration/60:.2f} minutes")
    print(f"  Total duration: {total_duration/3600:.2f} hours")
    print("="*60)
    
    return total_duration, files_processed

if __name__ == "__main__":
    get_total_audio_duration()