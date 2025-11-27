#!/usr/bin/env python3
"""
Script to batch create TextGrid files for multiple conversations.
Processes conversations 12-16 in the Arabic folder.
"""

import sys
from pathlib import Path
from create_textgrid import create_textgrid


def main():
    """Main function to create TextGrid files for conversations 12-16."""
    base_folder = Path(__file__).parent / "Recording" / "Arabic"
    conversation_numbers = [12, 13, 14, 15, 16]
    
    print(f"Processing conversations {conversation_numbers[0]}-{conversation_numbers[-1]} in Arabic folder\n")
    
    success_count = 0
    error_count = 0
    
    for conv_num in conversation_numbers:
        print(f"\n{'='*60}")
        print(f"Processing Conversation {conv_num}")
        print(f"{'='*60}")
        
        conversation_folder = base_folder / f"Conversation_{conv_num}"
        # Check for transcription.txt first (new format), fallback to transcript.txt (old format)
        transcription_file = conversation_folder / "transcription.txt"
        if not transcription_file.exists():
            transcription_file = conversation_folder / "transcript.txt"
        output_file = conversation_folder / "transcription.TextGrid"
        
        # Check if folder exists
        if not conversation_folder.exists():
            print(f"❌ Conversation folder not found: {conversation_folder}")
            error_count += 1
            continue
        
        # Check if transcription file exists
        if not transcription_file.exists():
            print(f"❌ Transcription file not found: {transcription_file}")
            error_count += 1
            continue
        
        try:
            create_textgrid(transcription_file, output_file)
            success_count += 1
        except Exception as e:
            print(f"❌ Error creating TextGrid for Conversation {conv_num}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"Total conversations: {len(conversation_numbers)}")
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

