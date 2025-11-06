#!/usr/bin/env python3
"""
Script to create final_transcript.txt for conversations 3-10
by merging deepgram_transcript.txt (or timestamped_transcription.txt) with hume_emotion.txt
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def merge_transcripts(deepgram_file, hume_file, output_file):
    """Merge Deepgram and Hume transcripts using OpenAI.
    
    Args:
        deepgram_file: Path to Deepgram transcript file
        hume_file: Path to Hume emotion transcript file
        output_file: Path to save the merged transcript
    
    Returns:
        str: Merged transcript text or None if failed
    """
    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
            return None
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Read both input files
        with open(deepgram_file, "r", encoding="utf-8") as f:
            deepgram_text = f.read()
        
        with open(hume_file, "r", encoding="utf-8") as f:
            hume_text = f.read()
        
        # Prepare prompt for the LLM
        prompt = f"""
You are an expert transcript editor.

I have two sources of transcription data:
1. Deepgram transcript (accurate text + timestamps)
2. Hume emotion transcript (speaker emotion context but possibly less accurate timing)

Your task:
- Merge both into a single coherent transcript.
- Keep timestamps from Deepgram.
- Integrate emotional tone or speaker intent from Hume where relevant.
- Preserve filler words ("uh", "um", "hmm") from Deepgram.
- Ensure formatting is clean and readable.

Here is the data:

### Deepgram Transcript:
{deepgram_text}

### Hume Emotion Transcript:
{hume_text}

Now produce the final merged transcript.
- Do NOT use Markdown formatting.
- Do NOT use asterisks (*) or double asterisks (**).
- Simply write plain text like:
  [9.280,11.120] Speaker 1: Hello there. (Emotion: Sadness, Surprise)


Now produce the final merged transcript.
"""
        
        # Call the model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that merges transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Extract the model output
        merged_transcript = response.choices[0].message.content
        
        # Save the result to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(merged_transcript)
        
        return merged_transcript
        
    except Exception as e:
        print(f"❌ Failed to merge transcripts: {e}")
        import traceback
        traceback.print_exc()
        return None

def format_deepgram_transcript(timestamped_file):
    """Format timestamped_transcription.txt to readable format for merge function.
    
    Args:
        timestamped_file: Path to timestamped_transcription.txt
    
    Returns:
        str: Formatted transcript text
    """
    with open(timestamped_file, "r", encoding="utf-8") as f:
        timestamped_lines = f.readlines()
    
    # Format it for better readability in the merge prompt
    formatted_deepgram = ""
    for line in timestamped_lines:
        line = line.strip()
        if line:
            # Format: [start,end]  SPEAKER_ID  transcript
            parts = line.split('\t')
            if len(parts) >= 3:
                time_range = parts[0]
                speaker = parts[1]
                transcript = '\t'.join(parts[2:])  # In case transcript contains tabs
                formatted_deepgram += f"{time_range} {speaker}: {transcript}\n"
    
    return formatted_deepgram

def main():
    """Main function to create final transcripts for conversations 3-10."""
    base_folder = "Recording"
    
    # Process conversations 3 to 10
    start_num = 3
    end_num = 10
    
    print(f"Creating final transcripts for conversations {start_num} to {end_num}...")
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
        
        # Check for required files
        timestamped_file = os.path.join(folder_path, "timestamped_transcription.txt")
        deepgram_file = os.path.join(folder_path, "deepgram_transcript.txt")
        hume_file = os.path.join(folder_path, "hume_emotion.txt")
        final_file = os.path.join(folder_path, "final_transcript.txt")
        
        # Check if final_transcript.txt already exists
        if os.path.exists(final_file):
            print(f"  ℹ️  final_transcript.txt already exists, skipping")
            skipped += 1
            continue
        
        # Check if hume_emotion.txt exists
        if not os.path.exists(hume_file):
            print(f"  ⚠️  hume_emotion.txt not found, skipping")
            skipped += 1
            continue
        
        # Check for deepgram transcript (prefer deepgram_transcript.txt, fallback to timestamped_transcription.txt)
        deepgram_text = None
        if os.path.exists(deepgram_file):
            print(f"  ✓ Found deepgram_transcript.txt")
            with open(deepgram_file, "r", encoding="utf-8") as f:
                deepgram_text = f.read()
        elif os.path.exists(timestamped_file):
            print(f"  ✓ Found timestamped_transcription.txt, formatting...")
            deepgram_text = format_deepgram_transcript(timestamped_file)
            # Save formatted version for future use
            with open(deepgram_file, "w", encoding="utf-8") as f:
                f.write(deepgram_text)
            print(f"  ✓ Created deepgram_transcript.txt")
        else:
            print(f"  ⚠️  No Deepgram transcript found (neither deepgram_transcript.txt nor timestamped_transcription.txt), skipping")
            skipped += 1
            continue
        
        try:
            # Create temporary deepgram file if we formatted it
            temp_deepgram_file = os.path.join(folder_path, "temp_deepgram_for_merge.txt")
            with open(temp_deepgram_file, "w", encoding="utf-8") as f:
                f.write(deepgram_text)
            
            # Merge the transcripts
            print(f"  🔄 Merging transcripts with OpenAI...")
            result = merge_transcripts(temp_deepgram_file, hume_file, final_file)
            
            # Clean up temp file
            if os.path.exists(temp_deepgram_file):
                os.remove(temp_deepgram_file)
            
            if result:
                print(f"  ✅ Successfully created final_transcript.txt")
                successful += 1
            else:
                print(f"  ❌ Failed to create final_transcript.txt")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Error processing {folder_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print("Final transcript creation complete!")
    print(f"  ✅ Successful: {successful}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Total processed: {successful + skipped + failed}")

if __name__ == "__main__":
    main()

