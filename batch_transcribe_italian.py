#!/usr/bin/env python3
"""
Batch transcription script for Italian conversations.
Uses Deepgram API to transcribe conversation.wav files and saves as transcript.txt
"""

import os
import json
import argparse
from pathlib import Path
from deepgram import DeepgramClient
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

def transcribe_audio(audio_path, output_file, language="it"):
    """Transcribe audio file with speaker diarization."""
    try:
        # Get API key from environment
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            print(f"❌ DEEPGRAM_API_KEY not found in environment variables. Please set it in .env file.")
            return None
        
        # Create Deepgram client with API key
        deepgram = DeepgramClient(api_key=api_key)
        
        # Prepare transcription parameters
        transcription_params = {
            "model": "nova-3",
            "language": language,
            "smart_format": True,
            "paragraphs": True,
            "diarize": True,
            "filler_words": True,
        }
        
        # Transcribe with diarization and summary
        with open(audio_path, "rb") as audio_file:
            response = deepgram.listen.v1.media.transcribe_file(
                request=audio_file.read(),
                **transcription_params
            )
        
        # Convert response to dict
        response_json = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
        
        # Debug: Save full response to see structure
        debug_file = output_file.replace('.txt', '_full_response.json')
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, indent=2, default=str, ensure_ascii=False)
        
        # Extract utterances with speaker information
        utterances = []
        
        if 'results' in response_json and 'channels' in response_json['results']:
            for channel in response_json['results']['channels']:
                if 'alternatives' in channel and len(channel['alternatives']) > 0:
                    alternative = channel['alternatives'][0]
                    
                    # Use paragraphs/sentences approach
                    if 'paragraphs' in alternative:
                        paragraphs_data = alternative['paragraphs']
                        
                        # Handle both dict and list structures
                        if isinstance(paragraphs_data, dict):
                            paragraphs_list = paragraphs_data.get('paragraphs', [])
                        elif isinstance(paragraphs_data, list):
                            paragraphs_list = paragraphs_data
                        else:
                            paragraphs_list = []
                        
                        for para_idx, paragraph in enumerate(paragraphs_list):
                            if paragraph and 'sentences' in paragraph:
                                sentences_list = paragraph['sentences']
                                if sentences_list:
                                    for sentence in sentences_list:
                                        speaker_id = paragraph.get('speaker', 0)  # Speaker is on paragraph level
                                        start = sentence.get('start', 0)
                                        end = sentence.get('end', 0)
                                        transcript = sentence.get('text', '').strip()
                                        
                                        if transcript:
                                            utterances.append({
                                                'start': start,
                                                'end': end,
                                                'speaker': f'Speaker {int(speaker_id) + 1}',
                                                'transcript': transcript
                                            })
        
        # Sort by start time
        utterances.sort(key=lambda x: x['start'])
        
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            for utt in utterances:
                # Format: [start,end]  SPEAKER_ID  transcript
                line = f"[{utt['start']:.3f},{utt['end']:.3f}]\t{utt['speaker']}\t{utt['transcript']}\n"
                f.write(line)
        
        # Extract summary if available
        summary_text = None
        if 'results' in response_json and 'summary' in response_json['results']:
            summary = response_json['results']['summary']
            if summary and 'short' in summary:
                summary_text = summary['short']
                
                # Save summary to separate file
                summary_file = output_file.replace('transcript.txt', 'summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
        
        speakers = set(utt['speaker'] for utt in utterances)
        return {
            'utterances': utterances,
            'speakers': speakers,
            'count': len(utterances),
            'summary': summary_text
        }
    except Exception as e:
        print(f"❌ Transcription failed for {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to batch transcribe conversations in a specified range."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Batch transcribe Italian conversations using Deepgram API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_transcribe_italian.py          # Transcribe conversations 1-4 (default)
  python batch_transcribe_italian.py 1 4       # Transcribe conversations 1-4
  python batch_transcribe_italian.py 1 2       # Transcribe conversations 1-2
        """
    )
    parser.add_argument(
        "start",
        type=int,
        nargs="?",
        default=1,
        help="Starting conversation number (default: 1)"
    )
    parser.add_argument(
        "end",
        type=int,
        nargs="?",
        default=4,
        help="Ending conversation number (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Validate range
    if args.start < 1:
        print("❌ Error: Start conversation number must be >= 1")
        return
    
    if args.end < args.start:
        print(f"❌ Error: End conversation number ({args.end}) must be >= start ({args.start})")
        return
    
    base_folder = Path("Recording/Italian")
    
    if not base_folder.exists():
        print(f"❌ Error: Folder {base_folder} does not exist!")
        return
    
    # Check for API key
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("❌ DEEPGRAM_API_KEY not found in environment variables. Please set it in .env file.")
        return
    
    print(f"✅ Starting batch transcription for conversations {args.start}-{args.end} in {base_folder}")
    print(f"📝 Output files will be saved as 'transcript.txt' in each conversation folder\n")
    
    # Process conversations in specified range
    successful = 0
    failed = 0
    skipped = 0
    
    for conv_num in tqdm(range(args.start, args.end + 1), desc="Processing conversations"):
        conv_folder = base_folder / f"Conversation_{conv_num}"
        audio_file = conv_folder / "conversation.wav"
        output_file = conv_folder / "transcript.txt"
        
        # Check if conversation folder exists
        if not conv_folder.exists():
            print(f"⚠️  Skipping Conversation_{conv_num}: Folder does not exist")
            skipped += 1
            continue
        
        # Check if audio file exists
        if not audio_file.exists():
            print(f"⚠️  Skipping Conversation_{conv_num}: conversation.wav not found")
            skipped += 1
            continue
        
        # Check if transcript already exists (optional - comment out if you want to overwrite)
        # if output_file.exists():
        #     print(f"ℹ️  Skipping Conversation_{conv_num}: transcript.txt already exists")
        #     skipped += 1
        #     continue
        
        # Transcribe
        print(f"\n🎙️  Transcribing Conversation_{conv_num}...")
        result = transcribe_audio(str(audio_file), str(output_file), language="it")
        
        if result:
            print(f"✅ Conversation_{conv_num}: {result['count']} utterances, {len(result['speakers'])} speakers")
            if result.get('summary'):
                print(f"   📄 Summary generated")
            successful += 1
        else:
            print(f"❌ Conversation_{conv_num}: Transcription failed")
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 Batch Transcription Summary")
    print("="*60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped: {skipped}")
    print(f"📁 Total processed: {successful + failed + skipped}")
    print("="*60)


if __name__ == "__main__":
    main()

