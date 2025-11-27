#!/usr/bin/env python3
"""
Script to transcribe Arabic conversation.wav files using Speechmatics API.
Transcribes a specific conversation by number or a range of conversations.
"""

import argparse
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError

load_dotenv()

# Speechmatics API configuration
API_KEY = os.getenv("SPEECHMATICS_API_KEY")
if not API_KEY:
    raise ValueError("SPEECHMATICS_API_KEY not found in environment variables. Please set it in .env file.")

BASE_FOLDER = Path(__file__).parent / "Recording" / "Arabic"

# Transcription configuration as specified by user
TRANSCRIPTION_CONFIG = {
    "audio_events_config": {
        "types": [
            "laughter",
            "music",
            "applause"
        ]
    },
    "auto_chapters_config": {},
    "summarization_config": {},
    "topic_detection_config": {},
    "transcription_config": {
        "audio_filtering_config": {
            "volume_threshold": 0
        },
        "diarization": "speaker",
        "enable_entities": True,
        "language": "auto",
        "operating_point": "enhanced"
    },
    "type": "transcription"
}

settings = ConnectionSettings(
    url="https://asr.api.speechmatics.com/v2",
    auth_token=API_KEY,
)


def extract_transcription_text(transcript):
    """
    Extract text transcription from Speechmatics JSON response.
    Returns a list of utterances with timestamps and speaker labels.
    Creates sentence-level segments for better readability.
    """
    results = transcript.get("results", [])
    
    # Build list of items with their speaker, content, and timestamps
    items = []
    for item in results:
        if isinstance(item, dict) and "alternatives" in item:
            start_time = item.get("start_time", 0)
            end_time = item.get("end_time", 0)
            item_type = item.get("type", "")
            is_eos = item.get("is_eos", False)
            
            # Get the best alternative
            alternatives = item.get("alternatives", [])
            if alternatives:
                best_alt = alternatives[0]  # Usually the first is the best
                content = best_alt.get("content", "")
                speaker = best_alt.get("speaker", "?")
                
                items.append({
                    "content": content,
                    "speaker": speaker,
                    "start": start_time,
                    "end": end_time,
                    "type": item_type,
                    "is_eos": is_eos
                })
    
    # Group items into sentence-level utterances
    utterances = []
    current_sentence = None
    current_text = []
    sentence_start = None
    
    for item in items:
        speaker = item.get("speaker", "?")
        content = item.get("content", "")
        start_time = item.get("start", 0)
        end_time = item.get("end", 0)
        is_eos = item.get("is_eos", False)
        item_type = item.get("type", "")
        
        # Start new sentence if needed (new speaker or first item)
        if current_sentence is None or speaker != current_sentence.get("speaker"):
            # Save previous sentence if exists
            if current_sentence is not None and current_text:
                timestamp = f"[{current_sentence.get('start', 0):.3f},{current_sentence.get('end', 0):.3f}]"
                current_speaker = current_sentence.get('speaker', '?')
                if current_speaker.startswith('S'):
                    speaker_label = f"Speaker {current_speaker[1:]}"
                else:
                    speaker_label = f"Speaker {current_speaker}"
                text = ''.join(current_text).strip()
                if text:
                    utterances.append({
                        "timestamp": timestamp,
                        "speaker": speaker_label,
                        "text": text,
                        "start": current_sentence.get('start', 0),
                        "end": current_sentence.get('end', 0)
                    })
            
            # Start new sentence
            current_sentence = {
                "speaker": speaker,
                "start": start_time if item_type == "word" else None,
                "end": end_time
            }
            current_text = []
            sentence_start = start_time if item_type == "word" else None
        
        # Set sentence start time on first word
        if sentence_start is None and item_type == "word":
            sentence_start = start_time
            if current_sentence["start"] is None:
                current_sentence["start"] = start_time
        
        # Add content to current sentence
        if item_type == "word":
            if current_text and current_text[-1] not in '.!,?;:':
                current_text.append(' ')
            current_text.append(content)
        elif item_type == "punctuation":
            # Punctuation attaches to previous word, no space
            current_text.append(content)
        
        # Update end time
        if current_sentence is not None:
            current_sentence["end"] = end_time
        
        # If end of sentence, finalize this sentence and prepare for next
        if is_eos:
            if current_sentence is not None and current_text:
                timestamp = f"[{current_sentence.get('start', 0):.3f},{current_sentence.get('end', 0):.3f}]"
                current_speaker = current_sentence.get('speaker', '?')
                if current_speaker.startswith('S'):
                    speaker_label = f"Speaker {current_speaker[1:]}"
                else:
                    speaker_label = f"Speaker {current_speaker}"
                text = ''.join(current_text).strip()
                if text:
                    utterances.append({
                        "timestamp": timestamp,
                        "speaker": speaker_label,
                        "text": text,
                        "start": current_sentence.get('start', 0),
                        "end": current_sentence.get('end', 0)
                    })
            
            # Reset for next sentence (but keep same speaker if no change)
            current_text = []
            sentence_start = None
            # Don't reset current_sentence yet - wait to see if speaker changes
    
    # Save last sentence if exists
    if current_sentence is not None and current_text:
        timestamp = f"[{current_sentence.get('start', 0):.3f},{current_sentence.get('end', 0):.3f}]"
        current_speaker = current_sentence.get('speaker', '?')
        if current_speaker.startswith('S'):
            speaker_label = f"Speaker {current_speaker[1:]}"
        else:
            speaker_label = f"Speaker {current_speaker}"
        text = ''.join(current_text).strip()
        if text:
            utterances.append({
                "timestamp": timestamp,
                "speaker": speaker_label,
                "text": text,
                "start": current_sentence.get('start', 0),
                "end": current_sentence.get('end', 0)
            })
    
    # Chain timestamps for consecutive same-speaker utterances
    # When the same speaker continues, next utterance starts where previous ended
    # When speaker changes, keep original start time
    chained_utterances = []
    prev_speaker = None
    prev_end_time = None
    
    for utt in utterances:
        # Parse timestamp to get original start and end times
        timestamp_str = utt.get("timestamp", "")
        match = re.match(r'\[([\d.]+),([\d.]+)\]', timestamp_str)
        if match:
            original_start = float(match.group(1))
            original_end = float(match.group(2))
        else:
            # Fallback to stored values if available
            original_start = utt.get("start", 0)
            original_end = utt.get("end", 0)
        
        speaker = utt.get("speaker", "")
        
        # If same speaker as previous, chain timestamps
        if speaker == prev_speaker and prev_end_time is not None:
            start_time = prev_end_time
        else:
            # New speaker or first utterance - keep original start time
            start_time = original_start
        
        # Update timestamp string
        new_timestamp = f"[{start_time:.3f},{original_end:.3f}]"
        chained_utterances.append({
            "timestamp": new_timestamp,
            "speaker": speaker,
            "text": utt.get("text", "")
        })
        
        # Update tracking variables
        prev_speaker = speaker
        prev_end_time = original_end
    
    return chained_utterances


def transcribe_conversation(conversation_path, output_folder):
    """
    Transcribe a single conversation.wav file using Speechmatics API.
    
    Args:
        conversation_path: Path to conversation.wav file
        output_folder: Folder where transcription will be saved
    """
    print(f"\n🎤 Transcribing: {conversation_path}")
    
    try:
        with BatchClient(settings) as client:
            # Submit job
            job_id = client.submit_job(
                audio=str(conversation_path),
                transcription_config=TRANSCRIPTION_CONFIG,
            )
            print(f"   ✓ Job {job_id} submitted successfully, waiting for transcript...")
            
            # Wait for completion
            transcript = client.wait_for_completion(job_id, transcription_format="json-v2")
            print(f"   ✓ Transcription completed!")
            
            # Save full JSON response for debugging
            json_file = output_folder / "speechmatics_full_response.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False, default=str)
            print(f"   ✓ Saved full JSON response to: {json_file}")
            
            # Extract text transcription
            utterances = extract_transcription_text(transcript)
            
            # Save transcription to file
            output_file = output_folder / "transcription.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for utt in utterances:
                    f.write(f"{utt['timestamp']}\t{utt['speaker']}\t{utt['text']}\n")
            
            print(f"   ✓ Saved transcription to: {output_file}")
            print(f"   ✓ Total utterances: {len(utterances)}")
            
            return True
            
    except HTTPStatusError as e:
        if e.response.status_code == 401:
            print(f"   ❌ Invalid API key - Check your SPEECHMATICS_API_KEY!")
        elif e.response.status_code == 400:
            print(f"   ❌ Bad request: {e.response.json()}")
        elif e.response.status_code == 403:
            print(f"   ❌ Forbidden - Speechmatics API access denied")
        else:
            print(f"   ❌ HTTP error ({e.response.status_code}): {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error transcribing {conversation_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to transcribe Arabic conversation(s)."""
    parser = argparse.ArgumentParser(
        description="Transcribe Arabic conversation.wav file(s) using Speechmatics API"
    )
    parser.add_argument(
        "start",
        type=int,
        help="First conversation number (e.g., 1 for Conversation_1)"
    )
    parser.add_argument(
        "end",
        type=int,
        nargs="?",
        default=None,
        help="Last conversation number (optional). If provided, processes range from start to end (inclusive)"
    )
    
    args = parser.parse_args()
    start_number = args.start
    end_number = args.end if args.end is not None else args.start
    
    # Validate range
    if start_number > end_number:
        print(f"❌ Error: Start number ({start_number}) must be <= end number ({end_number})")
        return
    
    if not BASE_FOLDER.exists():
        print(f"❌ Base folder not found: {BASE_FOLDER}")
        return
    
    # Determine if processing single or range
    if start_number == end_number:
        print(f"📁 Processing Conversation_{start_number}")
        conversation_numbers = [start_number]
    else:
        print(f"📁 Processing Conversations {start_number} to {end_number} (inclusive)")
        conversation_numbers = list(range(start_number, end_number + 1))
    
    # Process each conversation
    successful = 0
    failed = 0
    
    for conversation_number in conversation_numbers:
        # Construct path to specific conversation folder
        conversation_folder = BASE_FOLDER / f"Conversation_{conversation_number}"
        conversation_path = conversation_folder / "conversation.wav"
        
        if not conversation_folder.exists():
            print(f"❌ Conversation folder not found: {conversation_folder}")
            failed += 1
            continue
        
        if not conversation_path.exists():
            print(f"❌ conversation.wav not found in: {conversation_folder}")
            failed += 1
            continue
        
        # Transcribe the conversation
        if transcribe_conversation(conversation_path, conversation_folder):
            successful += 1
        else:
            failed += 1
    
    # Summary
    if start_number == end_number:
        if successful > 0:
            print(f"\n✅ Successfully transcribed Conversation_{start_number}")
        else:
            print(f"\n❌ Failed to transcribe Conversation_{start_number}")
    else:
        print(f"\n📊 Summary: {successful} successful, {failed} failed out of {len(conversation_numbers)} conversations")


if __name__ == "__main__":
    main()

