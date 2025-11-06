#!/usr/bin/env python3
"""
Batch processing script to generate final transcripts for multiple conversations.
Processes Conversation_2 through Conversation_10 (excluding Conversation_6).
"""

import os
import json
import re
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from deepgram import DeepgramClient
from hume import AsyncHumeClient
from hume.expression_measurement.batch import Face, Prosody, Models
from hume.expression_measurement.batch.types import InferenceBaseRequest
from openai import OpenAI

# Load environment variables
load_dotenv()

def transcribe_audio(audio_path, output_file, language="es"):
    """Transcribe audio file with speaker diarization (non-Streamlit version)."""
    try:
        print(f"  📝 Transcribing {audio_path}...")
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            print("  ❌ DEEPGRAM_API_KEY not found in environment variables.")
            return None
        
        deepgram = DeepgramClient(api_key=api_key)
        
        transcription_params = {
            "model": "nova-3",
            "language": language,
            "smart_format": True,
            "paragraphs": True,
            "diarize": True,
            "filler_words": True,
        }
        
        with open(audio_path, "rb") as audio_file:
            response = deepgram.listen.v1.media.transcribe_file(
                request=audio_file.read(),
                **transcription_params
            )
        
        response_json = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
        
        # Save full response
        debug_file = output_file.replace('.txt', '_full_response.json')
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, indent=2, default=str, ensure_ascii=False)
        
        # Extract utterances
        utterances = []
        if 'results' in response_json and 'channels' in response_json['results']:
            for channel in response_json['results']['channels']:
                if 'alternatives' in channel and len(channel['alternatives']) > 0:
                    alternative = channel['alternatives'][0]
                    if 'paragraphs' in alternative:
                        paragraphs_data = alternative['paragraphs']
                        if isinstance(paragraphs_data, dict):
                            paragraphs_list = paragraphs_data.get('paragraphs', [])
                        elif isinstance(paragraphs_data, list):
                            paragraphs_list = paragraphs_data
                        else:
                            paragraphs_list = []
                        
                        for paragraph in paragraphs_list:
                            if paragraph and 'sentences' in paragraph:
                                sentences_list = paragraph['sentences']
                                if sentences_list:
                                    for sentence in sentences_list:
                                        speaker_id = paragraph.get('speaker', 0)
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
        
        utterances.sort(key=lambda x: x['start'])
        
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            for utt in utterances:
                line = f"[{utt['start']:.3f},{utt['end']:.3f}]\t{utt['speaker']}\t{utt['transcript']}\n"
                f.write(line)
        
        # Extract summary if available
        summary_text = None
        if 'results' in response_json and 'summary' in response_json['results']:
            summary = response_json['results']['summary']
            if summary and 'short' in summary:
                summary_text = summary['short']
                summary_file = output_file.replace('timestamped_transcription.txt', 'summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                print(f"  ✅ Summary saved to: {summary_file}")
        
        print(f"  ✅ Transcription completed! ({len(utterances)} utterances)")
        return {
            'utterances': utterances,
            'speakers': set(utt['speaker'] for utt in utterances),
            'count': len(utterances),
            'summary': summary_text
        }
    except Exception as e:
        print(f"  ❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_hume_predictions(job_predictions):
    """Parse Hume predictions and format them for output."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("FORMATTED TRANSCRIPT WITH EMOTION ANALYSIS")
    lines.append("="*100)
    
    pred_str = str(job_predictions[0]) if job_predictions else ""
    predictions = []
    
    text_start = 'ProsodyPrediction(text='
    segments = pred_str.split(text_start)[1:]
    
    for segment in segments:
        if segment.startswith("'"):
            quote_char = "'"
        elif segment.startswith('"'):
            quote_char = '"'
        else:
            continue
        
        text_start_idx = 1
        text_end_idx = text_start_idx
        while text_end_idx < len(segment) and segment[text_end_idx] != quote_char:
            text_end_idx += 1
        
        if text_end_idx >= len(segment):
            continue
        
        text = segment[text_start_idx:text_end_idx]
        
        time_match = re.search(r"time=TimeInterval\(begin=([\d.]+), end=([\d.]+)\)", segment)
        if not time_match:
            continue
        begin = float(time_match.group(1))
        end = float(time_match.group(2))
        
        conf_match = re.search(r"confidence=([\d.]+)", segment)
        confidence = float(conf_match.group(1)) if conf_match else 1.0
        
        emotion_pattern = r"EmotionScore\(name='([^']+)', score=([\d.]+)\)"
        emotions = []
        for emo_match in re.finditer(emotion_pattern, segment):
            name = emo_match.group(1)
            score = float(emo_match.group(2))
            emotions.append((name, score))
        
        predictions.append({
            'text': text,
            'begin': begin,
            'end': end,
            'confidence': confidence,
            'emotions': emotions
        })
    
    if not predictions:
        lines.append("No predictions found in the response.")
        return "\n".join(lines)
    
    lines.append(f"\n{'Time':<20} {'Transcription':<55} {'Top Emotions'}")
    lines.append("-" * 120)
    
    for pred in predictions:
        text = pred['text']
        time_str = f"{pred['begin']:.1f}s-{pred['end']:.1f}s"
        
        pred['emotions'].sort(key=lambda x: x[1], reverse=True)
        top_emotions = pred['emotions'][:3]
        emotions_text = ", ".join([f"{name} ({score:.0%})" for name, score in top_emotions])
        
        if len(text) > 52:
            text = text[:49] + "..."
        
        lines.append(f"{time_str:<20} {text:<55} {emotions_text}")
    
    lines.append("-" * 120)
    lines.append(f"\nTotal segments: {len(predictions)}")
    lines.append("")
    
    return "\n".join(lines)

async def run_hume_analysis_async(audio_path):
    """Run Hume API analysis on audio file (async function, non-Streamlit version)."""
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        raise ValueError("HUME_API_KEY not found in environment variables")
    
    client = AsyncHumeClient(api_key=api_key)
    
    print(f"  😊 Submitting job to Hume API...")
    
    audio_file = open(audio_path, mode="rb")
    
    prosody_config = Prosody()
    models_chosen = Models(prosody=prosody_config)
    stringified_configs = InferenceBaseRequest(models=models_chosen)
    
    job_id = await client.expression_measurement.batch.start_inference_job_from_local_file(
        json=stringified_configs, file=[audio_file]
    )
    
    audio_file.close()
    
    print(f"  ✅ Job submitted! Job ID: {job_id}")
    print(f"  ⏳ Waiting for predictions (this may take several minutes)...")
    
    max_wait = 600  # Wait up to 10 minutes
    waited = 0
    
    while waited < max_wait:
        try:
            job_predictions = await client.expression_measurement.batch.get_job_predictions(
                id=job_id
            )
            
            if job_predictions and len(job_predictions) > 0:
                pred_str = str(job_predictions[0])
                if 'results=InferenceResults(predictions=[]' not in pred_str:
                    print(f"  ✅ Predictions received!")
                    return {
                        'predictions': job_predictions,
                        'job_id': job_id
                    }
            
            job_details = await client.expression_measurement.batch.get_job_details(
                id=job_id
            )
            status = str(job_details).split("status='")[1].split("'")[0] if "status='" in str(job_details) else "UNKNOWN"
            
            if waited % 30 == 0:  # Print status every 30 seconds
                print(f"  ⏳ Status: {status}, waiting... ({waited}s/{max_wait}s)")
            
            if status == "FAILED":
                raise Exception(f"Job failed: {json.dumps(job_details, indent=2, default=str)}")
            
            await asyncio.sleep(2)
            waited += 2
            
        except Exception as e:
            if "predictions" in locals():
                raise e
            await asyncio.sleep(2)
            waited += 2
    
    raise Exception("Timed out waiting for predictions.")

def hume_emotion_analysis(audio_path, output_folder):
    """Run Hume emotion analysis on audio file and save results."""
    try:
        print(f"  😊 Running Hume emotion analysis...")
        api_key = os.getenv("HUME_API_KEY")
        if not api_key:
            print("  ❌ HUME_API_KEY not found in environment variables.")
            return None
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_hume_analysis_async(audio_path))
        finally:
            loop.close()
        
        formatted_output = parse_hume_predictions(result['predictions'])
        formatted_output += "\n" + "-" * 100 + "\n"
        formatted_output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        output_file = os.path.join(output_folder, "hume_emotion.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        
        print(f"  ✅ Hume emotion analysis completed!")
        return {
            'output_file': output_file,
            'formatted_text': formatted_output,
            'job_id': result['job_id']
        }
        
    except Exception as e:
        print(f"  ❌ Hume emotion analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def merge_transcripts(deepgram_file, hume_file, output_file):
    """Merge Deepgram and Hume transcripts using OpenAI."""
    try:
        print(f"  🔄 Merging transcripts with AI...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("  ❌ OPENAI_API_KEY not found in environment variables.")
            return None
        
        client = OpenAI(api_key=api_key)
        
        with open(deepgram_file, "r", encoding="utf-8") as f:
            deepgram_text = f.read()
        
        with open(hume_file, "r", encoding="utf-8") as f:
            hume_text = f.read()
        
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
  [00:00:03.2] Speaker 1: Hello there. (Emotion: Sadness, Surprise)


Now produce the final merged transcript.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that merges transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        merged_transcript = response.choices[0].message.content
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(merged_transcript)
        
        print(f"  ✅ Transcripts merged successfully!")
        return merged_transcript
        
    except Exception as e:
        print(f"  ❌ Failed to merge transcripts: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_conversation(conversation_folder):
    """Process a single conversation folder to generate final transcript."""
    folder_name = os.path.basename(conversation_folder)
    print(f"\n{'='*80}")
    print(f"Processing: {folder_name}")
    print(f"{'='*80}")
    
    conversation_wav = os.path.join(conversation_folder, "conversation.wav")
    
    if not os.path.exists(conversation_wav):
        print(f"  ❌ conversation.wav not found in {conversation_folder}")
        return False
    
    transcript_file = os.path.join(conversation_folder, "timestamped_transcription.txt")
    deepgram_transcript_file = os.path.join(conversation_folder, "deepgram_transcript.txt")
    final_transcript_file = os.path.join(conversation_folder, "final_transcript.txt")
    
    # Step 1: Transcribe with Deepgram if needed
    if not os.path.exists(transcript_file):
        transcript_result = transcribe_audio(conversation_wav, transcript_file)
        if not transcript_result:
            print(f"  ❌ Transcription failed for {folder_name}")
            return False
    else:
        print(f"  ℹ️  Using existing transcription file.")
    
    # Create deepgram_transcript.txt from timestamped_transcription.txt
    with open(transcript_file, "r", encoding="utf-8") as f:
        timestamped_lines = f.readlines()
    
    formatted_deepgram = ""
    for line in timestamped_lines:
        line = line.strip()
        if line:
            parts = line.split('\t')
            if len(parts) >= 3:
                time_range = parts[0]
                speaker = parts[1]
                transcript = '\t'.join(parts[2:])
                formatted_deepgram += f"{time_range} {speaker}: {transcript}\n"
    
    with open(deepgram_transcript_file, "w", encoding="utf-8") as f:
        f.write(formatted_deepgram)
    
    # Step 2: Run Hume emotion analysis
    hume_result = hume_emotion_analysis(conversation_wav, conversation_folder)
    if not hume_result:
        print(f"  ❌ Hume emotion analysis failed for {folder_name}")
        return False
    
    # Step 3: Merge transcripts
    hume_file = os.path.join(conversation_folder, "hume_emotion.txt")
    merged_transcript = merge_transcripts(deepgram_transcript_file, hume_file, final_transcript_file)
    
    if merged_transcript:
        print(f"  ✅ Final transcript created successfully for {folder_name}!")
        return True
    else:
        print(f"  ❌ Failed to merge transcripts for {folder_name}")
        return False

def main():
    """Main function to batch process conversations."""
    base_folder = "Recording"
    
    # Process Conversation_2 through Conversation_10, excluding Conversation_6
    conversations_to_process = []
    for i in range(2, 11):
        if i != 6:  # Exclude Conversation_6
            folder_name = f"Conversation_{i}"
            conversation_folder = os.path.join(base_folder, folder_name)
            if os.path.exists(conversation_folder):
                conversations_to_process.append(conversation_folder)
            else:
                print(f"⚠️  Warning: {conversation_folder} not found, skipping...")
    
    print(f"\n🚀 Starting batch processing of {len(conversations_to_process)} conversations...")
    print(f"   Folders to process: {[os.path.basename(f) for f in conversations_to_process]}\n")
    
    results = {}
    for conversation_folder in conversations_to_process:
        folder_name = os.path.basename(conversation_folder)
        try:
            success = process_conversation(conversation_folder)
            results[folder_name] = "✅ Success" if success else "❌ Failed"
        except Exception as e:
            print(f"  ❌ Error processing {folder_name}: {e}")
            results[folder_name] = f"❌ Error: {str(e)}"
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    for folder_name, status in results.items():
        print(f"  {folder_name}: {status}")
    
    successful = sum(1 for status in results.values() if "✅" in status)
    print(f"\n✅ Successfully processed: {successful}/{len(results)} conversations")
    print(f"❌ Failed: {len(results) - successful}/{len(results)} conversations")

if __name__ == "__main__":
    main()

