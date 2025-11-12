import streamlit as st
import soundfile as sf
import librosa
import numpy as np
import os
import json
import tempfile
import asyncio
import re
import time
import base64
from pathlib import Path
from datetime import datetime
from deepgram import DeepgramClient
from dotenv import load_dotenv
from httpx import HTTPStatusError
# from speechmatics.models import ConnectionSettings
# from speechmatics.batch_client import BatchClient
# from hume import AsyncHumeClient
# from hume.expression_measurement.batch import Face, Prosody, Models
# from hume.expression_measurement.batch.types import InferenceBaseRequest
# from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Annotation and Emotion Marking", page_icon="📝", layout="wide")

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'annotation'
    
# Initialize session state for password authentication
if 'merging_page_authenticated' not in st.session_state:
    st.session_state.merging_page_authenticated = False
if 'show_password_input' not in st.session_state:
    st.session_state.show_password_input = False

# ============================================================================
# Function Definitions
# ============================================================================

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

def process_audio(first_file, second_file, folder_name):
    """Process and merge two audio files."""
    
    # Create output directory structure
    base_folder = "Recording"
    output_folder = os.path.join(base_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert both files to mono
    st.info("🔄 Converting files to mono...")
    first_data, first_sr = convert_to_mono(first_file)
    second_data, second_sr = convert_to_mono(second_file)
    
    # Save mono files
    first_mono_path = os.path.join(output_folder, "first_speaker.wav")
    second_mono_path = os.path.join(output_folder, "second_speaker.wav")
    
    sf.write(first_mono_path, first_data, first_sr, format='WAV', subtype='PCM_16')
    sf.write(second_mono_path, second_data, second_sr, format='WAV', subtype='PCM_16')
    
    # Ensure both are the same sample rate
    if first_sr != second_sr:
        st.info(f"🔄 Aligning sample rates ({first_sr} Hz vs {second_sr} Hz)...")
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
    st.info("🔄 Aligning audio durations...")
    max_len = max(len(first_data), len(second_data))
    if len(first_data) < max_len:
        first_data = np.pad(first_data, (0, max_len - len(first_data)), 'constant')
    if len(second_data) < max_len:
        second_data = np.pad(second_data, (0, max_len - len(second_data)), 'constant')
    
    # Combine into stereo
    st.info("🎵 Creating stereo output...")
    stereo = np.column_stack([first_data, second_data])
    
    # Save conversation file
    conversation_path = os.path.join(output_folder, "conversation.wav")
    sf.write(conversation_path, stereo, sample_rate)
    
    # Return file info
    duration = len(first_data) / sample_rate
    return {
        'first_mono': first_mono_path,
        'second_mono': second_mono_path,
        'conversation': conversation_path,
        'duration': duration,
        'sample_rate': sample_rate,
        'output_folder': output_folder
    }

def split_stereo_conversation(stereo_file, folder_name):
    """Split a stereo conversation file into two mono files (one per speaker)."""
    
    # Create output directory structure
    base_folder = "Recording"
    output_folder = os.path.join(base_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Load stereo audio file
    st.info("🔄 Loading stereo audio file...")
    audio_data, sample_rate = librosa.load(stereo_file, sr=None, mono=False)
    
    # Check if audio is actually stereo
    if len(audio_data.shape) == 1:
        st.error("❌ The uploaded file is mono, not stereo. Cannot split into separate speakers.")
        return None
    
    # Extract left and right channels
    # librosa returns (channels, samples), so left channel is audio_data[0], right is audio_data[1]
    left_channel = audio_data[0]  # Speaker 1
    right_channel = audio_data[1]  # Speaker 2
    
    # Save individual speaker files
    first_speaker_path = os.path.join(output_folder, "first_speaker.wav")
    second_speaker_path = os.path.join(output_folder, "second_speaker.wav")
    
    st.info("💾 Saving speaker files...")
    sf.write(first_speaker_path, left_channel, sample_rate, format='WAV', subtype='PCM_16')
    sf.write(second_speaker_path, right_channel, sample_rate, format='WAV', subtype='PCM_16')
    
    # Also save the original stereo file for reference
    conversation_path = os.path.join(output_folder, "conversation.wav")
    # Transpose to (samples, channels) for soundfile
    stereo_transposed = audio_data.T
    sf.write(conversation_path, stereo_transposed, sample_rate, format='WAV', subtype='PCM_16')
    
    # Calculate duration
    duration = len(left_channel) / sample_rate
    
    return {
        'first_speaker': first_speaker_path,
        'second_speaker': second_speaker_path,
        'conversation': conversation_path,
        'duration': duration,
        'sample_rate': sample_rate,
        'output_folder': output_folder
    }

def display_basic_results(result):
    """Display basic file information after merging."""
    # Display success message
    st.success(f"✅ Audio files processed successfully!")
    
    # Display results
    st.markdown("### 📁 Output Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**First Speaker**")
        st.info(f"✓ Mono")
    
    with col2:
        st.markdown("**Second Speaker**")
        st.info(f"✓ Mono")
    
    with col3:
        st.markdown("**Conversation**")
        st.success(f"✓ Stereo")
    
    st.markdown(f"**📂 Files saved to:** `{result['output_folder']}/`")
    
    # Show file sizes
    st.markdown("### 📊 File Information")
    file_sizes = {}
    files_to_show = [
        ('first_speaker.wav', result['first_mono']),
        ('second_speaker.wav', result['second_mono']),
        ('conversation.wav', result['conversation'])
    ]
    
    for name, path in files_to_show:
        size = os.path.getsize(path) / (1024 * 1024)
        file_sizes[name] = size
    
    df = {
        'File': list(file_sizes.keys()),
        'Size (MB)': [f"{size:.2f}" for size in file_sizes.values()]
    }
    st.table(df)

def transcribe_audio(audio_path, output_file, language="es"):
    """Transcribe audio file with speaker diarization."""
    try:
        # Get API key from environment
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            st.error("❌ DEEPGRAM_API_KEY not found in environment variables. Please set it in .env file.")
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
        
        st.write("🔍 Debug - Full response saved to:", debug_file)
        st.write("🔍 Debug - Response keys:", list(response_json.keys()) if response_json else "Empty response")
        
        # Show structure
        if 'results' in response_json:
            st.write("🔍 Results structure:")
            utt_count = response_json['results'].get('utterances')
            if utt_count is None:
                utt_count = 0
            else:
                utt_count = len(utt_count)
            
            st.json({
                "has_channels": 'channels' in response_json['results'],
                "has_utterances": 'utterances' in response_json['results'],
                "channels_count": len(response_json['results'].get('channels', [])) if 'channels' in response_json['results'] else 0,
                "utterances_count": utt_count,
            })
        
        # Extract utterances with speaker information - using same approach as transcribe.py
        utterances = []
        
        if 'results' in response_json and 'channels' in response_json['results']:
            for channel in response_json['results']['channels']:
                if 'alternatives' in channel and len(channel['alternatives']) > 0:
                    alternative = channel['alternatives'][0]
                    
                    # Use paragraphs/sentences approach (same as transcribe.py)
                    if 'paragraphs' in alternative:
                        paragraphs_data = alternative['paragraphs']
                        st.write(f"🔍 Paragraphs type: {type(paragraphs_data)}")
                        
                        # Handle both dict and list structures
                        if isinstance(paragraphs_data, dict):
                            # If it's a dict, get the 'paragraphs' list inside it
                            paragraphs_list = paragraphs_data.get('paragraphs', [])
                        elif isinstance(paragraphs_data, list):
                            paragraphs_list = paragraphs_data
                        else:
                            paragraphs_list = []
                        
                        st.write(f"🔍 Found {len(paragraphs_list)} paragraphs")
                        
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
                summary_file = output_file.replace('timestamped_transcription.txt', 'summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                
                st.write(f"✅ Summary saved to: {summary_file}")
        
        speakers = set(utt['speaker'] for utt in utterances)
        return {
            'utterances': utterances,
            'speakers': speakers,
            'count': len(utterances),
            'summary': summary_text
        }
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def speechmatics_analysis(audio_path, output_folder, language="es"):
    """Get summary and audio events from Speechmatics API."""
    try:
        # Get API key from environment
        api_key = os.getenv("SPEECHMATICS_API_KEY")
        if not api_key:
            st.error("❌ SPEECHMATICS_API_KEY not found in environment variables. Please set it in .env file.")
            return None
        
        # Define transcription parameters
        conf = {
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
                "language": language,
                "operating_point": "enhanced"
            },
            "type": "transcription"
        }
        
        # Create connection settings
        settings = ConnectionSettings(
            url="https://asr.api.speechmatics.com/v2",
            auth_token=api_key,
        )
        
        # Open the client using a context manager
        with BatchClient(settings) as client:
            job_id = client.submit_job(
                audio=audio_path,
                transcription_config=conf,
            )
            
            # Wait for completion
            transcript = client.wait_for_completion(job_id, transcription_format="json-v2")
            
            # Extract summary
            summary = transcript.get("summary", {}).get("content", "")
            
            # Extract audio events
            audio_events = transcript.get("audio_events", [])
            audio_event_summary = transcript.get("audio_event_summary", {})
            
            # Save summary to file
            summary_file = os.path.join(output_folder, "summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            # Save metadata with audio events
            metadata_file = os.path.join(output_folder, "metadata.txt")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write("=== AUDIO EVENT SUMMARY ===\n\n")
                f.write(json.dumps(audio_event_summary, indent=2, ensure_ascii=False))
                f.write("\n\n=== DETECTED AUDIO EVENTS ===\n\n")
                for event in audio_events:
                    f.write(f"{event['type']} from {event['start_time']} to {event['end_time']}, confidence: {event['confidence']}\n")
            
            return {
                'summary': summary,
                'audio_events': audio_events,
                'audio_event_summary': audio_event_summary,
                'summary_file': summary_file,
                'metadata_file': metadata_file
            }
            
    except HTTPStatusError as e:
        if e.response.status_code == 401:
            st.error("❌ Invalid SPEECHMATICS_API_KEY - Check your API key in .env file!")
        elif e.response.status_code == 403:
            error_detail = "Access Forbidden"
            try:
                error_response = e.response.json()
                error_detail = error_response.get('detail', str(error_response))
            except:
                error_detail = str(e.response.text) if hasattr(e.response, 'text') else str(e)
            
            st.error("❌ **403 Forbidden - Speechmatics API Access Denied**")
            st.warning("""
            **Possible causes:**
            1. **Invalid or expired API key** - Verify your API key is correct
            2. **Account permissions** - Your account may not have access to batch transcription
            3. **Feature not enabled** - Summarization, audio events, or other features may require account upgrade
            4. **Account credits/billing** - Check if your account has credits or active billing
            5. **API endpoint restriction** - Your account type may not support this endpoint
            
            **Troubleshooting:**
            - Check your Speechmatics account dashboard for API key validity
            - Verify account status and available credits
            - Contact Speechmatics support if the issue persists
            """)
            st.error(f"Error details: {error_detail}")
        elif e.response.status_code == 400:
            st.error(f"❌ Speechmatics API error: {e.response.json()}")
        else:
            st.error(f"❌ Speechmatics API error (Status {e.response.status_code}): {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        st.error(f"❌ Speechmatics analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_hume_predictions(job_predictions):
    """Parse Hume predictions and format them for output.
    
    Args:
        job_predictions: The predictions from the Hume API
    
    Returns:
        str: Formatted output text
    """
    # Prepare output lines
    lines = []
    
    lines.append("\n" + "="*100)
    lines.append("FORMATTED TRANSCRIPT WITH EMOTION ANALYSIS")
    lines.append("="*100)
    
    # Parse the prediction string
    pred_str = str(job_predictions[0]) if job_predictions else ""
    
    predictions = []
    
    # Find all occurrences of ProsodyPrediction(text='...')
    text_start = 'ProsodyPrediction(text='
    segments = pred_str.split(text_start)[1:]  # Get all segments after first occurrence
    
    for segment in segments:
        # Extract text - handle quotes properly
        if segment.startswith("'"):
            quote_char = "'"
        elif segment.startswith('"'):
            quote_char = '"'
        else:
            continue
        
        # Find the closing quote
        text_start_idx = 1  # Skip opening quote
        text_end_idx = text_start_idx
        while text_end_idx < len(segment) and segment[text_end_idx] != quote_char:
            text_end_idx += 1
        
        if text_end_idx >= len(segment):
            continue
        
        text = segment[text_start_idx:text_end_idx]
        
        # Extract time
        time_match = re.search(r"time=TimeInterval\(begin=([\d.]+), end=([\d.]+)\)", segment)
        if not time_match:
            continue
        begin = float(time_match.group(1))
        end = float(time_match.group(2))
        
        # Extract confidence
        conf_match = re.search(r"confidence=([\d.]+)", segment)
        confidence = float(conf_match.group(1)) if conf_match else 1.0
        
        # Extract emotions
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
        
        # Get top 3 emotions
        pred['emotions'].sort(key=lambda x: x[1], reverse=True)
        top_emotions = pred['emotions'][:3]
        emotions_text = ", ".join([f"{name} ({score:.0%})" for name, score in top_emotions])
        
        # Truncate text if too long for display
        if len(text) > 52:
            text = text[:49] + "..."
        
        lines.append(f"{time_str:<20} {text:<55} {emotions_text}")
    
    lines.append("-" * 120)
    lines.append(f"\nTotal segments: {len(predictions)}")
    lines.append("")
    
    return "\n".join(lines)

async def run_hume_analysis_async(audio_path, status_placeholder=None):
    """Run Hume API analysis on audio file (async function).
    
    Args:
        audio_path: Path to audio file
        status_placeholder: Optional Streamlit placeholder for status updates
    
    Returns:
        dict: Analysis results with predictions and job_id
    """
    # Initialize an authenticated client
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        raise ValueError("HUME_API_KEY not found in environment variables")
    
    client = AsyncHumeClient(api_key=api_key)
    
    if status_placeholder:
        status_placeholder.info("🔄 Submitting job to Hume API...")
    
    # Open audio file
    audio_file = open(audio_path, mode="rb")
    
    # Create configurations for audio analysis using Prosody model
    prosody_config = Prosody()
    models_chosen = Models(prosody=prosody_config)
    stringified_configs = InferenceBaseRequest(models=models_chosen)
    
    # Start an inference job
    job_id = await client.expression_measurement.batch.start_inference_job_from_local_file(
        json=stringified_configs, file=[audio_file]
    )
    
    audio_file.close()
    
    if status_placeholder:
        status_placeholder.info(f"✅ Job submitted! Job ID: {job_id}\n🔄 Waiting for predictions...")
    
    # Wait for predictions
    max_wait = 600  # Wait up to 10 minutes
    waited = 0
    
    while waited < max_wait:
        try:
            job_predictions = await client.expression_measurement.batch.get_job_predictions(
                id=job_id
            )
            
            # Check if we got actual predictions
            if job_predictions and len(job_predictions) > 0:
                pred_str = str(job_predictions[0])
                if 'results=InferenceResults(predictions=[]' not in pred_str:
                    if status_placeholder:
                        status_placeholder.success("✅ Predictions received!")
                    return {
                        'predictions': job_predictions,
                        'job_id': job_id
                    }
            
            # Check job status
            job_details = await client.expression_measurement.batch.get_job_details(
                id=job_id
            )
            status = str(job_details).split("status='")[1].split("'")[0] if "status='" in str(job_details) else "UNKNOWN"
            
            if status_placeholder:
                status_placeholder.info(f"Status: {status}, waiting... ({waited}s/{max_wait}s)")
            
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
    """Run Hume emotion analysis on audio file and save results.
    
    Args:
        audio_path: Path to audio file
        output_folder: Folder to save the output file
    
    Returns:
        dict: Results with output file path and formatted text
    """
    try:
        # Get API key from environment
        api_key = os.getenv("HUME_API_KEY")
        if not api_key:
            st.error("❌ HUME_API_KEY not found in environment variables. Please set it in .env file.")
            return None
        
        # Create status placeholder
        status_placeholder = st.empty()
        
        # Run async analysis - create new event loop for Streamlit
        # Streamlit doesn't run in an async context, so we create a new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_hume_analysis_async(audio_path, status_placeholder))
        finally:
            loop.close()
        
        # Parse predictions
        formatted_output = parse_hume_predictions(result['predictions'])
        
        # Add timestamp
        formatted_output += "\n" + "-" * 100 + "\n"
        formatted_output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Save to file
        output_file = os.path.join(output_folder, "hume_emotion.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        
        return {
            'output_file': output_file,
            'formatted_text': formatted_output,
            'job_id': result['job_id']
        }
        
    except Exception as e:
        st.error(f"❌ Hume emotion analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def merge_transcripts(deepgram_file, hume_file, output_file):
    """Merge Deepgram and Hume transcripts using OpenAI.
    
    Args:
        deepgram_file: Path to Deepgram transcript file
        hume_file: Path to Hume emotion transcript file
        output_file: Path to save the merged transcript
    
    Returns:
        str: Merged transcript text
    """
    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
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

- timestamps should be in the format [start,end] not [HH:MM:SS.mmm]


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
        st.error(f"❌ Failed to merge transcripts: {e}")
        import traceback
        traceback.print_exc()
        return None

# Page navigation
if st.session_state.current_page == 'annotation':
    # Annotation and Emotion Marking Page (Main Page)
    st.title("📝 Annotation and Emotion Marking")
    
    # Button to go to audio merging page
    if st.button("➡️ Go to Audio Merging Tools"):
        st.session_state.show_password_input = True
        st.rerun()
    
    # Password input for accessing merging page
    if st.session_state.show_password_input:
        st.markdown("---")
        st.markdown("### 🔒 Password Required")
        password_input = st.text_input("Enter password to access Audio Merging Tools:", type="password", key="password_input")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Submit", type="primary", use_container_width=True):
                if password_input == "Hotburger@123":
                    st.session_state.merging_page_authenticated = True
                    st.session_state.show_password_input = False
                    st.session_state.current_page = 'main'
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Access denied.")
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_password_input = False
                st.rerun()
    
    st.markdown("---")
    
    # File uploaders
    conversation_audio = st.file_uploader("Upload conversation.wav file", type=['wav', 'm4a', 'mp3'], key="annotation_audio")
    transcript_file = st.file_uploader("Upload final_transcript.txt file", type=['txt'], key="annotation_transcript")
    
    if conversation_audio and transcript_file:
        # Parse transcript file - keep original format for saving
        transcript_text = transcript_file.read().decode('utf-8')
        transcript_lines = transcript_text.strip().split('\n')
        
        # Store original lines for saving
        original_transcript_lines = [line.strip() for line in transcript_lines if line.strip()]
        
        # Parse transcript entries
        transcript_entries = []
        for line in transcript_lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse format: [00:00:00.240] Speaker 1: Text (Emotion: ...)
            # OR [13.135,13.775] Speaker 1: Text (Emotion: ...)
            # Extract timestamp - try both formats
            total_seconds = None
            end_seconds = None
            timestamp_format = None  # 'time_format' or 'range_format'
            
            # Try old format: [HH:MM:SS.mmm]
            timestamp_match = re.search(r'\[(\d{2}):(\d{2}):(\d{2}\.\d+)\]', line)
            if timestamp_match:
                hours = int(timestamp_match.group(1))
                minutes = int(timestamp_match.group(2))
                seconds = float(timestamp_match.group(3))
                total_seconds = hours * 3600 + minutes * 60 + seconds
                end_seconds = total_seconds  # For old format, end equals start
                timestamp_format = 'time_format'
            else:
                # Try new format: [start,end]
                timestamp_match = re.search(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', line)
                if timestamp_match:
                    # Use the start time (first number) as the timestamp
                    total_seconds = float(timestamp_match.group(1))
                    end_seconds = float(timestamp_match.group(2))
                    timestamp_format = 'range_format'
            
            if total_seconds is None:
                continue
            
            # Extract speaker
            speaker_match = re.search(r'Speaker \d+', line)
            speaker = speaker_match.group(0) if speaker_match else "Unknown"
            
            # Extract text (everything after speaker, before emotion if present)
            text_start = line.find(':', line.find(speaker))
            if text_start == -1:
                continue
            
            text_part = line[text_start + 1:].strip()
            # Extract emotion part if present
            emotion_match = re.search(r'\s*\(Emotion:.*?\)\s*$', text_part)
            emotion_part = ""
            emotions_list = []
            if emotion_match:
                emotion_part = emotion_match.group(0).strip()
                text_part = text_part[:emotion_match.start()].strip()
                # Extract individual emotions from format: (Emotion: Emotion1, Emotion2)
                emotion_content = emotion_part.replace('(Emotion:', '').replace(')', '').strip()
                if emotion_content:
                    # Split by comma and clean up
                    emotions_list = [e.strip() for e in emotion_content.split(',') if e.strip()]
            
            # Store original line format for reconstruction
            original_line = line
            
            transcript_entries.append({
                'time': total_seconds,
                'end_time': end_seconds,
                'timestamp_format': timestamp_format,
                'speaker': speaker,
                'text': text_part,
                'emotion': emotion_part,
                'emotions': emotions_list,  # Store as array for easier manipulation
                'original_line': original_line,
                'index': len(transcript_entries)  # Store index for matching with original lines
            })
        
        # Sort by time
        transcript_entries.sort(key=lambda x: x['time'])
        
        if transcript_entries:
            st.success(f"✅ Loaded {len(transcript_entries)} transcript entries")
            
            # Save audio file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{conversation_audio.name.split('.')[-1]}") as tmp_audio:
                tmp_audio.write(conversation_audio.read())
                tmp_audio_path = tmp_audio.name
            
            # Convert to base64 for embedding
            with open(tmp_audio_path, 'rb') as f:
                audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()
            
            # Determine audio format
            audio_ext = conversation_audio.name.split('.')[-1].lower()
            audio_format_map = {'wav': 'wav', 'mp3': 'mp3', 'm4a': 'mp4'}
            audio_format = audio_format_map.get(audio_ext, 'wav')
            
            # Create JSON data for the transcript
            transcript_json = json.dumps(transcript_entries)
            
            # Store original transcript format in session state for saving
            if 'original_transcript_data' not in st.session_state:
                st.session_state.original_transcript_data = {
                    'entries': transcript_entries,
                    'original_lines': original_transcript_lines
                }
            
            # Container for edited transcript data
            edited_transcript_container = st.container()
            
            # Create HTML component with audio player and synchronized transcript
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        padding: 20px;
                        background-color: #0e1117;
                        color: #fafafa;
                    }}
                    .audio-container {{
                        margin-bottom: 30px;
                        background-color: #1e1e1e;
                        padding: 20px;
                        border-radius: 10px;
                    }}
                    audio {{
                        width: 100%;
                        margin-bottom: 10px;
                    }}
                    .controls {{
                        display: flex;
                        gap: 10px;
                        margin-top: 10px;
                    }}
                    button {{
                        padding: 10px 20px;
                        font-size: 16px;
                        background-color: #ff4b4b;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                    }}
                    button:hover {{
                        background-color: #ff3333;
                    }}
                    .transcript-container {{
                        background-color: #1e1e1e;
                        padding: 30px;
                        border-radius: 10px;
                        min-height: 200px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-top: 20px;
                    }}
                    .transcript-display {{
                        text-align: center;
                        width: 100%;
                    }}
                    .speaker {{
                        font-size: 18px;
                        font-weight: bold;
                        color: #ff4b4b;
                        margin-bottom: 10px;
                    }}
                    .text {{
                        font-size: 24px;
                        line-height: 1.6;
                        color: #fafafa;
                        min-height: 60px;
                        padding: 10px;
                        border: 2px solid transparent;
                        border-radius: 5px;
                        cursor: text;
                        outline: none;
                    }}
                    .text:focus {{
                        border-color: #ff4b4b;
                        background-color: #2a2a2a;
                    }}
                    .text.editing {{
                        border-color: #ff4b4b;
                        background-color: #2a2a2a;
                    }}
                    .transcript-lines {{
                        display: flex;
                        flex-direction: column;
                        gap: 8px;
                        width: 100%;
                    }}
                    .transcript-line {{
                        font-size: 20px;
                        line-height: 1.6;
                        color: #aaa;
                        padding: 8px;
                        border-radius: 5px;
                        transition: all 0.3s ease;
                    }}
                    .transcript-line.current {{
                        font-weight: bold;
                        font-size: 24px;
                        color: #fafafa;
                        background-color: rgba(255, 75, 75, 0.1);
                        border-left: 3px solid #ff4b4b;
                        padding-left: 15px;
                    }}
                    .transcript-line.editable {{
                        cursor: text;
                    }}
                    .transcript-line.editable:hover {{
                        background-color: rgba(255, 255, 255, 0.05);
                    }}
                    .transcript-line.editing {{
                        border: 2px solid #ff4b4b;
                        background-color: #2a2a2a;
                        color: #fafafa;
                    }}
                    .time-display {{
                        font-size: 14px;
                        color: #888;
                        margin-top: 10px;
                    }}
                    .fade-in {{
                        animation: fadeIn 0.5s;
                    }}
                    .save-button {{
                        background-color: #4CAF50 !important;
                        margin-top: 0;
                        padding: 12px 24px !important;
                        font-size: 16px !important;
                        font-weight: bold;
                        color: white !important;
                        border: none !important;
                        border-radius: 5px;
                        cursor: pointer;
                        width: auto;
                        min-width: 150px;
                        flex-shrink: 0;
                        display: inline-block;
                        box-sizing: border-box;
                    }}
                    .save-button:hover {{
                        background-color: #45a049;
                    }}
                    .resume-button {{
                        background-color: #2196F3;
                    }}
                    .resume-button:hover {{
                        background-color: #0b7dda;
                    }}
                    .edit-notice {{
                        font-size: 12px;
                        color: #ffa500;
                        margin-top: 5px;
                        font-style: italic;
                    }}
                    .transcript-line-content {{
                        display: flex;
                        align-items: center;
                        flex-wrap: wrap;
                        gap: 8px;
                    }}
                    .transcript-text-part {{
                        flex: 1;
                        min-width: 200px;
                    }}
                    .transcript-text-part.editable {{
                        cursor: text;
                    }}
                    .transcript-text-part.editing {{
                        border: 2px solid #ff4b4b;
                        background-color: #2a2a2a;
                        color: #fafafa;
                        padding: 4px;
                        border-radius: 3px;
                    }}
                    .emotions-container {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 6px;
                        align-items: center;
                    }}
                    .emotion-box {{
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        background-color: rgba(255, 75, 75, 0.2);
                        color: #fafafa;
                        padding: 8px 12px;
                        border-radius: 6px;
                        font-size: 14px;
                        border: 1px solid rgba(255, 75, 75, 0.4);
                        min-height: 32px;
                    }}
                    .emotion-remove {{
                        cursor: pointer;
                        font-weight: bold;
                        color: #ff8888;
                        padding: 4px 6px;
                        border: none;
                        background: none;
                        font-size: 20px;
                        line-height: 1;
                        min-width: 28px;
                        min-height: 28px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border-radius: 4px;
                        transition: all 0.2s ease;
                    }}
                    .emotion-remove:hover {{
                        color: #ff4b4b;
                        background-color: rgba(255, 75, 75, 0.2);
                    }}
                    .add-emotion-btn {{
                        cursor: pointer;
                        background-color: rgba(75, 175, 80, 0.2);
                        color: #4CAF50;
                        border: 1px solid rgba(75, 175, 80, 0.4);
                        padding: 8px 12px;
                        border-radius: 6px;
                        font-size: 18px;
                        font-weight: bold;
                        min-width: 32px;
                        min-height: 32px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: all 0.2s ease;
                    }}
                    .add-emotion-btn:hover {{
                        background-color: rgba(75, 175, 80, 0.3);
                        transform: scale(1.05);
                    }}
                    .emotion-modal {{
                        display: none;
                        position: fixed;
                        z-index: 99999;
                        left: 0;
                        top: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(0, 0, 0, 0.7);
                        overflow: auto;
                    }}
                    .emotion-modal-content {{
                        background-color: #1e1e1e;
                        margin: 5% auto;
                        padding: 20px;
                        border: 1px solid #444;
                        border-radius: 10px;
                        width: 80%;
                        max-width: 600px;
                        max-height: 80vh;
                        overflow-y: auto;
                    }}
                    .emotion-modal-header {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 20px;
                    }}
                    .emotion-modal-title {{
                        font-size: 18px;
                        font-weight: bold;
                        color: #fafafa;
                    }}
                    .emotion-modal-close {{
                        cursor: pointer;
                        font-size: 24px;
                        color: #aaa;
                        border: none;
                        background: none;
                    }}
                    .emotion-modal-close:hover {{
                        color: #fafafa;
                    }}
                    .emotion-options {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                        gap: 10px;
                        margin-top: 10px;
                    }}
                    .emotion-option {{
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        padding: 8px;
                        background-color: #2a2a2a;
                        border-radius: 5px;
                        cursor: pointer;
                    }}
                    .emotion-option:hover {{
                        background-color: #333;
                    }}
                    .emotion-option input[type="checkbox"] {{
                        width: 18px;
                        height: 18px;
                        cursor: pointer;
                    }}
                    .emotion-option label {{
                        cursor: pointer;
                        color: #fafafa;
                        flex: 1;
                    }}
                    .emotion-modal-actions {{
                        margin-top: 20px;
                        display: flex;
                        justify-content: flex-end;
                        gap: 10px;
                    }}
                    .emotion-modal-btn {{
                        padding: 10px 20px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 14px;
                    }}
                    .emotion-modal-btn-primary {{
                        background-color: #4CAF50;
                        color: white;
                    }}
                    .emotion-modal-btn-primary:hover {{
                        background-color: #45a049;
                    }}
                    .emotion-modal-btn-secondary {{
                        background-color: #666;
                        color: white;
                    }}
                    .emotion-modal-btn-secondary:hover {{
                        background-color: #777;
                    }}
                    @keyframes fadeIn {{
                        from {{ opacity: 0; transform: translateY(10px); }}
                        to {{ opacity: 1; transform: translateY(0); }}
                    }}
                </style>
            </head>
            <body>
                <div class="audio-container">
                    <audio id="audioPlayer" controls>
                        <source src="data:audio/{audio_format};base64,{audio_base64}" type="audio/{audio_format}">
                        Your browser does not support the audio element.
                    </audio>
                    <div class="controls">
                        <button onclick="playAudio()">▶ Play</button>
                        <button onclick="pauseAudio()">⏸ Pause</button>
                        <button onclick="goBack5Seconds()">⏪ Go Back 5s</button>
                        <button id="speedButton" onclick="togglePlaybackSpeed()">1.0x Speed</button>
                        <button id="resumeButton" onclick="resumeAudio()" class="resume-button" style="display: none;">▶ Resume</button>
                    </div>
                </div>
                
                <div class="transcript-container">
                    <div class="transcript-display" id="transcriptDisplay">
                        <div class="speaker" id="speakerDisplay">Ready to play...</div>
                        <div class="transcript-lines" id="transcriptLines">
                            <div class="transcript-line">Click Play to start</div>
                        </div>
                        <div class="edit-notice" id="editNotice" style="display: none;">Editing mode - Audio paused. Click Resume to continue.</div>
                        <div class="time-display" id="timeDisplay">00:00:00</div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 20px; margin-bottom: 20px; position: sticky; bottom: 0; background-color: #0e1117; padding: 10px 0; z-index: 100;">
                    <button onclick="prepareSave()" class="save-button">💾 Prepare Save</button>
                </div>
                
                <!-- Emotion Modal -->
                <div id="emotionModal" class="emotion-modal">
                    <div class="emotion-modal-content">
                        <div class="emotion-modal-header">
                            <div class="emotion-modal-title">Select Emotions</div>
                            <button class="emotion-modal-close" onclick="closeEmotionModal()">&times;</button>
                        </div>
                        <div class="emotion-options" id="emotionOptions">
                            <!-- Will be populated by JavaScript -->
                        </div>
                        <div class="emotion-modal-actions">
                            <button class="emotion-modal-btn emotion-modal-btn-secondary" onclick="closeEmotionModal()">Cancel</button>
                            <button class="emotion-modal-btn emotion-modal-btn-primary" onclick="applySelectedEmotions()">Apply</button>
                        </div>
                    </div>
                </div>
                
                <!-- Transcript Copy Modal -->
                <div id="transcriptCopyModal" class="emotion-modal">
                    <div class="emotion-modal-content" style="max-width: 800px;">
                        <div class="emotion-modal-header">
                            <div class="emotion-modal-title">📋 Transcript Ready - Copy to Streamlit Text Area</div>
                            <button class="emotion-modal-close" onclick="closeTranscriptCopyModal()">&times;</button>
                        </div>
                        <div style="margin: 20px 0;">
                            <p style="color: #fafafa; margin-bottom: 15px;">✅ Transcript copied to clipboard! The text is also shown below. Click in the Streamlit text area below and press <strong>Ctrl+V</strong> (or <strong>Cmd+V</strong> on Mac) to paste.</p>
                            <textarea id="transcriptTextArea" readonly style="width: 100%; min-height: 300px; padding: 15px; background-color: #2a2a2a; color: #fafafa; border: 1px solid #444; border-radius: 5px; font-family: monospace; font-size: 12px; resize: vertical; overflow-y: auto;" placeholder="Transcript will appear here..."></textarea>
                        </div>
                        <div class="emotion-modal-actions">
                            <button class="emotion-modal-btn emotion-modal-btn-secondary" onclick="copyTranscriptAgain()">📋 Copy Again</button>
                            <button class="emotion-modal-btn emotion-modal-btn-primary" onclick="closeTranscriptCopyModal()">Got It</button>
                        </div>
                    </div>
                </div>
                
                <script>
                    const transcriptData = {transcript_json};
                    const audio = document.getElementById('audioPlayer');
                    let currentEntryIndex = 0;
                    let updateInterval = null;
                    let isEditing = false;
                    let editedEntries = {{}}; // Store edited entries by index
                    let editedEmotions = {{}}; // Store edited emotions by index
                    let wasPlayingBeforeEdit = false;
                    let playbackSpeed = 1.0; // Track current playback speed (default 1.0x)
                    let currentEmotionEntryIndex = null; // Track which entry is having emotions edited
                    
                    // Available emotions
                    const availableEmotions = [
                        'Admiration', 'Adoration', 'Amusement', 'Anger', 'Anxiety',
                        'Awe', 'Awkwardness', 'Boredom', 'Calmness', 'Concentration', 'Confusion',
                        'Contemplation', 'Contempt', 'Contentment', 'Craving', 'Desire', 'Disappointment', 'Distress', 'Determination',
                        'Disgust', 'Distress', 'Doubt', 'Ecstasy', 'Embarrassment', 'Excitement', 'Fear', 'Interest', 'Joy', 'Nostalgia', 'Relief', 'Sadness', 'Surprise'
                    ];
                    
                    // Set default playback speed to 1.0x
                    audio.playbackRate = 1.0;
                    document.getElementById('speedButton').textContent = '1.0x Speed';
                    
                    // Initialize emotion modal options
                    function initializeEmotionModal() {{
                        const optionsContainer = document.getElementById('emotionOptions');
                        optionsContainer.innerHTML = '';
                        availableEmotions.forEach(emotion => {{
                            const optionDiv = document.createElement('div');
                            optionDiv.className = 'emotion-option';
                            const checkbox = document.createElement('input');
                            checkbox.type = 'checkbox';
                            checkbox.id = 'emotion-' + emotion.replace(/\\s+/g, '-');
                            checkbox.value = emotion;
                            const label = document.createElement('label');
                            label.htmlFor = checkbox.id;
                            label.textContent = emotion;
                            optionDiv.appendChild(checkbox);
                            optionDiv.appendChild(label);
                            optionsContainer.appendChild(optionDiv);
                        }});
                    }}
                    
                    // Initialize on load
                    initializeEmotionModal();
                    
                    // Wire up modal buttons after DOM is ready
                    document.addEventListener('DOMContentLoaded', function() {{
                        const closeBtn = document.querySelector('.emotion-modal-close');
                        if (closeBtn) {{
                            closeBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                closeEmotionModal();
                            }});
                        }}
                        
                        const cancelBtn = document.querySelector('.emotion-modal-btn-secondary');
                        if (cancelBtn) {{
                            cancelBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                closeEmotionModal();
                            }});
                        }}
                        
                        const applyBtn = document.querySelector('.emotion-modal-btn-primary');
                        if (applyBtn) {{
                            applyBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                applySelectedEmotions();
                            }});
                        }}
                    }});
                    
                    // Also try to wire up immediately (in case DOM is already loaded)
                    setTimeout(function() {{
                        const closeBtn = document.querySelector('.emotion-modal-close');
                        if (closeBtn && !closeBtn.dataset.wired) {{
                            closeBtn.dataset.wired = 'true';
                            closeBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                closeEmotionModal();
                            }});
                        }}
                        
                        const cancelBtn = document.querySelector('.emotion-modal-btn-secondary');
                        if (cancelBtn && !cancelBtn.dataset.wired) {{
                            cancelBtn.dataset.wired = 'true';
                            cancelBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                closeEmotionModal();
                            }});
                        }}
                        
                        const applyBtn = document.querySelector('.emotion-modal-btn-primary');
                        if (applyBtn && !applyBtn.dataset.wired) {{
                            applyBtn.dataset.wired = 'true';
                            applyBtn.addEventListener('click', function(e) {{
                                e.preventDefault();
                                e.stopPropagation();
                                applySelectedEmotions();
                            }});
                        }}
                    }}, 100);
                    
                    function formatTime(seconds) {{
                        const h = Math.floor(seconds / 3600);
                        const m = Math.floor((seconds % 3600) / 60);
                        const s = Math.floor(seconds % 60);
                        const ms = Math.floor((seconds % 1) * 100);
                        return `${{h.toString().padStart(2, '0')}}:${{m.toString().padStart(2, '0')}}:${{s.toString().padStart(2, '0')}}.${{ms.toString().padStart(2, '0')}}`;
                    }}
                    
                    function formatTimeForLine(seconds) {{
                        const h = Math.floor(seconds / 3600);
                        const m = Math.floor((seconds % 3600) / 60);
                        const s = seconds % 60;
                        return `${{h.toString().padStart(2, '0')}}:${{m.toString().padStart(2, '0')}}:${{s.toFixed(3).padStart(6, '0')}}`;
                    }}
                    
                    function updateTranscript(forceUpdate = false) {{
                        if (isEditing && !forceUpdate) {{
                            // Don't update while editing - but keep the current display
                            // Unless forceUpdate is true (for emotion changes)
                            return;
                        }}
                        
                        const currentTime = audio.currentTime;
                        const timeDisplay = document.getElementById('timeDisplay');
                        timeDisplay.textContent = formatTime(currentTime);
                        
                        // Find the current entry based on time
                        let activeEntry = null;
                        let newEntryIndex = currentEntryIndex; // Preserve current index if force updating
                        
                        if (!forceUpdate) {{
                            // Only update entry index if not forcing (normal time-based update)
                            for (let i = transcriptData.length - 1; i >= 0; i--) {{
                                if (currentTime >= transcriptData[i].time) {{
                                    activeEntry = transcriptData[i];
                                    newEntryIndex = i;
                                    break;
                                }}
                            }}
                            
                            // If we're past the last entry, show the last one
                            if (!activeEntry && transcriptData.length > 0) {{
                                activeEntry = transcriptData[transcriptData.length - 1];
                                newEntryIndex = transcriptData.length - 1;
                            }}
                            
                            // If we're before the first entry, show nothing or first entry
                            if (!activeEntry && transcriptData.length > 0 && currentTime < transcriptData[0].time) {{
                                document.getElementById('speakerDisplay').textContent = 'Waiting...';
                                const transcriptLinesContainer = document.getElementById('transcriptLines');
                                transcriptLinesContainer.innerHTML = '<div class="transcript-line">Audio will start soon</div>';
                                return;
                            }}
                            
                            currentEntryIndex = newEntryIndex;
                        }} else {{
                            // When forcing update, use current entry index
                            if (currentEntryIndex >= 0 && currentEntryIndex < transcriptData.length) {{
                                activeEntry = transcriptData[currentEntryIndex];
                            }}
                        }}
                        
                        // If we still don't have an active entry, use first one
                        if (!activeEntry && transcriptData.length > 0) {{
                            activeEntry = transcriptData[0];
                            currentEntryIndex = 0;
                        }}
                        
                        if (activeEntry) {{
                            const speakerDisplay = document.getElementById('speakerDisplay');
                            const transcriptLinesContainer = document.getElementById('transcriptLines');
                            
                            // Show speaker name
                            speakerDisplay.textContent = activeEntry.speaker;
                            
                            // Get previous 2, current, and next 2 entries
                            const startIndex = Math.max(0, currentEntryIndex - 2);
                            const endIndex = Math.min(transcriptData.length - 1, currentEntryIndex + 2);
                            
                            // Clear previous lines
                            transcriptLinesContainer.innerHTML = '';
                            
                            // Create lines for visible entries
                            for (let i = startIndex; i <= endIndex; i++) {{
                                const entry = transcriptData[i];
                                const isCurrent = (i === currentEntryIndex);
                                
                                // Use edited text if available, otherwise use original
                                const displayText = editedEntries[i] || entry.text;
                                
                                // Get emotions for this entry (use edited if available, otherwise original)
                                let entryEmotions = editedEmotions[i];
                                if (entryEmotions === undefined) {{
                                    entryEmotions = entry.emotions || [];
                                }}
                                
                                // Create line element
                                const lineDiv = document.createElement('div');
                                lineDiv.className = 'transcript-line' + (isCurrent ? ' current' : '');
                                lineDiv.setAttribute('data-index', i);
                                
                                // Create content container
                                const contentDiv = document.createElement('div');
                                contentDiv.className = 'transcript-line-content';
                                
                                // Create text part
                                const textPart = document.createElement('div');
                                textPart.className = 'transcript-text-part';
                                
                                if (isCurrent) {{
                                    // Make current line editable
                                    textPart.contentEditable = 'true';
                                    textPart.classList.add('editable');
                                    
                                    // Handle focus event to start editing mode
                                    textPart.addEventListener('focus', function() {{
                                        if (!isEditing) {{
                                            onTextClick(i);
                                        }}
                                    }}, true);
                                    
                                    // Handle blur event to save edits
                                    textPart.addEventListener('blur', function() {{
                                        onTextBlur(i);
                                    }});
                                    
                                    // Handle click to ensure editing starts
                                    textPart.addEventListener('click', function(e) {{
                                        if (!isEditing) {{
                                            onTextClick(i);
                                        }}
                                    }});
                                }}
                                
                                textPart.textContent = entry.speaker + ': ' + displayText;
                                
                                // Create emotions container
                                const emotionsContainer = document.createElement('div');
                                emotionsContainer.className = 'emotions-container';
                                // Prevent clicks on emotion container from triggering text editing
                                emotionsContainer.addEventListener('click', function(e) {{
                                    e.stopPropagation();
                                }});
                                
                                // Add emotion boxes
                                entryEmotions.forEach(emotion => {{
                                    const emotionBox = document.createElement('div');
                                    emotionBox.className = 'emotion-box';
                                    
                                    const emotionText = document.createElement('span');
                                    emotionText.textContent = emotion;
                                    
                                    const removeBtn = document.createElement('button');
                                    removeBtn.className = 'emotion-remove';
                                    removeBtn.textContent = '×';
                                    removeBtn.type = 'button'; // Prevent form submission
                                    removeBtn.addEventListener('click', function(e) {{
                                        e.preventDefault();
                                        e.stopPropagation();
                                        removeEmotion(i, emotion);
                                    }});
                                    
                                    emotionBox.appendChild(emotionText);
                                    emotionBox.appendChild(removeBtn);
                                    emotionsContainer.appendChild(emotionBox);
                                }});
                                
                                // Add + button to add emotions
                                const addEmotionBtn = document.createElement('button');
                                addEmotionBtn.className = 'add-emotion-btn';
                                addEmotionBtn.textContent = '+';
                                addEmotionBtn.type = 'button'; // Prevent form submission
                                addEmotionBtn.addEventListener('click', function(e) {{
                                    e.preventDefault();
                                    e.stopPropagation();
                                    openEmotionModal(i);
                                }});
                                emotionsContainer.appendChild(addEmotionBtn);
                                
                                // Assemble the line
                                contentDiv.appendChild(textPart);
                                contentDiv.appendChild(emotionsContainer);
                                lineDiv.appendChild(contentDiv);
                                
                                transcriptLinesContainer.appendChild(lineDiv);
                            }}
                        }}
                    }}
                    
                    function onTextClick(entryIndex) {{
                        // Only allow editing the current line
                        if (entryIndex !== currentEntryIndex) {{
                            return;
                        }}
                        
                        if (!isEditing) {{
                            isEditing = true;
                            wasPlayingBeforeEdit = !audio.paused;
                            
                            // Pause audio
                            if (wasPlayingBeforeEdit) {{
                                audio.pause();
                            }}
                            
                            const editNotice = document.getElementById('editNotice');
                            const resumeButton = document.getElementById('resumeButton');
                            
                            // Find and highlight the editable text part
                            const textParts = document.querySelectorAll('.transcript-text-part.editable');
                            textParts.forEach(textPart => {{
                                const lineDiv = textPart.closest('.transcript-line');
                                if (lineDiv) {{
                                    const lineIndex = parseInt(lineDiv.getAttribute('data-index'));
                                    if (lineIndex === entryIndex) {{
                                        textPart.classList.add('editing');
                                    }}
                                }}
                            }});
                            
                            editNotice.style.display = 'block';
                            resumeButton.style.display = 'inline-block';
                        }}
                    }}
                    
                    function onTextBlur(entryIndex) {{
                        // Save the edited text
                        const textParts = document.querySelectorAll('.transcript-text-part.editable');
                        textParts.forEach(textPart => {{
                            const lineDiv = textPart.closest('.transcript-line');
                            if (lineDiv) {{
                                const lineIndex = parseInt(lineDiv.getAttribute('data-index'));
                                if (lineIndex === entryIndex) {{
                                    const editedText = textPart.textContent.trim();
                                    // Remove speaker prefix if present
                                    const speakerPrefix = transcriptData[entryIndex].speaker + ': ';
                                    const cleanText = editedText.startsWith(speakerPrefix) 
                                        ? editedText.substring(speakerPrefix.length) 
                                        : editedText;
                                    
                                    if (cleanText && cleanText !== transcriptData[entryIndex].text) {{
                                        editedEntries[entryIndex] = cleanText;
                                    }}
                                    textPart.classList.remove('editing');
                                }}
                            }}
                        }});
                    }}
                    
                    function openEmotionModal(entryIndex) {{
                        console.log('Opening emotion modal for entry:', entryIndex);
                        currentEmotionEntryIndex = entryIndex;
                        
                        // Get current emotions for this entry
                        let currentEmotions = editedEmotions[entryIndex];
                        if (currentEmotions === undefined) {{
                            currentEmotions = transcriptData[entryIndex].emotions || [];
                        }}
                        
                        console.log('Current emotions:', currentEmotions);
                        
                        // Check the checkboxes for current emotions
                        const checkboxes = document.querySelectorAll('#emotionOptions input[type="checkbox"]');
                        checkboxes.forEach(checkbox => {{
                            checkbox.checked = currentEmotions.includes(checkbox.value);
                        }});
                        
                        // Show modal
                        const modal = document.getElementById('emotionModal');
                        if (modal) {{
                            modal.style.display = 'block';
                            console.log('Modal displayed');
                        }} else {{
                            console.error('Modal element not found!');
                        }}
                    }}
                    
                    function closeEmotionModal() {{
                        const modal = document.getElementById('emotionModal');
                        if (modal) {{
                            modal.style.display = 'none';
                        }}
                        currentEmotionEntryIndex = null;
                    }}
                    
                    function applySelectedEmotions() {{
                        if (currentEmotionEntryIndex === null) {{
                            console.log('No entry index set');
                            return;
                        }}
                        
                        // Get selected emotions
                        const checkboxes = document.querySelectorAll('#emotionOptions input[type="checkbox"]:checked');
                        const selectedEmotions = Array.from(checkboxes).map(cb => cb.value);
                        
                        console.log('Applying emotions:', selectedEmotions, 'to entry:', currentEmotionEntryIndex);
                        
                        // Save to editedEmotions
                        editedEmotions[currentEmotionEntryIndex] = selectedEmotions;
                        
                        // Force update display immediately
                        updateTranscript(true);
                        
                        // Close modal
                        closeEmotionModal();
                    }}
                    
                    function removeEmotion(entryIndex, emotionToRemove) {{
                        console.log('Removing emotion:', emotionToRemove, 'from entry:', entryIndex);
                        
                        // Get current emotions for this entry
                        let currentEmotions = editedEmotions[entryIndex];
                        if (currentEmotions === undefined) {{
                            currentEmotions = [...(transcriptData[entryIndex].emotions || [])];
                        }} else {{
                            currentEmotions = [...currentEmotions];
                        }}
                        
                        console.log('Current emotions before removal:', currentEmotions);
                        
                        // Remove the emotion
                        currentEmotions = currentEmotions.filter(e => e !== emotionToRemove);
                        
                        console.log('Current emotions after removal:', currentEmotions);
                        
                        // Save back
                        editedEmotions[entryIndex] = currentEmotions;
                        
                        // Force update display immediately
                        updateTranscript(true);
                    }}
                    
                    
                    function resumeAudio() {{
                        isEditing = false;
                        
                        const editNotice = document.getElementById('editNotice');
                        const resumeButton = document.getElementById('resumeButton');
                        
                        // Remove editing class from all text parts
                        const editingTextParts = document.querySelectorAll('.transcript-text-part.editing');
                        editingTextParts.forEach(textPart => {{
                            const lineDiv = textPart.closest('.transcript-line');
                            if (lineDiv) {{
                                const entryIndex = parseInt(lineDiv.getAttribute('data-index'));
                                if (entryIndex !== null) {{
                                    onTextBlur(entryIndex);
                                }}
                            }}
                        }});
                        
                        editNotice.style.display = 'none';
                        resumeButton.style.display = 'none';
                        
                        // Resume playback if it was playing before
                        if (wasPlayingBeforeEdit) {{
                            audio.play();
                        }}
                    }}
                    
                    function playAudio() {{
                        if (isEditing) {{
                            resumeAudio();
                        }}
                        audio.play();
                        if (!updateInterval) {{
                            updateInterval = setInterval(updateTranscript, 100);
                        }}
                        updateTranscript();
                    }}
                    
                    function pauseAudio() {{
                        audio.pause();
                    }}
                    
                    function stopAudio() {{
                        audio.pause();
                        audio.currentTime = 0;
                        currentEntryIndex = 0;
                        isEditing = false;
                        document.getElementById('speakerDisplay').textContent = 'Stopped';
                        const transcriptLinesContainer = document.getElementById('transcriptLines');
                        transcriptLinesContainer.innerHTML = '<div class="transcript-line">Click Play to start</div>';
                        document.getElementById('editNotice').style.display = 'none';
                        document.getElementById('resumeButton').style.display = 'none';
                        document.getElementById('timeDisplay').textContent = '00:00:00';
                    }}
                    
                    function goBack5Seconds() {{
                        // Go back 5 seconds, but don't go below 0
                        const newTime = Math.max(0, audio.currentTime - 5);
                        audio.currentTime = newTime;
                        
                        // Immediately update transcript to show what was 5 seconds ago
                        updateTranscript();
                    }}
                    
                    function togglePlaybackSpeed() {{
                        // Toggle between 1.0x (default) and 0.75x speed
                        if (playbackSpeed === 0.75) {{
                            playbackSpeed = 1.0;
                            audio.playbackRate = 1.0;
                            document.getElementById('speedButton').textContent = '1.0x Speed';
                        }} else {{
                            playbackSpeed = 0.75;
                            audio.playbackRate = 0.75;
                            document.getElementById('speedButton').textContent = '0.75x Speed';
                        }}
                    }}
                    
                    function prepareSave() {{
                        // Build the edited transcript in the original format
                        let editedLines = [];
                        
                        for (let i = 0; i < transcriptData.length; i++) {{
                            const entry = transcriptData[i];
                            const editedText = editedEntries[i] || entry.text;
                            
                            // Get emotions (use edited if available, otherwise original)
                            let entryEmotions = editedEmotions[i];
                            if (entryEmotions === undefined) {{
                                entryEmotions = entry.emotions || [];
                            }}
                            
                            // Reconstruct the line in original format
                            let timestamp;
                            if (entry.timestamp_format === 'range_format') {{
                                // Use [start,end] format
                                timestamp = `${{entry.time.toFixed(3)}},${{entry.end_time.toFixed(3)}}`;
                            }} else {{
                                // Use [HH:MM:SS.mmm] format
                                timestamp = formatTimeForLine(entry.time);
                            }}
                            
                            // Format emotions as (Emotion: Emotion1, Emotion2) or empty string
                            let emotionPart = '';
                            if (entryEmotions.length > 0) {{
                                emotionPart = ' (Emotion: ' + entryEmotions.join(', ') + ')';
                            }}
                            
                            const line = `[${{timestamp}}] ${{entry.speaker}}: ${{editedText}}${{emotionPart}}`;
                            
                            editedLines.push(line);
                        }}
                        
                        const transcriptText = editedLines.join('\\n');
                        
                        // Populate the modal textarea
                        const textarea = document.getElementById('transcriptTextArea');
                        if (textarea) {{
                            textarea.value = transcriptText;
                        }}
                        
                        // Copy to clipboard automatically
                        navigator.clipboard.writeText(transcriptText).then(function() {{
                            // Show the modal
                            const modal = document.getElementById('transcriptCopyModal');
                            if (modal) {{
                                modal.style.display = 'block';
                                // Auto-select text in textarea for easy copying
                                if (textarea) {{
                                    textarea.select();
                                    textarea.setSelectionRange(0, transcriptText.length);
                                }}
                            }}
                        }}, function(err) {{
                            // Fallback if clipboard API fails - still show modal
                            const modal = document.getElementById('transcriptCopyModal');
                            if (modal) {{
                                modal.style.display = 'block';
                                if (textarea) {{
                                    textarea.select();
                                    textarea.setSelectionRange(0, transcriptText.length);
                                }}
                            }}
                        }});
                    }}
                    
                    function closeTranscriptCopyModal() {{
                        const modal = document.getElementById('transcriptCopyModal');
                        if (modal) {{
                            modal.style.display = 'none';
                        }}
                    }}
                    
                    function copyTranscriptAgain() {{
                        const textarea = document.getElementById('transcriptTextArea');
                        if (textarea) {{
                            const text = textarea.value;
                            navigator.clipboard.writeText(text).then(function() {{
                                alert('✅ Transcript copied to clipboard again!');
                            }}, function(err) {{
                                // Select text as fallback
                                textarea.select();
                                textarea.setSelectionRange(0, text.length);
                                alert('Please copy the text manually (Ctrl+C or Cmd+C)');
                            }});
                        }}
                    }}
                    
                    // Close transcript modal when clicking outside
                    window.onclick = function(event) {{
                        const emotionModal = document.getElementById('emotionModal');
                        const transcriptModal = document.getElementById('transcriptCopyModal');
                        if (event.target === emotionModal) {{
                            closeEmotionModal();
                        }}
                        if (event.target === transcriptModal) {{
                            closeTranscriptCopyModal();
                        }}
                    }}
                    
                    // Update transcript when audio time changes
                    audio.addEventListener('timeupdate', updateTranscript);
                    
                    // Initialize
                    updateTranscript();
                </script>
            </body>
            </html>
            """
            
            # Display the HTML component
            html_component = st.components.v1.html(html_content, height=700)
            
            # Handle saving the edited transcript
            # Text area for editing the transcript
            edited_text = st.text_area("Edited transcript:", height=300, key="manual_edit", 
                                      placeholder="Click 'Prepare Save' above and paste here (Ctrl+V or Cmd+V), then edit and download below...")
            
            # Download button - will trigger file save dialog
            if edited_text and edited_text.strip():
                st.download_button(
                    label="💾 Download Annotation",
                    data=edited_text.strip(),
                    file_name="human_annotated_transcript.txt",
                    mime="text/plain",
                    type="primary",
                    use_container_width=True
                )
            else:
                st.info("💡 Paste the edited transcript above to enable download.")
            
            
            # Cleanup
            if os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)
        else:
            st.error("❌ Could not parse transcript file. Please check the format.")
    
    elif conversation_audio or transcript_file:
        st.warning("⚠️ Please upload both the audio file and transcript file to proceed.")
    
else:
    # Main Page (Audio Merging Tools)
    # Check authentication before allowing access
    if not st.session_state.merging_page_authenticated:
        st.error("🔒 Access Denied: Authentication required")
        st.info("Please use the password-protected access from the Annotation page.")
        if st.button("← Back to Annotation Page", use_container_width=True):
            st.session_state.current_page = 'annotation'
            st.rerun()
        st.stop()
    
    st.title("🎙️ Audio Conversation Merger")
    st.markdown("Upload two audio files (.m4a or .wav) to create a stereo conversation file")
    
    # Button to go back to annotation page (main page)
    if st.button("← Back to Annotation Page", use_container_width=True):
        st.session_state.current_page = 'annotation'
        st.rerun()
    
    st.markdown("---")
    
    # File uploaders
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("First Speaker")
        first_file = st.file_uploader("Upload first speaker audio", type=['m4a', 'wav', 'mp3'], key="first")

    with col2:
        st.subheader("Second Speaker")
        second_file = st.file_uploader("Upload second speaker audio", type=['m4a', 'wav', 'mp3'], key="second")

    # Folder name input
    st.markdown("---")
    folder_name = st.text_input("Enter folder name for this recording:", placeholder="e.g., Meeting_2024_10_29")

    # Merge buttons
    col1, col2 = st.columns(2)
    with col1:
        merge_button = st.button("🎵 Merge Audio", type="primary", use_container_width=True)
    with col2:
        just_merge_button = st.button("⚡ Just Merge", use_container_width=True)

    # Separate section for splitting stereo conversation
    st.markdown("---")
    st.markdown("### 🔊 Split Stereo Conversation")
    st.markdown("Upload a stereo conversation file to separate it into individual speaker files (left channel = speaker 1, right channel = speaker 2)")

    split_conversation_file = st.file_uploader("Upload stereo conversation audio", type=['m4a', 'wav', 'mp3'], key="split_conversation")
    split_folder_name = st.text_input("Enter folder name for split files:", placeholder="e.g., Meeting_2024_10_29", key="split_folder")

    split_button = st.button("🔊 Create Separate Conversations", type="primary", use_container_width=True)

    # Separate section for transcribing existing conversation
    st.markdown("---")
    st.markdown("### 📝 Or Transcribe an Existing Conversation")
    st.markdown("Upload a stereo conversation file to transcribe it (no merging needed)")

    conversation_file = st.file_uploader("Upload conversation audio", type=['m4a', 'wav', 'mp3'], key="conversation")
    transcribe_folder_name = st.text_input("Enter folder name for transcription:", placeholder="e.g., Meeting_2024_10_29", key="transcribe_folder")

    transcribe_button = st.button("📝 Transcribe the Final Conversation", type="primary", use_container_width=True)

    # Process audio when "Just Merge" button is clicked
    if just_merge_button:
        # Validation
        if not first_file or not second_file:
            st.error("❌ Please upload both audio files")
        elif not folder_name:
            st.error("❌ Please enter a folder name")
        else:
            try:
                # Create temporary files for uploaded audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{first_file.name}") as tmp_first:
                    tmp_first.write(first_file.read())
                    tmp_first_path = tmp_first.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{second_file.name}") as tmp_second:
                    tmp_second.write(second_file.read())
                    tmp_second_path = tmp_second.name
                
                # Process audio
                result = process_audio(tmp_first_path, tmp_second_path, folder_name)
                
                # Clean up temp files
                os.unlink(tmp_first_path)
                os.unlink(tmp_second_path)
                
                # Display basic results (no transcription/analysis)
                display_basic_results(result)
                
            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")
                st.exception(e)

    # Process audio when button is clicked
    if merge_button:
        # Validation
        if not first_file or not second_file:
            st.error("❌ Please upload both audio files")
        elif not folder_name:
            st.error("❌ Please enter a folder name")
        else:
            try:
                # Create temporary files for uploaded audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{first_file.name}") as tmp_first:
                    tmp_first.write(first_file.read())
                    tmp_first_path = tmp_first.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{second_file.name}") as tmp_second:
                    tmp_second.write(second_file.read())
                    tmp_second_path = tmp_second.name
                
                # Process audio
                result = process_audio(tmp_first_path, tmp_second_path, folder_name)
                
                # Clean up temp files
                os.unlink(tmp_first_path)
                os.unlink(tmp_second_path)
                
                # Display basic results
                display_basic_results(result)
                
                # Transcribe the conversation
                api_key = os.getenv("DEEPGRAM_API_KEY")
                if api_key:
                    st.markdown("---")
                    st.markdown("### 🎙️ Transcription")
                    
                    transcript_file = os.path.join(result['output_folder'], "timestamped_transcription.txt")
                    
                    with st.spinner("Transcribing audio... This may take a moment."):
                        transcript_result = transcribe_audio(result['conversation'], transcript_file)
                    
                    if transcript_result:
                        st.success(f"✅ Transcription completed!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Utterances", transcript_result['count'])
                        with col2:
                            st.metric("Speakers Detected", len(transcript_result['speakers']))
                        with col3:
                            if transcript_result.get('summary'):
                                st.metric("Summary", "✅ Generated")
                        
                        st.markdown(f"**📝 Transcription saved to:** `{transcript_file}`")
                        if transcript_result.get('summary'):
                            summary_file = transcript_file.replace('timestamped_transcription.txt', 'summary.txt')
                            st.markdown(f"**📄 Summary saved to:** `{summary_file}`")
                        
                        # Show preview
                        st.markdown("**Preview:**")
                        preview_text = ""
                        for i, utt in enumerate(transcript_result['utterances'][:5]):
                            preview_text += f"[{utt['start']:.1f}s] {utt['speaker']}: {utt['transcript'][:60]}...\n"
                        
                        st.text(preview_text)
                        
                        # Show summary if available
                        if transcript_result.get('summary'):
                            with st.expander("📄 Conversation Summary"):
                                st.write(transcript_result['summary'])
                else:
                    st.warning("⚠️ DEEPGRAM_API_KEY not set. Transcription skipped.")
                
                # Run Speechmatics analysis
                speechmatics_api_key = os.getenv("SPEECHMATICS_API_KEY")
                if speechmatics_api_key:
                    st.markdown("---")
                    st.markdown("### 🎤 Speechmatics Analysis")
                    
                    with st.spinner("Running Speechmatics analysis for summary and audio events... This may take a moment."):
                        speechmatics_result = speechmatics_analysis(result['conversation'], result['output_folder'])
                    
                    if speechmatics_result:
                        st.success("✅ Speechmatics analysis completed!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Summary Generated", "✅")
                        with col2:
                            st.metric("Audio Events Detected", len(speechmatics_result['audio_events']))
                        
                        st.markdown(f"**📄 Summary saved to:** `{speechmatics_result['summary_file']}`")
                        st.markdown(f"**📊 Metadata saved to:** `{speechmatics_result['metadata_file']}`")
                        
                        # Show summary
                        with st.expander("📄 Speechmatics Summary"):
                            st.write(speechmatics_result['summary'])
                        
                        # Show audio events if any
                        if speechmatics_result['audio_events']:
                            with st.expander("🎵 Detected Audio Events"):
                                for event in speechmatics_result['audio_events']:
                                    emoji = "😂" if event['type'] == "laughter" else "🎵" if event['type'] == "music" else "👏"
                                    st.write(f"{emoji} **{event['type'].title()}** from {event['start_time']}s to {event['end_time']}s (confidence: {event['confidence']:.2f})")
                else:
                    st.warning("⚠️ SPEECHMATICS_API_KEY not set. Speechmatics analysis skipped.")
                
            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")
                st.exception(e)

    # Split stereo conversation when button is clicked
    if split_button:
        # Validation
        if not split_conversation_file:
            st.error("❌ Please upload a stereo conversation audio file")
        elif not split_folder_name:
            st.error("❌ Please enter a folder name for split files")
        else:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{split_conversation_file.name}") as tmp_audio:
                    tmp_audio.write(split_conversation_file.read())
                    tmp_audio_path = tmp_audio.name
                
                # Split the stereo conversation
                result = split_stereo_conversation(tmp_audio_path, split_folder_name)
                
                # Clean up temp file
                os.unlink(tmp_audio_path)
                
                if result:
                    st.success(f"✅ Conversation split successfully!")
                    
                    # Display results
                    st.markdown("### 📁 Output Files")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**First Speaker**")
                        st.info(f"✓ Mono (Left Channel)")
                    
                    with col2:
                        st.markdown("**Second Speaker**")
                        st.info(f"✓ Mono (Right Channel)")
                    
                    with col3:
                        st.markdown("**Original Stereo**")
                        st.success(f"✓ Preserved")
                    
                    st.markdown(f"**📂 Files saved to:** `{result['output_folder']}/`")
                    
                    # Show file sizes
                    st.markdown("### 📊 File Information")
                    file_sizes = {}
                    files_to_show = [
                        ('first_speaker.wav', result['first_speaker']),
                        ('second_speaker.wav', result['second_speaker']),
                        ('conversation.wav', result['conversation'])
                    ]
                    
                    for name, path in files_to_show:
                        size = os.path.getsize(path) / (1024 * 1024)
                        file_sizes[name] = size
                    
                    df = {
                        'File': list(file_sizes.keys()),
                        'Size (MB)': [f"{size:.2f}" for size in file_sizes.values()]
                    }
                    st.table(df)
                    
                    st.markdown(f"**Duration:** {result['duration']:.2f} seconds")
                    st.markdown(f"**Sample Rate:** {result['sample_rate']} Hz")
                
            except Exception as e:
                st.error(f"❌ Error splitting conversation: {str(e)}")
                st.exception(e)

    # Transcribe existing conversation when button is clicked
    if transcribe_button:
        # Validation
        if not conversation_file:
            st.error("❌ Please upload a conversation audio file")
        elif not transcribe_folder_name:
            st.error("❌ Please enter a folder name for transcription")
        else:
            try:
                # Create output directory structure
                base_folder = "Recording"
                output_folder = os.path.join(base_folder, transcribe_folder_name)
                os.makedirs(output_folder, exist_ok=True)
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{conversation_file.name}") as tmp_audio:
                    tmp_audio.write(conversation_file.read())
                    tmp_audio_path = tmp_audio.name
                
                # Optionally save the conversation file to the output folder
                conversation_path = os.path.join(output_folder, "conversation.wav")
                
                # Convert to WAV if needed and save to output folder
                audio_data, sample_rate = librosa.load(tmp_audio_path, sr=None, mono=False)
                
                # Ensure audio_data is in the correct format (samples x channels)
                if len(audio_data.shape) == 1:
                    # Mono audio
                    sf.write(conversation_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                else:
                    # Stereo or multi-channel: librosa returns (channels, samples), need (samples, channels)
                    audio_data = audio_data.T
                    sf.write(conversation_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                
                # Clean up temp file
                os.unlink(tmp_audio_path)
                
                st.success(f"✅ Conversation file saved to: `{conversation_path}`")
                
                # Transcribe the conversation
                api_key = os.getenv("DEEPGRAM_API_KEY")
                if api_key:
                    st.markdown("---")
                    st.markdown("### 🎙️ Transcription")
                    
                    transcript_file = os.path.join(output_folder, "timestamped_transcription.txt")
                    
                    with st.spinner("Transcribing audio... This may take a moment."):
                        transcript_result = transcribe_audio(conversation_path, transcript_file)
                    
                    if transcript_result:
                        st.success(f"✅ Transcription completed!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Utterances", transcript_result['count'])
                        with col2:
                            st.metric("Speakers Detected", len(transcript_result['speakers']))
                        with col3:
                            if transcript_result.get('summary'):
                                st.metric("Summary", "✅ Generated")
                        
                        st.markdown(f"**📝 Transcription saved to:** `{transcript_file}`")
                        if transcript_result.get('summary'):
                            summary_file = transcript_file.replace('timestamped_transcription.txt', 'summary.txt')
                            st.markdown(f"**📄 Summary saved to:** `{summary_file}`")
                        
                        # Show preview
                        st.markdown("**Preview:**")
                        preview_text = ""
                        for i, utt in enumerate(transcript_result['utterances'][:5]):
                            preview_text += f"[{utt['start']:.1f}s] {utt['speaker']}: {utt['transcript'][:60]}...\n"
                        
                        st.text(preview_text)
                        
                        # Show summary if available
                        if transcript_result.get('summary'):
                            with st.expander("📄 Conversation Summary"):
                                st.write(transcript_result['summary'])
                else:
                    st.warning("⚠️ DEEPGRAM_API_KEY not set. Transcription skipped.")
                
                # Run Speechmatics analysis
                speechmatics_api_key = os.getenv("SPEECHMATICS_API_KEY")
                if speechmatics_api_key:
                    st.markdown("---")
                    st.markdown("### 🎤 Speechmatics Analysis")
                    
                    with st.spinner("Running Speechmatics analysis for summary and audio events... This may take a moment."):
                        speechmatics_result = speechmatics_analysis(conversation_path, output_folder)
                    
                    if speechmatics_result:
                        st.success("✅ Speechmatics analysis completed!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Summary Generated", "✅")
                        with col2:
                            st.metric("Audio Events Detected", len(speechmatics_result['audio_events']))
                        
                        st.markdown(f"**📄 Summary saved to:** `{speechmatics_result['summary_file']}`")
                        st.markdown(f"**📊 Metadata saved to:** `{speechmatics_result['metadata_file']}`")
                        
                        # Show summary
                        with st.expander("📄 Speechmatics Summary"):
                            st.write(speechmatics_result['summary'])
                        
                        # Show audio events if any
                        if speechmatics_result['audio_events']:
                            with st.expander("🎵 Detected Audio Events"):
                                for event in speechmatics_result['audio_events']:
                                    emoji = "😂" if event['type'] == "laughter" else "🎵" if event['type'] == "music" else "👏"
                                    st.write(f"{emoji} **{event['type'].title()}** from {event['start_time']}s to {event['end_time']}s (confidence: {event['confidence']:.2f})")
                else:
                    st.warning("⚠️ SPEECHMATICS_API_KEY not set. Speechmatics analysis skipped.")
                
                if not api_key and not speechmatics_api_key:
                    st.warning("⚠️ No API keys found. Please set DEEPGRAM_API_KEY or SPEECHMATICS_API_KEY in your .env file to transcribe.")
                
            except Exception as e:
                st.error(f"❌ Error transcribing audio: {str(e)}")
                st.exception(e)

    # Separate section for Final transcript creation
    st.markdown("---")
    st.markdown("### 📝 Create Final Transcript")
    st.markdown("Upload an audio file to generate a final merged transcript with emotion analysis. This will:")
    st.markdown("- Transcribe the audio using Deepgram (if not already done)")
    st.markdown("- Analyze emotions using Hume API")
    st.markdown("- Merge both transcripts into a final transcript using AI")

    final_audio_file = st.file_uploader("Upload audio file for final transcript", type=['m4a', 'wav', 'mp3'], key="final_audio")
    final_folder_name = st.text_input("Enter folder name for final transcript:", placeholder="e.g., Meeting_2024_10_29", key="final_folder")

    final_button = st.button("📝 Create Final Transcript", type="primary", use_container_width=True)

    # Process Final transcript creation when button is clicked
    if final_button:
        # Validation
        if not final_audio_file:
            st.error("❌ Please upload an audio file")
        elif not final_folder_name:
            st.error("❌ Please enter a folder name")
        else:
            try:
                # Create output directory structure
                base_folder = "Recording"
                output_folder = os.path.join(base_folder, final_folder_name)
                os.makedirs(output_folder, exist_ok=True)
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{final_audio_file.name}") as tmp_audio:
                    tmp_audio.write(final_audio_file.read())
                    tmp_audio_path = tmp_audio.name
                
                # Check if timestamped_transcription.txt already exists
                transcript_file = os.path.join(output_folder, "timestamped_transcription.txt")
                deepgram_transcript_file = os.path.join(output_folder, "deepgram_transcript.txt")
                conversation_path = os.path.join(output_folder, "conversation.wav")
                
                # Save uploaded audio file as conversation.wav if it doesn't exist
                if not os.path.exists(conversation_path):
                    audio_data, sample_rate = librosa.load(tmp_audio_path, sr=None, mono=False)
                    if len(audio_data.shape) == 1:
                        sf.write(conversation_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                    else:
                        audio_data = audio_data.T
                        sf.write(conversation_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                
                # Step 1: Transcribe with Deepgram if needed
                if not os.path.exists(transcript_file):
                    api_key = os.getenv("DEEPGRAM_API_KEY")
                    if not api_key:
                        st.error("❌ DEEPGRAM_API_KEY not found. Please set it in .env file to transcribe audio.")
                        os.unlink(tmp_audio_path)
                        st.stop()
                    
                    st.markdown("---")
                    st.markdown("### 🎙️ Step 1: Transcription")
                    with st.spinner("Transcribing audio with Deepgram... This may take a moment."):
                        transcript_result = transcribe_audio(conversation_path, transcript_file)
                    
                    if not transcript_result:
                        st.error("❌ Transcription failed. Cannot proceed.")
                        os.unlink(tmp_audio_path)
                        st.stop()
                    
                    st.success("✅ Transcription completed!")
                else:
                    st.info("ℹ️ Using existing transcription file.")
                
                # Create deepgram_transcript.txt from timestamped_transcription.txt for the merge function
                # Read the timestamped format and convert to readable format
                with open(transcript_file, "r", encoding="utf-8") as f:
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
                
                # Save formatted version for merge function
                with open(deepgram_transcript_file, "w", encoding="utf-8") as f:
                    f.write(formatted_deepgram)
                
                # Step 2: Run Hume emotion analysis
                st.markdown("---")
                st.markdown("### 😊 Step 2: Emotion Analysis")
                
                with st.spinner("Running Hume emotion analysis... This may take several minutes for longer audio files."):
                    hume_result = hume_emotion_analysis(conversation_path, output_folder)
                
                if not hume_result:
                    st.error("❌ Hume emotion analysis failed. Cannot proceed.")
                    os.unlink(tmp_audio_path)
                    st.stop()
                
                st.success("✅ Hume emotion analysis completed!")
                
                # Step 3: Merge transcripts
                st.markdown("---")
                st.markdown("### 🔄 Step 3: Merging Transcripts")
                
                hume_file = os.path.join(output_folder, "hume_emotion.txt")
                final_transcript_file = os.path.join(output_folder, "final_transcript.txt")
                
                with st.spinner("Merging transcripts with AI... This may take a moment."):
                    merged_transcript = merge_transcripts(deepgram_transcript_file, hume_file, final_transcript_file)
                
                # Clean up temp file
                os.unlink(tmp_audio_path)
                
                if merged_transcript:
                    st.success("✅ Final transcript created successfully!")
                    
                    st.markdown(f"**📄 Final transcript saved to:** `{final_transcript_file}`")
                    st.markdown(f"**📝 Deepgram transcript:** `{transcript_file}`")
                    st.markdown(f"**😊 Hume emotion transcript:** `{hume_result['output_file']}`")
                    
                    # Show preview
                    st.markdown("---")
                    st.markdown("### 📄 Preview of Final Transcript")
                    preview_lines = merged_transcript.split('\n')[:30]  # First 30 lines
                    st.text('\n'.join(preview_lines))
                    
                    # Show full output in expander
                    with st.expander("📄 View Full Final Transcript"):
                        st.text(merged_transcript)
                else:
                    st.error("❌ Failed to merge transcripts. Please check the error messages above.")
                
            except Exception as e:
                st.error(f"❌ Error creating final transcript: {str(e)}")
                st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** The first speaker will be on the left channel, second speaker on the right channel in the stereo output.")
