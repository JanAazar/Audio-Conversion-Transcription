import streamlit as st
import soundfile as sf
import librosa
import numpy as np
import os
import json
import tempfile
import asyncio
import re
import base64
from datetime import datetime
from pathlib import Path
from deepgram import DeepgramClient
from dotenv import load_dotenv
import time
import constants

load_dotenv()

st.set_page_config(page_title="Annotation and Emotion Marking", page_icon="📝", layout="wide")

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'annotation'
    
if 'merging_page_authenticated' not in st.session_state:
    st.session_state.merging_page_authenticated = False
if 'show_password_input' not in st.session_state:
    st.session_state.show_password_input = False

def convert_to_mono(audio_file, target_sr=48000):
    audio_data, sample_rate = librosa.load(audio_file, sr=target_sr, mono=False)
    
    if audio_data.ndim > 1:
        channel_rms = np.sqrt(np.mean(audio_data**2, axis=1))
        best_channel = int(np.argmax(channel_rms))
        
        averaged = np.mean(audio_data, axis=0)
        averaged_rms = np.sqrt(np.mean(averaged**2))
        best_rms = channel_rms[best_channel] if channel_rms.size else 0.0
        
        if best_rms > 0 and (averaged_rms < 0.1 * best_rms):
            try:
                st.warning(
                    f"Detected unbalanced stereo in {os.path.basename(audio_file)}. "
                    f"Using channel {best_channel + 1} only."
                )
            except Exception:
                pass
            audio_data = audio_data[best_channel]
        else:
            audio_data = averaged
    
    return audio_data.astype(np.float32), sample_rate

def process_audio(first_file, second_file, folder_name):
    base_folder = "Recording"
    output_folder = os.path.join(base_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    st.info("🔄 Converting files to mono...")
    first_data, first_sr = convert_to_mono(first_file)
    second_data, second_sr = convert_to_mono(second_file)
    
    first_mono_path = os.path.join(output_folder, "first_speaker.wav")
    second_mono_path = os.path.join(output_folder, "second_speaker.wav")
    
    sf.write(first_mono_path, first_data, first_sr, format='WAV', subtype='PCM_16')
    sf.write(second_mono_path, second_data, second_sr, format='WAV', subtype='PCM_16')
    
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
    
    st.info("🔄 Aligning audio durations...")
    max_len = max(len(first_data), len(second_data))
    if len(first_data) < max_len:
        first_data = np.pad(first_data, (0, max_len - len(first_data)), 'constant')
    if len(second_data) < max_len:
        second_data = np.pad(second_data, (0, max_len - len(second_data)), 'constant')
    
    st.info("🎵 Creating stereo output...")
    stereo = np.column_stack([first_data, second_data])
    
    conversation_path = os.path.join(output_folder, "conversation.wav")
    sf.write(conversation_path, stereo, sample_rate)
    
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
    base_folder = "Recording"
    output_folder = os.path.join(base_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    st.info("🔄 Loading stereo audio file...")
    audio_data, sample_rate = librosa.load(stereo_file, sr=None, mono=False)
    
    if len(audio_data.shape) == 1:
        st.error("❌ The uploaded file is mono, not stereo. Cannot split into separate speakers.")
        return None
    
    left_channel = audio_data[0] 
    right_channel = audio_data[1]
    
    first_speaker_path = os.path.join(output_folder, "first_speaker.wav")
    second_speaker_path = os.path.join(output_folder, "second_speaker.wav")
    
    st.info("💾 Saving speaker files...")
    sf.write(first_speaker_path, left_channel, sample_rate, format='WAV', subtype='PCM_16')
    sf.write(second_speaker_path, right_channel, sample_rate, format='WAV', subtype='PCM_16')
    
    conversation_path = os.path.join(output_folder, "conversation.wav")
    stereo_transposed = audio_data.T
    sf.write(conversation_path, stereo_transposed, sample_rate, format='WAV', subtype='PCM_16')

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
    st.success(f"✅ Audio files processed successfully!")
    
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
    try:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            st.error("❌ DEEPGRAM_API_KEY not found in environment variables. Please set it in .env file.")
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
        
        debug_file = output_file.replace('.txt', '_full_response.json')
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, indent=2, default=str, ensure_ascii=False)
        
        st.write("🔍 Debug - Full response saved to:", debug_file)
        st.write("🔍 Debug - Response keys:", list(response_json.keys()) if response_json else "Empty response")
        
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
        
        utterances = []
        
        if 'results' in response_json and 'channels' in response_json['results']:
            for channel in response_json['results']['channels']:
                if 'alternatives' in channel and len(channel['alternatives']) > 0:
                    alternative = channel['alternatives'][0]
                    
                    if 'paragraphs' in alternative:
                        paragraphs_data = alternative['paragraphs']
                        st.write(f"🔍 Paragraphs type: {type(paragraphs_data)}")
                        
                        if isinstance(paragraphs_data, dict):
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
        
        with open(output_file, "w", encoding="utf-8") as f:
            for utt in utterances:
                line = f"[{utt['start']:.3f},{utt['end']:.3f}]\t{utt['speaker']}\t{utt['transcript']}\n"
                f.write(line)
        
        summary_text = None
        if 'results' in response_json and 'summary' in response_json['results']:
            summary = response_json['results']['summary']
            if summary and 'short' in summary:
                summary_text = summary['short']
                
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
    try:
        api_key = os.getenv("SPEECHMATICS_API_KEY")
        if not api_key:
            st.error("❌ SPEECHMATICS_API_KEY not found in environment variables. Please set it in .env file.")
            return None
        
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
        
        settings = ConnectionSettings(
            url="https://asr.api.speechmatics.com/v2",
            auth_token=api_key,
        )
        
        with BatchClient(settings) as client:
            job_id = client.submit_job(
                audio=audio_path,
                transcription_config=conf,
            )
            
            transcript = client.wait_for_completion(job_id, transcription_format="json-v2")
            
            summary = transcript.get("summary", {}).get("content", "")
            
            audio_events = transcript.get("audio_events", [])
            audio_event_summary = transcript.get("audio_event_summary", {})
            
            summary_file = os.path.join(output_folder, "summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
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

async def run_hume_analysis_async(audio_path, status_placeholder=None):
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        raise ValueError("HUME_API_KEY not found in environment variables")
    
    client = AsyncHumeClient(api_key=api_key)
    
    if status_placeholder:
        status_placeholder.info("🔄 Submitting job to Hume API...")
    
    audio_file = open(audio_path, mode="rb")
    
    prosody_config = Prosody()
    models_chosen = Models(prosody=prosody_config)
    stringified_configs = InferenceBaseRequest(models=models_chosen)
    
    job_id = await client.expression_measurement.batch.start_inference_job_from_local_file(
        json=stringified_configs, file=[audio_file]
    )
    
    audio_file.close()
    
    if status_placeholder:
        status_placeholder.info(f"✅ Job submitted! Job ID: {job_id}\n🔄 Waiting for predictions...")
    
    max_wait = 600 
    waited = 0
    
    while waited < max_wait:
        try:
            job_predictions = await client.expression_measurement.batch.get_job_predictions(
                id=job_id
            )
            
            if job_predictions and len(job_predictions) > 0:
                pred_str = str(job_predictions[0])
                if 'results=InferenceResults(predictions=[]' not in pred_str:
                    if status_placeholder:
                        status_placeholder.success("✅ Predictions received!")
                    return {
                        'predictions': job_predictions,
                        'job_id': job_id
                    }
            
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
    try:
        api_key = os.getenv("HUME_API_KEY")
        if not api_key:
            st.error("❌ HUME_API_KEY not found in environment variables. Please set it in .env file.")
            return None
        
        status_placeholder = st.empty()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_hume_analysis_async(audio_path, status_placeholder))
        finally:
            loop.close()
        
        formatted_output = parse_hume_predictions(result['predictions'])
        
        formatted_output += "\n" + "-" * 100 + "\n"
        formatted_output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
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
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
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
  [9.280,11.120] Speaker 1: Hello there. (Emotion: Sadness, Surprise)

- timestamps should be in the format [start,end] not [HH:MM:SS.mmm]


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
        
        return merged_transcript
        
    except Exception as e:
        st.error(f"❌ Failed to merge transcripts: {e}")
        import traceback
        traceback.print_exc()
        return None

if st.session_state.current_page == 'annotation':
    st.title("📝 Annotation and Emotion Marking")
    
    if st.button("➡️ Go to Audio Merging Tools"):
        st.session_state.show_password_input = True
        st.rerun()
    
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
    
    conversation_audio = st.file_uploader("Upload conversation.wav file", type=['wav', 'm4a', 'mp3'], key="annotation_audio")
    transcript_file = st.file_uploader("Upload final_transcript.txt file", type=['txt'], key="annotation_transcript")
    
    if conversation_audio and transcript_file:
        transcript_text = transcript_file.read().decode('utf-8')
        transcript_lines = transcript_text.strip().split('\n')
        
        original_transcript_lines = [line.strip() for line in transcript_lines if line.strip()]
        
        transcript_entries = []
        for line in transcript_lines:
            line = line.strip()
            if not line:
                continue
            
            total_seconds = None
            end_seconds = None
            timestamp_format = None 
            
            timestamp_match = re.search(r'\[(\d{2}):(\d{2}):(\d{2}\.\d+)\]', line)
            if timestamp_match:
                hours = int(timestamp_match.group(1))
                minutes = int(timestamp_match.group(2))
                seconds = float(timestamp_match.group(3))
                total_seconds = hours * 3600 + minutes * 60 + seconds
                end_seconds = total_seconds 
                timestamp_format = 'time_format'
            else:
                timestamp_match = re.search(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', line)
                if timestamp_match:
                    total_seconds = float(timestamp_match.group(1))
                    end_seconds = float(timestamp_match.group(2))
                    timestamp_format = 'range_format'
            
            if total_seconds is None:
                continue
            
            speaker_match = re.search(r'Speaker \d+', line)
            speaker = speaker_match.group(0) if speaker_match else "Unknown"
            
            text_start = line.find(':', line.find(speaker))
            if text_start == -1:
                continue
            
            text_part = line[text_start + 1:].strip()
            intensity = 3 
            intensity_match = re.search(r'\[Intensity:\s*(\d+)\]|Intensity:\s*(\d+)', text_part, re.IGNORECASE)
            if intensity_match:
                intensity = int(intensity_match.group(1) or intensity_match.group(2))
                text_part = re.sub(r'\[Intensity:\s*\d+\]|Intensity:\s*\d+', '', text_part, flags=re.IGNORECASE).strip()
            
            emotion_match = re.search(r'\s*\(Emotion:.*?\)\s*$', text_part)
            emotion_part = ""
            emotions_list = []
            if emotion_match:
                emotion_part = emotion_match.group(0).strip()
                text_part = text_part[:emotion_match.start()].strip()
                emotion_content = emotion_part.replace('(Emotion:', '').replace(')', '').strip()
                if emotion_content:
                    emotions_list = [e.strip() for e in emotion_content.split(',') if e.strip()]
            
            original_line = line
            
            transcript_entries.append({
                'time': total_seconds,
                'end_time': end_seconds,
                'timestamp_format': timestamp_format,
                'speaker': speaker,
                'text': text_part,
                'emotion': emotion_part,
                'emotions': emotions_list, 
                'intensity': intensity, 
                'original_line': original_line,
                'index': len(transcript_entries)  
            })
        
        transcript_entries.sort(key=lambda x: x['time'])
        
        if transcript_entries:
            st.success(f"✅ Loaded {len(transcript_entries)} transcript entries")
            
            audio_bytes = conversation_audio.read()
            audio_ext = conversation_audio.name.split('.')[-1].lower()
            audio_format_map = {'wav': 'wav', 'mp3': 'mp3', 'm4a': 'mp4'}
            audio_format = audio_format_map.get(audio_ext, 'wav')
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            transcript_json = json.dumps(transcript_entries)
            
            if 'original_transcript_data' not in st.session_state:
                st.session_state.original_transcript_data = {
                    'entries': transcript_entries,
                    'original_lines': original_transcript_lines
                }

            edited_transcript_container = st.container()

            html_component = st.components.v1.html(constants.html_content, height=700)
            
            edited_text = st.text_area("Edited transcript:", height=300, key="manual_edit", 
                                      placeholder="Click 'Prepare Save' above and paste here (Ctrl+V or Cmd+V), then edit and download below...")
            
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
            
        else:
            st.error("❌ Could not parse transcript file. Please check the format.")
    
    elif conversation_audio or transcript_file:
        st.warning("⚠️ Please upload both the audio file and transcript file to proceed.")
        
else:
    if not st.session_state.merging_page_authenticated:
        st.error("🔒 Access Denied: Authentication required")
        st.info("Please use the password-protected access from the Annotation page.")
        if st.button("← Back to Annotation Page", use_container_width=True):
            st.session_state.current_page = 'annotation'
            st.rerun()
        st.stop()
    
    st.title("🎙️ Audio Conversation Merger")
    st.markdown("Upload two audio files (.m4a or .wav) to create a stereo conversation file")
    
    if st.button("← Back to Annotation Page", use_container_width=True):
        st.session_state.current_page = 'annotation'
        st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("First Speaker")
        first_file = st.file_uploader("Upload first speaker audio", type=['m4a', 'wav', 'mp3'], key="first")

    with col2:
        st.subheader("Second Speaker")
        second_file = st.file_uploader("Upload second speaker audio", type=['m4a', 'wav', 'mp3'], key="second")

    st.markdown("---")
    folder_name = st.text_input("Enter folder name for this recording:", placeholder="e.g., Meeting_2024_10_29")

    col1, col2 = st.columns(2)
    with col1:
        merge_button = st.button("🎵 Merge Audio", type="primary", use_container_width=True)
    with col2:
        just_merge_button = st.button("⚡ Just Merge", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔊 Split Stereo Conversation")
    st.markdown("Upload a stereo conversation file to separate it into individual speaker files (left channel = speaker 1, right channel = speaker 2)")

    split_conversation_file = st.file_uploader("Upload stereo conversation audio", type=['m4a', 'wav', 'mp3'], key="split_conversation")
    split_folder_name = st.text_input("Enter folder name for split files:", placeholder="e.g., Meeting_2024_10_29", key="split_folder")

    split_button = st.button("🔊 Create Separate Conversations", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📝 Or Transcribe an Existing Conversation")
    st.markdown("Upload a stereo conversation file to transcribe it (no merging needed)")

    conversation_file = st.file_uploader("Upload conversation audio", type=['m4a', 'wav', 'mp3'], key="conversation")
    transcribe_folder_name = st.text_input("Enter folder name for transcription:", placeholder="e.g., Meeting_2024_10_29", key="transcribe_folder")

    transcribe_button = st.button("📝 Transcribe the Final Conversation", type="primary", use_container_width=True)

    if just_merge_button:
        if not first_file or not second_file:
            st.error("❌ Please upload both audio files")
        elif not folder_name:
            st.error("❌ Please enter a folder name")
        else:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{first_file.name}") as tmp_first:
                    tmp_first.write(first_file.read())
                    tmp_first_path = tmp_first.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{second_file.name}") as tmp_second:
                    tmp_second.write(second_file.read())
                    tmp_second_path = tmp_second.name
                
                result = process_audio(tmp_first_path, tmp_second_path, folder_name)
                
                os.unlink(tmp_first_path)
                os.unlink(tmp_second_path)
                
                display_basic_results(result)
                
            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")
                st.exception(e)

    if merge_button:
        if not first_file or not second_file:
            st.error("❌ Please upload both audio files")
        elif not folder_name:
            st.error("❌ Please enter a folder name")
        else:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{first_file.name}") as tmp_first:
                    tmp_first.write(first_file.read())
                    tmp_first_path = tmp_first.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{second_file.name}") as tmp_second:
                    tmp_second.write(second_file.read())
                    tmp_second_path = tmp_second.name
                
                result = process_audio(tmp_first_path, tmp_second_path, folder_name)
                
                os.unlink(tmp_first_path)
                os.unlink(tmp_second_path)
                
                display_basic_results(result)
                
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
                        
                        st.markdown("**Preview:**")
                        preview_text = ""
                        for i, utt in enumerate(transcript_result['utterances'][:5]):
                            preview_text += f"[{utt['start']:.1f}s] {utt['speaker']}: {utt['transcript'][:60]}...\n"
                        
                        st.text(preview_text)
                        
                        if transcript_result.get('summary'):
                            with st.expander("📄 Conversation Summary"):
                                st.write(transcript_result['summary'])
                else:
                    st.warning("⚠️ DEEPGRAM_API_KEY not set. Transcription skipped.")
                
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
                        
                        with st.expander("📄 Speechmatics Summary"):
                            st.write(speechmatics_result['summary'])
                        
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

    if split_button:
        if not split_conversation_file:
            st.error("❌ Please upload a stereo conversation audio file")
        elif not split_folder_name:
            st.error("❌ Please enter a folder name for split files")
        else:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{split_conversation_file.name}") as tmp_audio:
                    tmp_audio.write(split_conversation_file.read())
                    tmp_audio_path = tmp_audio.name
                
                result = split_stereo_conversation(tmp_audio_path, split_folder_name)
                
                os.unlink(tmp_audio_path)
                
                if result:
                    st.success(f"✅ Conversation split successfully!")
                    
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

    if transcribe_button:
        if not conversation_file:
            st.error("❌ Please upload a conversation audio file")
        elif not transcribe_folder_name:
            st.error("❌ Please enter a folder name for transcription")
        else:
            try:
                base_folder = "Recording"
                output_folder = os.path.join(base_folder, transcribe_folder_name)
                os.makedirs(output_folder, exist_ok=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{conversation_file.name}") as tmp_audio:
                    tmp_audio.write(conversation_file.read())
                    tmp_audio_path = tmp_audio.name
                
                conversation_path = os.path.join(output_folder, "conversation.wav")
                
                audio_data, sample_rate = librosa.load(tmp_audio_path, sr=None, mono=False)
                
                if len(audio_data.shape) == 1:
                    sf.write(conversation_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                else:
                    audio_data = audio_data.T
                    sf.write(conversation_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                
                os.unlink(tmp_audio_path)
                
                st.success(f"✅ Conversation file saved to: `{conversation_path}`")
                
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
                        
                        st.markdown("**Preview:**")
                        preview_text = ""
                        for i, utt in enumerate(transcript_result['utterances'][:5]):
                            preview_text += f"[{utt['start']:.1f}s] {utt['speaker']}: {utt['transcript'][:60]}...\n"
                        
                        st.text(preview_text)
                        
                        if transcript_result.get('summary'):
                            with st.expander("📄 Conversation Summary"):
                                st.write(transcript_result['summary'])
                else:
                    st.warning("⚠️ DEEPGRAM_API_KEY not set. Transcription skipped.")
                
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
                        
                        with st.expander("📄 Speechmatics Summary"):
                            st.write(speechmatics_result['summary'])
                        
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

    st.markdown("---")
    st.markdown("### 📝 Create Final Transcript")
    st.markdown("Upload an audio file to generate a final merged transcript with emotion analysis. This will:")
    st.markdown("- Transcribe the audio using Deepgram (if not already done)")
    st.markdown("- Analyze emotions using Hume API")
    st.markdown("- Merge both transcripts into a final transcript using AI")

    final_audio_file = st.file_uploader("Upload audio file for final transcript", type=['m4a', 'wav', 'mp3'], key="final_audio")
    final_folder_name = st.text_input("Enter folder name for final transcript:", placeholder="e.g., Meeting_2024_10_29", key="final_folder")

    final_button = st.button("📝 Create Final Transcript", type="primary", use_container_width=True)

    if final_button:
        if not final_audio_file:
            st.error("❌ Please upload an audio file")
        elif not final_folder_name:
            st.error("❌ Please enter a folder name")
        else:
            try:
                base_folder = "Recording"
                output_folder = os.path.join(base_folder, final_folder_name)
                os.makedirs(output_folder, exist_ok=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{final_audio_file.name}") as tmp_audio:
                    tmp_audio.write(final_audio_file.read())
                    tmp_audio_path = tmp_audio.name
                
                transcript_file = os.path.join(output_folder, "timestamped_transcription.txt")
                deepgram_transcript_file = os.path.join(output_folder, "deepgram_transcript.txt")
                conversation_path = os.path.join(output_folder, "conversation.wav")
                
                if not os.path.exists(conversation_path):
                    audio_data, sample_rate = librosa.load(tmp_audio_path, sr=None, mono=False)
                    if len(audio_data.shape) == 1:
                        sf.write(conversation_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                    else:
                        audio_data = audio_data.T
                        sf.write(conversation_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                
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
                
                st.markdown("---")
                st.markdown("### 😊 Step 2: Emotion Analysis")
                
                with st.spinner("Running Hume emotion analysis... This may take several minutes for longer audio files."):
                    hume_result = hume_emotion_analysis(conversation_path, output_folder)
                
                if not hume_result:
                    st.error("❌ Hume emotion analysis failed. Cannot proceed.")
                    os.unlink(tmp_audio_path)
                    st.stop()
                
                st.success("✅ Hume emotion analysis completed!")
                
                st.markdown("---")
                st.markdown("### 🔄 Step 3: Merging Transcripts")
                
                hume_file = os.path.join(output_folder, "hume_emotion.txt")
                final_transcript_file = os.path.join(output_folder, "final_transcript.txt")
                
                with st.spinner("Merging transcripts with AI... This may take a moment."):
                    merged_transcript = merge_transcripts(deepgram_transcript_file, hume_file, final_transcript_file)
                
                os.unlink(tmp_audio_path)
                
                if merged_transcript:
                    st.success("✅ Final transcript created successfully!")
                    
                    st.markdown(f"**📄 Final transcript saved to:** `{final_transcript_file}`")
                    st.markdown(f"**📝 Deepgram transcript:** `{transcript_file}`")
                    st.markdown(f"**😊 Hume emotion transcript:** `{hume_result['output_file']}`")
                    
                    st.markdown("---")
                    st.markdown("### 📄 Preview of Final Transcript")
                    preview_lines = merged_transcript.split('\n')[:30]
                    st.text('\n'.join(preview_lines))
                    
                    with st.expander("📄 View Full Final Transcript"):
                        st.text(merged_transcript)
                else:
                    st.error("❌ Failed to merge transcripts. Please check the error messages above.")
                
            except Exception as e:
                st.error(f"❌ Error creating final transcript: {str(e)}")
                st.exception(e)

    st.markdown("---")
    st.markdown("💡 **Tip:** The first speaker will be on the left channel, second speaker on the right channel in the stereo output.")