import streamlit as st
import soundfile as sf
import librosa
import numpy as np
import os
import json
import tempfile
from pathlib import Path
from deepgram import DeepgramClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Audio Merger", page_icon="🎙️", layout="wide")

st.title("🎙️ Audio Conversation Merger")
st.markdown("Upload two audio files (.m4a or .wav) to create a stereo conversation file")

# Sidebar for instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload two audio files (first speaker and second speaker)
    2. Enter a folder name to save the outputs
    3. Click 'Merge Audio' button
    4. Audio will be merged and automatically transcribed
    5. Download your files from the 'Recording' folder
    """)
    
    st.markdown("---")
    st.markdown("**Configuration**")
    enable_transcription = st.checkbox("Enable transcription (requires DEEPGRAM_API_KEY)", value=True)

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

# Merge button
merge_button = st.button("🎵 Merge Audio", type="primary", use_container_width=True)

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

def transcribe_audio(audio_path, output_file):
    """Transcribe audio file with speaker diarization."""
    try:
        # Get API key from environment
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            st.error("❌ DEEPGRAM_API_KEY not found in environment variables. Please set it in .env file.")
            return None
        
        # Create Deepgram client with API key
        deepgram = DeepgramClient(api_key=api_key)
        
        # Transcribe with diarization, sentiment, topics, and summary
        with open(audio_path, "rb") as audio_file:
            response = deepgram.listen.v1.media.transcribe_file(
                request=audio_file.read(),
                model="nova-3",
                language="en",
                summarize="v2",
                topics=True,
                intents=True,
                sentiment=True,
                smart_format=True,
                paragraphs=True,
                diarize=True,
                filler_words=True,
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
                                        
                                        # Get sentiment if available
                                        sentiment = sentence.get('sentiment', 'neutral')
                                        sentiment_score = sentence.get('sentiment_score', 0.0)
                                        
                                        if transcript:
                                            utterances.append({
                                                'start': start,
                                                'end': end,
                                                'speaker': f'Speaker {int(speaker_id) + 1}',
                                                'transcript': transcript,
                                                'sentiment': sentiment,
                                                'sentiment_score': sentiment_score
                                            })
        
        # Sort by start time
        utterances.sort(key=lambda x: x['start'])
        
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            for utt in utterances:
                # Format: [start,end]  SPEAKER_ID  SENTIMENT (score)  transcript
                sentiment_info = f"{utt.get('sentiment', 'neutral')} ({utt.get('sentiment_score', 0.0):.2f})"
                line = f"[{utt['start']:.3f},{utt['end']:.3f}]\t{utt['speaker']}\t{sentiment_info}\t{utt['transcript']}\n"
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
            
            # Display success message
            st.success(f"✅ Audio files processed successfully!")
            
            # Display results
            st.markdown("### 📁 Output Files")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**First Speaker (Mono)**")
                st.info(f"✓ {result['duration']:.2f}s @ {result['sample_rate']} Hz")
            
            with col2:
                st.markdown("**Second Speaker (Mono)**")
                st.info(f"✓ {result['duration']:.2f}s @ {result['sample_rate']} Hz")
            
            with col3:
                st.markdown("**Conversation (Stereo)**")
                st.success(f"✓ {result['duration']:.2f}s @ {result['sample_rate']} Hz")
            
            st.markdown(f"**📂 Files saved to:** `{result['output_folder']}/`")
            
            # Show file sizes
            st.markdown("### 📊 File Information")
            file_sizes = {}
            for name, path in [
                ('first_speaker.wav', result['first_mono']),
                ('second_speaker.wav', result['second_mono']),
                ('conversation.wav', result['conversation'])
            ]:
                size = os.path.getsize(path) / (1024 * 1024)
                file_sizes[name] = size
            
            df = {
                'File': list(file_sizes.keys()),
                'Size (MB)': [f"{size:.2f}" for size in file_sizes.values()]
            }
            st.table(df)
            
            # Transcribe the conversation if enabled
            if enable_transcription:
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
                        
                        # Show preview with sentiment
                        st.markdown("**Preview:**")
                        preview_text = ""
                        for i, utt in enumerate(transcript_result['utterances'][:5]):
                            sentiment = utt.get('sentiment', 'neutral')
                            sentiment_score = utt.get('sentiment_score', 0.0)
                            emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😞"
                            preview_text += f"[{utt['start']:.1f}s] {utt['speaker']} {emoji} {sentiment}({sentiment_score:.2f}): {utt['transcript'][:40]}...\n"
                        
                        st.text(preview_text)
                        
                        # Show summary if available
                        if transcript_result.get('summary'):
                            with st.expander("📄 Conversation Summary"):
                                st.write(transcript_result['summary'])
                else:
                    st.warning("⚠️ DEEPGRAM_API_KEY not set. Transcription skipped.")
            
        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** The first speaker will be on the left channel, second speaker on the right channel in the stereo output.")

