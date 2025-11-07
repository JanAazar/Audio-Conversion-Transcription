from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SPEECHMATICS_API_KEY")
PATH_TO_FILE = os.path.join(os.path.dirname(__file__), "Recording", "Jordy-Samantha", "conversation.wav")
#PATH_TO_FILE = os.path.join(os.path.dirname(__file__), "test.wav")
LANGUAGE = "en"

settings = ConnectionSettings(
    url="https://asr.api.speechmatics.com/v2",
    auth_token=API_KEY,
)

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
    "language": LANGUAGE,
    "operating_point": "enhanced"
  },
    "output_config": {
    "srt_overrides": {
      "max_line_length": 37,
      "max_lines": 2
    }
  },
  "type": "transcription"
}
# Open the client using a context manager
with BatchClient(settings) as client:
    try:
        job_id = client.submit_job(
            audio=PATH_TO_FILE,
            transcription_config=conf,
        )
        print(f"job {job_id} submitted successfully, waiting for transcript")

        # Note that in production, you should set up notifications instead of polling.
        # Notifications are described here: https://docs.speechmatics.com/speech-to-text/batch/notifications
        transcript = client.wait_for_completion(job_id, transcription_format="json-v2")

        # Process and print transcript with timestamps
        # Navigate to words in the JSON structure
        results = transcript.get("results", [])
        
        # Build list of items with their speaker, content, and timestamps
        items = []
        for item in results:
            if isinstance(item, dict) and "alternatives" in item:
                start_time = item.get("start_time", 0)
                end_time = item.get("end_time", 0)
                item_type = item.get("type", "")
                
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
                        "type": item_type
                    })
        
        # Group items by speaker into utterances
        current_utterance = None
        current_text = []
        
        for item in items:
            speaker = item.get("speaker", "?")
            content = item.get("content", "")
            start_time = item.get("start", 0)
            end_time = item.get("end", 0)
            
            
            # Check if this is a new utterance (new speaker)
            if current_utterance is None or speaker != current_utterance.get("speaker"):
                # Print previous utterance if exists
                if current_utterance is not None:
                    timestamp = f"[{current_utterance.get('start', 0):.3f},{current_utterance.get('end', 0):.3f}]"
                    current_speaker = current_utterance.get('speaker', '?')
                    # Convert S1, S2 to Speaker 1, Speaker 2
                    if current_speaker.startswith('S'):
                        speaker_label = f"Speaker {current_speaker[1:]}"
                    else:
                        speaker_label = f"Speaker {current_speaker}"
                    text = ''.join(current_text).strip()
                    if text:  # Only print if there's content
                        print(f'"{timestamp}"\t{speaker_label}\t"{text}"')
                
                # Start new utterance
                current_utterance = {
                    "speaker": speaker,
                    "start": start_time,
                    "end": end_time
                }
                current_text = []
            
            # Add content to current utterance
            # Add space ONLY before words (not punctuation)
            if current_text and content not in '.!,?;:':
                current_text.append(' ')
            
            current_text.append(content)
            
            # Update end time
            if current_utterance is not None:
                current_utterance["end"] = end_time
        
        # Print last utterance
        if current_utterance is not None:
            timestamp = f"[{current_utterance.get('start', 0):.3f},{current_utterance.get('end', 0):.3f}]"
            current_speaker = current_utterance.get('speaker', '?')
            if current_speaker.startswith('S'):
                speaker_label = f"Speaker {current_speaker[1:]}"
            else:
                speaker_label = f"Speaker {current_speaker}"
            text = ''.join(current_text).strip()
            if text:  # Only print if there's content
                print(f'"{timestamp}"\t{speaker_label}\t"{text}"')
    except HTTPStatusError as e:
        if e.response.status_code == 401:
            print("Invalid API key - Check your API_KEY at the top of the code!")
        elif e.response.status_code == 400:
            print(e.response.json())
        else:
            raise e