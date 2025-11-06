# Audio Conversation Merger

A Streamlit app for merging two audio files into a stereo conversation file, with integrated Hume API for emotion analysis and Speechmatics for transcription.

## Features

- 🎙️ Upload two audio files (.m4a, .wav, or .mp3)
- 🔄 Automatic conversion to mono WAV format
- 🎵 Creates stereo output with speakers on separate channels
- 📁 Organizes outputs in custom subfolders
- ⚡ Automatic sample rate alignment and duration matching
- 🧠 Hume API integration for emotion analysis (see `test_hume.py`)
- 🎤 Speechmatics integration for speech-to-text transcription

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install streamlit librosa soundfile numpy
```

## Usage

### Running the App

```bash
source venv/bin/activate
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

1. **Upload First Speaker**: Click "Upload first speaker audio" and select an audio file
2. **Upload Second Speaker**: Click "Upload second speaker audio" and select another audio file
3. **Enter Folder Name**: Type a name for the output folder (e.g., "Meeting_2024_10_29")
4. **Merge Audio**: Click the "🎵 Merge Audio" button
5. **Access Files**: Your files will be saved in `Recording/[your_folder_name]/`

## Output Files

After merging, three files are created in your specified folder:

- `first_speaker.wav` - Mono audio of the first speaker
- `second_speaker.wav` - Mono audio of the second speaker  
- `conversation.wav` - Stereo audio with first speaker on left channel, second speaker on right channel

## Technical Details

- **Sample Rate**: Automatically converts to 48kHz
- **Format**: PCM 16-bit WAV files
- **Channel Layout**: 
  - Mono files: Single channel
  - Stereo output: Left = First Speaker, Right = Second Speaker
- **Duration Alignment**: Automatically pads shorter audio with silence to match durations

## API Integration

### Hume API (Emotion Analysis)

The project includes integration with Hume's emotion analysis API via `test_hume.py`. This script uses the Prosody model to analyze audio and extract emotions from speech.

**Important Note**: The current version of the hume SDK (v0.13.1) has bugs that have been patched in the installed package. If you reinstall or upgrade the hume package, you'll need to reapply these patches:

1. In `venv/lib/python3.13/site-packages/hume/expression_measurement/client.py`, change line 36 to import from `client_with_utils` instead of `client`
2. In `venv/lib/python3.13/site-packages/hume/expression_measurement/batch/client_with_utils.py`, add `base_url` parameter to the request calls

To use the Hume API:
```bash
python test_hume.py
```

### Speechmatics (Transcription)

See `test_speechmatics.py` for example usage of the Speechmatics batch transcription API.

## Requirements

- Python 3.7+
- streamlit
- librosa
- soundfile
- numpy
- hume (for emotion analysis)
- speechmatics-python (for transcription)

