#!/bin/bash
# Setup script for transcription environment

echo "🎙️ Audio Transcription Setup"
echo "================================"
echo ""

# Check if DEEPGRAM_API_KEY is set
if [ -z "$DEEPGRAM_API_KEY" ]; then
    echo "⚠️  DEEPGRAM_API_KEY is not set"
    echo ""
    echo "Please run:"
    echo "  export DEEPGRAM_API_KEY='your-api-key-here'"
    echo ""
    echo "Or add it to your ~/.zshrc file:"
    echo "  echo 'export DEEPGRAM_API_KEY=\"your-api-key-here\"' >> ~/.zshrc"
    echo "  source ~/.zshrc"
    echo ""
else
    echo "✅ DEEPGRAM_API_KEY is set"
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Virtual environment exists"
else
    echo "⚠️  Virtual environment not found. Run: python3 -m venv venv"
fi

# Check if audio file exists
if [ -f "Recording/Aazar-Shahbaz/conversation.wav" ]; then
    echo "✅ Audio file found"
else
    echo "⚠️  Audio file not found at: Recording/Aazar-Shahbaz/conversation.wav"
fi

echo ""
echo "To run transcription:"
echo "  source venv/bin/activate"
echo "  python transcribe.py"
echo ""

