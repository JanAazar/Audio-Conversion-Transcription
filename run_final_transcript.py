#!/usr/bin/env python3
"""
CLI helper that replicates the Streamlit “📝 Create Final Transcript” button
for a single conversation folder (default: Conversation_59).
"""

import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from hume import AsyncHumeClient
from hume.core.api_error import ApiError
from hume.expression_measurement.batch import Models, Prosody
from hume.expression_measurement.batch.types import InferenceBaseRequest

# Reuse Deepgram + Speechmatics helpers from the existing batch script
from batch_transcribe_final_conversations import (
    transcribe_audio as deepgram_transcribe,
)

load_dotenv()

BASE_FOLDER = Path("/Users/aazarjan/Desktop/Audio-Merging/Recording")
DEFAULT_CONVERSATION_NUMBER = 59
LANGUAGE = "es"

WAIT_SECONDS = 600  # Hume polling limit (10 minutes)


# ---------------------------------------------------------------------------

def parse_hume_predictions(job_predictions):
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("FORMATTED TRANSCRIPT WITH EMOTION ANALYSIS")
    lines.append("=" * 100)

    pred_str = str(job_predictions[0]) if job_predictions else ""
    predictions = []
    text_start = "ProsodyPrediction(text="
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
        emotions = [
            (match.group(1), float(match.group(2)))
            for match in re.finditer(emotion_pattern, segment)
        ]

        predictions.append(
            {
                "text": text,
                "begin": begin,
                "end": end,
                "confidence": confidence,
                "emotions": emotions,
            }
        )

    if not predictions:
        lines.append("No predictions found in the response.")
        return "\n".join(lines)

    lines.append(f"\n{'Time':<20} {'Transcription':<55} {'Top Emotions'}")
    lines.append("-" * 120)

    for pred in predictions:
        text = pred["text"]
        time_str = f"{pred['begin']:.1f}s-{pred['end']:.1f}s"
        pred["emotions"].sort(key=lambda item: item[1], reverse=True)
        top_emotions = pred["emotions"][:3]
        emotions_text = ", ".join(f"{name} ({score:.0%})" for name, score in top_emotions)
        if len(text) > 52:
            text = text[:49] + "..."
        lines.append(f"{time_str:<20} {text:<55} {emotions_text}")

    lines.append("-" * 120)
    lines.append(f"\nTotal segments: {len(predictions)}")
    lines.append("")
    return "\n".join(lines)


async def run_hume_async(audio_path: Path):
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        raise RuntimeError("HUME_API_KEY not set")

    client = AsyncHumeClient(api_key=api_key)
    with open(audio_path, "rb") as audio_file:
        job_id = await client.expression_measurement.batch.start_inference_job_from_local_file(
            json=InferenceBaseRequest(models=Models(prosody=Prosody())),
            file=[audio_file],
        )
    print(f"🔁 Hume job submitted: {job_id}")

    waited = 0
    while waited < WAIT_SECONDS:
        try:
            job_predictions = await client.expression_measurement.batch.get_job_predictions(id=job_id)
        except ApiError as exc:
            message = ""
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                message = body.get("message", "")
            if exc.status_code == 400 and message.lower() == "job is in progress.":
                await asyncio.sleep(2)
                waited += 2
                continue
            raise
        if job_predictions:
            pred_str = str(job_predictions[0])
            if "results=InferenceResults(predictions=[]" not in pred_str:
                print("✅ Hume predictions received")
                return {"predictions": job_predictions, "job_id": job_id}

        await asyncio.sleep(2)
        waited += 2

    raise TimeoutError("Timed out waiting for Hume predictions")


def merge_transcripts(deepgram_file: Path, hume_file: Path, output_file: Path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    with open(deepgram_file, "r", encoding="utf-8") as fh:
        deepgram_text = fh.read()
    with open(hume_file, "r", encoding="utf-8") as fh:
        hume_text = fh.read()

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

    print("🧠 Merging transcripts with OpenAI ...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that merges transcripts."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    merged_transcript = response.choices[0].message.content
    output_file.write_text(merged_transcript, encoding="utf-8")
    print(f"✅ Final transcript saved to {output_file}")
    return merged_transcript


def ensure_deepgram_transcripts(conversation_folder: Path, language: str):
    conversation_audio = conversation_folder / "conversation.wav"
    if not conversation_audio.exists():
        raise FileNotFoundError(f"{conversation_audio} not found")

    timestamped = conversation_folder / "timestamped_transcription.txt"
    if timestamped.exists():
        print(f"ℹ️ Using existing Deepgram transcript: {timestamped}")
    else:
        print("🎙️ Running Deepgram transcription...")
        deepgram_transcribe(str(conversation_audio), str(timestamped), language)

    deepgram_plain = conversation_folder / "deepgram_transcript.txt"
    if deepgram_plain.exists():
        print(f"ℹ️ Using existing formatted Deepgram transcript: {deepgram_plain}")
    else:
        formatted_lines = []
        for line in timestamped.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                time_range, speaker, text = parts[0], parts[1], "\t".join(parts[2:])
                formatted_lines.append(f"{time_range} {speaker}: {text}")
        deepgram_plain.write_text("\n".join(formatted_lines), encoding="utf-8")
        print(f"✅ Saved formatted Deepgram transcript to {deepgram_plain}")

    return timestamped, deepgram_plain


def run_hume(conversation_folder: Path):
    conversation_audio = conversation_folder / "conversation.wav"
    hume_output = conversation_folder / "hume_emotion.txt"

    print("😊 Running Hume emotion analysis...")
    hume_result = asyncio.run(run_hume_async(conversation_audio))

    formatted_output = parse_hume_predictions(hume_result["predictions"])
    formatted_output += "\n" + "-" * 100 + "\n"
    formatted_output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    hume_output.write_text(formatted_output, encoding="utf-8")
    print(f"✅ Hume emotion output saved to {hume_output}")
    return hume_output


def process_conversation(conversation_number: int, language: str):
    folder = BASE_FOLDER / f"Conversation_{conversation_number}"
    if not folder.exists():
        print(f"⚠️ Conversation_{conversation_number} not found, skipping.")
        return False

    print("==============================================")
    print(f"Processing Conversation_{conversation_number}")
    print("==============================================")

    timestamped_path, deepgram_plain_path = ensure_deepgram_transcripts(folder, language)
    hume_output_path = run_hume(folder)

    final_transcript_path = folder / "final_transcript.txt"
    merge_transcripts(deepgram_plain_path, hume_output_path, final_transcript_path)

    print("\nCompleted Conversation_{conversation_number}:")
    print(f"  - {timestamped_path}")
    print(f"  - {deepgram_plain_path}")
    print(f"  - {hume_output_path}")
    print(f"  - {final_transcript_path}")
    print()
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the final transcript workflow for one or more conversations."
    )
    parser.add_argument(
        "--conversation",
        type=int,
        help="Single conversation number to process (e.g. 59).",
    )
    parser.add_argument(
        "--conversations",
        nargs="+",
        type=int,
        help="Specific conversation numbers to process (e.g. 63 64 65).",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive range of conversation numbers to process (e.g. 63 80).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        type=int,
        default=[],
        help="Conversation numbers to exclude when using --range or --conversations.",
    )
    parser.add_argument(
        "--language",
        default=LANGUAGE,
        help="Language code for Deepgram/Speechmatics (default: es).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.conversation is not None:
        candidates = {args.conversation}
    elif args.conversations:
        candidates = set(args.conversations)
    elif args.range:
        start, end = args.range
        if start > end:
            start, end = end, start
        candidates = set(range(start, end + 1))
    else:
        candidates = {DEFAULT_CONVERSATION_NUMBER}

    exclude = set(args.exclude or [])
    conversation_numbers = sorted(num for num in candidates if num not in exclude)

    if not conversation_numbers:
        raise SystemExit("No conversations selected to process.")

    print("==============================================")
    print("Final Transcript Runner")
    print(f"Base folder: {BASE_FOLDER}")
    print(f"Conversations: {conversation_numbers}")
    if exclude:
        print(f"Excluded: {sorted(exclude)}")
    print(f"Language: {args.language}")
    print("==============================================")

    processed = 0
    failed = 0
    for conv_num in conversation_numbers:
        if process_conversation(conv_num, args.language):
            processed += 1
        else:
            failed += 1

    print("==============================================")
    print("All done!")
    print(f"✅ Successful: {processed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total attempted: {len(conversation_numbers)}")


if __name__ == "__main__":
    main()