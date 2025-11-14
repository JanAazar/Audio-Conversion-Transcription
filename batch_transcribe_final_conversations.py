#!/usr/bin/env python3
"""
Batch runner for the "Transcribe the Final Conversation" workflow from app.py.

For each selected conversation folder this script will:
  1. Use Deepgram to create `timestamped_transcription.txt` (plus optional summary).
  2. Use Speechmatics to create `summary.txt` and `metadata.txt` (audio events).

The script intentionally mirrors the behaviour of the Streamlit button without
depending on Streamlit itself, so it can be executed from the command line.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from httpx import HTTPStatusError
from deepgram import DeepgramClient
from speechmatics.models import ConnectionSettings
from speechmatics.batch_client import BatchClient


load_dotenv()


def transcribe_audio(audio_path: str, output_file: str, language: str = "es"):
    """Transcribe audio file with Deepgram, mirroring app.py behaviour."""
    print(f"    🎙️ Transcribing with Deepgram: {audio_path}")

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("    ❌ DEEPGRAM_API_KEY not found. Skipping Deepgram transcription.")
        return None

    client = DeepgramClient(api_key=api_key)

    transcription_params = {
        "model": "nova-3",
        "language": language,
        "smart_format": True,
        "paragraphs": True,
        "diarize": True,
        "filler_words": True,
    }

    try:
        with open(audio_path, "rb") as audio_file:
            response = client.listen.v1.media.transcribe_file(
                request=audio_file.read(),
                **transcription_params,
            )
    except Exception as exc:
        print(f"    ❌ Deepgram transcription failed: {exc}")
        return None

    response_json = (
        response.model_dump()
        if hasattr(response, "model_dump")
        else response.dict()
    )

    debug_file = output_file.replace(".txt", "_full_response.json")
    with open(debug_file, "w", encoding="utf-8") as fh:
        json.dump(response_json, fh, indent=2, ensure_ascii=False, default=str)
    print(f"    🔍 Saved full Deepgram response to {debug_file}")

    utterances = []
    results = response_json.get("results", {}) if response_json else {}

    for channel in results.get("channels", []):
        alternatives = channel.get("alternatives", [])
        if not alternatives:
            continue

        alternative = alternatives[0]
        paragraphs_data = alternative.get("paragraphs", [])

        if isinstance(paragraphs_data, dict):
            paragraphs_list = paragraphs_data.get("paragraphs", [])
        elif isinstance(paragraphs_data, list):
            paragraphs_list = paragraphs_data
        else:
            paragraphs_list = []

        for paragraph in paragraphs_list:
            sentences = paragraph.get("sentences", [])
            speaker_id = paragraph.get("speaker", 0)
            for sentence in sentences:
                transcript = sentence.get("text", "").strip()
                if not transcript:
                    continue
                utterances.append(
                    {
                        "start": sentence.get("start", 0.0),
                        "end": sentence.get("end", 0.0),
                        "speaker": f"Speaker {int(speaker_id) + 1}",
                        "transcript": transcript,
                    }
                )

    utterances.sort(key=lambda item: item["start"])

    with open(output_file, "w", encoding="utf-8") as fh:
        for utt in utterances:
            fh.write(
                f"[{utt['start']:.3f},{utt['end']:.3f}]\t"
                f"{utt['speaker']}\t{utt['transcript']}\n"
            )

    summary_text = None
    summary = results.get("summary")
    if summary and "short" in summary:
        summary_text = summary["short"]
        summary_file = output_file.replace(
            "timestamped_transcription.txt", "summary.txt"
        )
        with open(summary_file, "w", encoding="utf-8") as fh:
            fh.write(summary_text)
        print(f"    📝 Saved Deepgram summary to {summary_file}")

    return {
        "utterances": utterances,
        "speakers": {item["speaker"] for item in utterances},
        "count": len(utterances),
        "summary": summary_text,
    }


def speechmatics_analysis(audio_path: str, output_folder: str, language: str = "es"):
    """Run Speechmatics summary + audio events, mirroring app.py behaviour."""
    print(f"    🎤 Running Speechmatics analysis: {audio_path}")

    api_key = os.getenv("SPEECHMATICS_API_KEY")
    if not api_key:
        print("    ⚠️ SPEECHMATICS_API_KEY not found. Skipping Speechmatics analysis.")
        return None

    conf = {
        "audio_events_config": {"types": ["laughter", "music", "applause"]},
        "auto_chapters_config": {},
        "summarization_config": {},
        "topic_detection_config": {},
        "transcription_config": {
            "audio_filtering_config": {"volume_threshold": 0},
            "diarization": "speaker",
            "enable_entities": True,
            "language": language,
            "operating_point": "enhanced",
        },
        "type": "transcription",
    }

    settings = ConnectionSettings(
        url="https://asr.api.speechmatics.com/v2",
        auth_token=api_key,
    )

    try:
        with BatchClient(settings) as client:
            job_id = client.submit_job(
                audio=audio_path,
                transcription_config=conf,
            )

            transcript = client.wait_for_completion(
                job_id, transcription_format="json-v2"
            )
    except HTTPStatusError as exc:
        status = exc.response.status_code
        print(
            f"    ❌ Speechmatics HTTP error ({status}): "
            f"{getattr(exc.response, 'text', str(exc))}"
        )
        return None
    except Exception as exc:
        print(f"    ❌ Speechmatics analysis failed: {exc}")
        return None

    summary = transcript.get("summary", {}).get("content", "")
    audio_events = transcript.get("audio_events", [])
    audio_event_summary = transcript.get("audio_event_summary", {})

    summary_file = os.path.join(output_folder, "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(f"    📝 Saved Speechmatics summary to {summary_file}")

    metadata_file = os.path.join(output_folder, "metadata.txt")
    with open(metadata_file, "w", encoding="utf-8") as fh:
        fh.write("=== AUDIO EVENT SUMMARY ===\n\n")
        json.dump(audio_event_summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n\n=== DETECTED AUDIO EVENTS ===\n\n")
        for event in audio_events:
            fh.write(
                f"{event['type']} from {event['start_time']} to "
                f"{event['end_time']}, confidence: {event['confidence']}\n"
            )
    print(f"    💾 Saved Speechmatics metadata to {metadata_file}")

    return {
        "summary": summary,
        "audio_events": audio_events,
        "audio_event_summary": audio_event_summary,
        "summary_file": summary_file,
        "metadata_file": metadata_file,
    }


def process_conversation(folder: Path, language: str = "es"):
    """Run Deepgram + Speechmatics for a single conversation folder."""
    conversation_path = folder / "conversation.wav"
    if not conversation_path.exists():
        print(f"  ⚠️ {conversation_path} not found, skipping.")
        return False

    print(f"\n➤ Processing {folder.name}")

    transcript_file = folder / "timestamped_transcription.txt"
    result = transcribe_audio(str(conversation_path), str(transcript_file), language)
    if not result:
        print("  ❌ Transcription failed.")
        return False
    print(
        f"  ✅ Deepgram transcription complete "
        f"({result['count']} utterances, {len(result['speakers'])} speakers)"
    )

    speechmatics_result = speechmatics_analysis(str(conversation_path), str(folder), language)
    if speechmatics_result:
        print(
            f"  ✅ Speechmatics analysis complete "
            f"({len(speechmatics_result['audio_events'])} audio events)"
        )
    else:
        print("  ⚠️ Speechmatics analysis skipped or failed.")

    return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Batch runner for the Streamlit 'Transcribe the Final Conversation' workflow."
    )
    parser.add_argument(
        "--base-folder",
        default="Recording",
        help="Base folder that contains Conversation_X subdirectories.",
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Language code to send to Deepgram and Speechmatics (default: es).",
    )
    parser.add_argument(
        "--conversations",
        nargs="+",
        type=int,
        help="Specific conversation numbers to process (e.g. 58 59 60).",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive range of conversation numbers to process (e.g. 58 80).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        type=int,
        default=[],
        help="Conversation numbers to exclude.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    base_folder = Path(args.base_folder)
    if not base_folder.exists():
        raise SystemExit(f"Base folder '{base_folder}' does not exist.")

    if args.conversations:
        candidates = set(args.conversations)
    elif args.range:
        start, end = args.range
        if start > end:
            start, end = end, start
        candidates = set(range(start, end + 1))
    else:
        raise SystemExit("Provide either --conversations or --range.")

    exclude = set(args.exclude or [])
    conversation_numbers = sorted(num for num in candidates if num not in exclude)

    print("==============================================")
    print("Batch Transcription Runner")
    print(f"Base folder: {base_folder}")
    print(f"Conversations: {conversation_numbers}")
    if exclude:
        print(f"Excluded: {sorted(exclude)}")
    print("==============================================")

    processed = 0
    failed = 0
    for conv_num in conversation_numbers:
        folder = base_folder / f"Conversation_{conv_num}"
        if process_conversation(folder, language=args.language):
            processed += 1
        else:
            failed += 1

    print("\n==============================================")
    print("Batch transcription complete.")
    print(f"✅ Successful: {processed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total attempted: {len(conversation_numbers)}")


if __name__ == "__main__":
    main()


