"""
Hume API Integration Test Script

This script demonstrates how to use the Hume API to:
1. Submit audio files for emotion analysis using the Prosody model
2. Retrieve predictions and transcripts

IMPORTANT: This script has been patched to fix bugs in hume SDK v0.13.1
The fixes are located in:
- venv/lib/python3.13/site-packages/hume/expression_measurement/client.py
- venv/lib/python3.13/site-packages/hume/expression_measurement/batch/client_with_utils.py

These patches will need to be reapplied if you reinstall/upgrade the hume package.
"""

import asyncio
from hume import AsyncHumeClient
from hume.expression_measurement.batch import Face, Prosody, Models
from hume.expression_measurement.batch.types import InferenceBaseRequest
import os
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

def parse_and_display_predictions(job_predictions, output_file=None):
    """Parse Hume predictions and display in a readable format with timestamps and emotions.
    
    Args:
        job_predictions: The predictions from the Hume API
        output_file: Optional file path to save the output
    """
    import re
    
    # Prepare output lines
    lines = []
    
    lines.append("\n" + "="*100)
    lines.append("FORMATTED TRANSCRIPT WITH EMOTION ANALYSIS")
    lines.append("="*100)
    
    # Parse the prediction string
    pred_str = str(job_predictions[0]) if job_predictions else ""
    
    # Simple approach: find all ProsodyPrediction with text field  
    predictions = []
    
    # Find all occurrences of ProsodyPrediction(text='...') - this is the actual prediction start
    # We need to find the whole prediction including its emotions
    # Strategy: find where each ProsodyPrediction(text=' starts, then parse to the end of its emotions array
    
    # Find all text=' occurrences that belong to ProsodyPrediction
    text_start = 'ProsodyPrediction(text='
    segments = pred_str.split(text_start)[1:]  # Get all segments after first occurrence
    
    for segment in segments:
        # Extract text - need to handle quotes properly for strings with apostrophes
        # Find the opening quote and the matching closing quote
        if segment.startswith("'"):
            quote_char = "'"
        elif segment.startswith('"'):
            quote_char = '"'
        else:
            continue
        
        # Find the closing quote (not the one inside the string)
        # Parse character by character to handle escaped quotes
        text_start = 1  # Skip opening quote
        text_end = text_start
        while text_end < len(segment) and segment[text_end] != quote_char:
            text_end += 1
        
        if text_end >= len(segment):
            continue
        
        text = segment[text_start:text_end]
        text_offset = text_end + 1  # Skip closing quote
        
        # Extract time
        time_match = re.search(r"time=TimeInterval\(begin=([\d.]+), end=([\d.]+)\)", segment)
        if not time_match:
            continue
        begin = float(time_match.group(1))
        end = float(time_match.group(2))
        
        # Extract confidence
        conf_match = re.search(r"confidence=([\d.]+)", segment)
        confidence = float(conf_match.group(1)) if conf_match else 1.0
        
        # Extract emotions - just search the whole segment for EmotionScore
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
        full_output = "\n".join(lines)
        print(full_output)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_output)
        return
    
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
    
    # Print to console
    full_output = "\n".join(lines)
    print(full_output)
    
    # Save to file if output_file is specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_output)
                f.write("\n")
                f.write("-" * 100 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"\n✓ Output saved to: {output_file}")
        except Exception as e:
            print(f"\n✗ Error saving to file: {e}")

async def main():
    # Initialize an authenticated client
    client = AsyncHumeClient(api_key=os.getenv("HUME_API_KEY"))

    # Submit a NEW job with the correct audio model
    print("="*80)
    print("SUBMITTING NEW JOB WITH PROSODY MODEL FOR AUDIO")
    print("="*80)
    
    # Define the filepath(s) of the file(s) you would like to analyze
    local_filepaths = [
        open("example_3.wav", mode="rb"),
    ]

    # Create configurations for audio analysis using Prosody model
    prosody_config = Prosody()
    
    # Create a Models object with Prosody instead of Face
    models_chosen = Models(prosody=prosody_config)
    
    # Create a stringified object containing the configuration
    stringified_configs = InferenceBaseRequest(models=models_chosen)

    # Start an inference job and print the job_id
    job_id = await client.expression_measurement.batch.start_inference_job_from_local_file(
        json=stringified_configs, file=local_filepaths
    )
    print(f"\nJob submitted successfully! Job ID: {job_id}")
    
    # Close the file
    for f in local_filepaths:
        f.close()
    
    # Now get the predictions
    print("\n" + "="*80)
    print("FETCHING PREDICTIONS (this may take a few minutes for longer audio files)")
    print("="*80)
    
    import time
    max_wait = 600  # Wait up to 10 minutes (longer for longer audio files)
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
                    print("\n✓ Predictions received!")
                    
                    # Create output filename with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_filename = f"hume_analysis_{timestamp}.txt"
                    
                    # Parse and display in a readable format
                    parse_and_display_predictions(job_predictions, output_file=output_filename)
                    break
            
            # Check job status
            job_details = await client.expression_measurement.batch.get_job_details(
                id=job_id
            )
            status = str(job_details).split("status='")[1].split("'")[0] if "status='" in str(job_details) else "UNKNOWN"
            print(f"Status: {status}, waiting... ({waited}s/{max_wait}s)")
            
            if status == "FAILED":
                print("\nJob failed!")
                print(json.dumps(job_details, indent=2, default=str))
                break
            
            time.sleep(2)
            waited += 2
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
            waited += 2
    
    if waited >= max_wait:
        print("\nTimed out waiting for predictions.")
        
if __name__ == "__main__":
    asyncio.run(main())