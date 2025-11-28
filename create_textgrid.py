#!/usr/bin/env python3
"""
Function to create TextGrid files from transcription files.
Converts transcript.txt format to Praat TextGrid format.
"""

import re
from pathlib import Path
from collections import defaultdict


def parse_transcript(transcript_file):
    """
    Parse a transcript file and extract utterances with timestamps and speakers.
    
    Expected format: [start,end]	Speaker X	text
    
    Returns:
        list of dicts with keys: 'start', 'end', 'speaker', 'text'
    """
    utterances = []
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse format: [start,end]	Speaker X	text
            match = re.match(r'\[([\d.]+),([\d.]+)\]\s+Speaker\s+(\d+)\s+(.*)', line)
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                speaker_num = int(match.group(3))
                text = match.group(4).strip()
                
                utterances.append({
                    'start': start,
                    'end': end,
                    'speaker': f'Speaker {speaker_num}',
                    'text': text
                })
    
    return utterances


def create_textgrid(transcript_file, output_file):
    """
    Create a TextGrid file from a transcript file.
    
    Args:
        transcript_file: Path to input transcript file (transcript.txt or transcription.txt)
        output_file: Path to output TextGrid file
    """
    transcript_file = Path(transcript_file)
    output_file = Path(output_file)
    
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")
    
    # Parse transcript
    utterances = parse_transcript(transcript_file)
    
    if not utterances:
        raise ValueError(f"No utterances found in transcript file: {transcript_file}")
    
    # Find total duration (use max end time)
    total_duration = max(utt['end'] for utt in utterances)
    
    # Group utterances by speaker
    speaker_utterances = defaultdict(list)
    for utt in utterances:
        speaker_utterances[utt['speaker']].append(utt)
    
    # Get sorted speaker names
    speakers = sorted(speaker_utterances.keys(), key=lambda x: int(x.split()[-1]))
    
    # Create intervals for each speaker tier
    speaker_intervals = {}
    for speaker in speakers:
        utts = sorted(speaker_utterances[speaker], key=lambda x: x['start'])
        intervals = []
        
        # Add initial empty interval if needed
        if utts[0]['start'] > 0:
            intervals.append({
                'start': 0.0,
                'end': utts[0]['start'],
                'text': ''
            })
        
        # Add intervals for each utterance
        prev_end = 0.0
        for utt in utts:
            # Add gap interval if there's a gap
            if prev_end < utt['start']:
                intervals.append({
                    'start': prev_end,
                    'end': utt['start'],
                    'text': ''
                })
            
            # Add utterance interval
            intervals.append({
                'start': utt['start'],
                'end': utt['end'],
                'text': utt['text']
            })
            prev_end = utt['end']
        
        # Add final empty interval if needed
        if prev_end < total_duration:
            intervals.append({
                'start': prev_end,
                'end': total_duration,
                'text': ''
            })
        
        speaker_intervals[speaker] = intervals
    
    # Write TextGrid file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n')
        f.write('\n')
        f.write(f'xmin = 0\n')
        f.write(f'xmax = {total_duration:.6f}\n')
        f.write('tiers? <exists>\n')
        f.write(f'size = {len(speakers)}\n')
        f.write('item []:\n')
        
        for tier_idx, speaker in enumerate(speakers, 1):
            intervals = speaker_intervals[speaker]
            f.write(f'    item [{tier_idx}]:\n')
            f.write('        class = "IntervalTier"\n')
            f.write(f'        name = "{speaker}"\n')
            f.write(f'        xmin = 0\n')
            f.write(f'        xmax = {total_duration:.6f}\n')
            f.write(f'        intervals: size = {len(intervals)}\n')
            
            for int_idx, interval in enumerate(intervals, 1):
                f.write(f'        intervals [{int_idx}]:\n')
                f.write(f'            xmin = {interval["start"]:.6f}\n')
                f.write(f'            xmax = {interval["end"]:.6f}\n')
                # Escape quotes in text
                text_escaped = interval['text'].replace('"', '\\"')
                f.write(f'            text = "{text_escaped}"\n')
    
    print(f"✅ Created TextGrid file: {output_file}")
    print(f"   Duration: {total_duration:.3f} seconds")
    print(f"   Speakers: {len(speakers)}")
    print(f"   Total utterances: {len(utterances)}")

