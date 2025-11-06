import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI client — make sure your API key is set as an environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# File paths
deepgram_file = "deepgram_transcript.txt"
hume_file = "hume_emotion.txt"
output_file = "final_transcript.txt"

# Step 1: Read both input files
with open(deepgram_file, "r", encoding="utf-8") as f:
    deepgram_text = f.read()

with open(hume_file, "r", encoding="utf-8") as f:
    hume_text = f.read()

# Step 2: Prepare a system + user prompt for the LLM
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
  [00:00:03.2] Speaker 1: Hello there. (Emotion: Sadness, Surprise)


Now produce the final merged transcript.
"""

# Step 3: Call the model
response = client.chat.completions.create(
    model="gpt-4o-mini",  # or "gpt-4o" if you have access
    messages=[
        {"role": "system", "content": "You are a helpful assistant that merges transcripts."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3
)

# Step 4: Extract the model output
merged_transcript = response.choices[0].message.content

# Step 5: Save the result to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(merged_transcript)

print(f"✅ Final merged transcript saved to: {output_file}")
