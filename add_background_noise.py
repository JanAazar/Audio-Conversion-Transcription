"""
Script to add background noise to audio files using audiomentations.
Downloads noise samples and applies them to conversation audio.
"""

import os
import soundfile as sf
import librosa
import numpy as np
from audiomentations import AddBackgroundNoise, Compose
import urllib.request
import zipfile
import argparse

def download_noise_samples(noise_dir="noise_samples"):
    """
    Download free noise samples from the internet.
    Creates a directory with various noise types.
    """
    if os.path.exists(noise_dir) and len(os.listdir(noise_dir)) > 0:
        print(f"✅ Noise samples directory already exists: {noise_dir}")
        return noise_dir
    
    os.makedirs(noise_dir, exist_ok=True)
    
    print("🔊 Downloading noise samples...")
    
    # Download from freesound or use embedded noise generation
    # For now, we'll create a simple white noise generator
    sample_rate = 48000
    duration = 10  # 10 seconds
    
    # Create different types of noise
    noise_types = {
        'white_noise.wav': lambda: np.random.randn(int(sample_rate * duration)),
        'pink_noise.wav': lambda: np.random.randn(int(sample_rate * duration)) * np.sqrt(np.arange(1, int(sample_rate * duration) + 1)),
        'brown_noise.wav': lambda: np.cumsum(np.random.randn(int(sample_rate * duration))),
        'cafe_ambience.wav': lambda: np.random.randn(int(sample_rate * duration)) * 0.3,
        'office_noise.wav': lambda: np.random.randn(int(sample_rate * duration)) * 0.4,
    }
    
    for filename, noise_generator in noise_types.items():
        noise = noise_generator()
        # Normalize to prevent clipping
        noise = noise / np.max(np.abs(noise)) * 0.8
        filepath = os.path.join(noise_dir, filename)
        sf.write(filepath, noise, sample_rate)
        print(f"  Created: {filename}")
    
    print(f"✅ Created noise samples in: {noise_dir}")
    return noise_dir

def add_noise_to_audio(
    input_file, 
    output_file, 
    noise_dir="noise_samples",
    min_snr_db=3.0,
    max_snr_db=30.0,
    probability=1.0
):
    """
    Add background noise to an audio file.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output audio file
        noise_dir: Directory containing noise samples
        min_snr_db: Minimum signal-to-noise ratio in dB
        max_snr_db: Maximum signal-to-noise ratio in dB
        probability: Probability of applying the transform (0.0 to 1.0)
    """
    # Load the audio file (preserve stereo)
    print(f"🎵 Loading audio: {input_file}")
    audio_data, sample_rate = librosa.load(input_file, sr=None, mono=False)
    
    # Create noise augmentation transform
    transform = AddBackgroundNoise(
        sounds_path=noise_dir,
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
        p=probability
    )
    
    print(f"🔊 Adding background noise (SNR: {min_snr_db}-{max_snr_db} dB)...")
    
    # Apply the transformation
    # audiomentations only supports mono, so handle stereo by processing each channel
    if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
        # Stereo audio - process each channel separately
        print("  Processing stereo audio (both channels)...")
        left_channel = transform(audio_data[0], sample_rate=sample_rate)
        right_channel = transform(audio_data[1], sample_rate=sample_rate)
        augmented_audio = np.array([left_channel, right_channel])
    else:
        # Mono audio
        if len(audio_data.shape) > 1:
            audio_data = audio_data[0]
        augmented_audio = transform(audio_data, sample_rate=sample_rate)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Save the augmented audio
    if len(augmented_audio.shape) == 1:
        # Mono
        sf.write(output_file, augmented_audio, sample_rate)
    else:
        # Stereo - transpose to match soundfile format
        sf.write(output_file, augmented_audio.T, sample_rate)
    
    print(f"✅ Saved noisy audio: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Add background noise to audio files')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('-o', '--output', help='Path to output file (default: adds _noisy suffix)')
    parser.add_argument('--noise-dir', default='noise_samples', help='Directory with noise samples')
    parser.add_argument('--min-snr', type=float, default=5.0, help='Minimum SNR in dB')
    parser.add_argument('--max-snr', type=float, default=25.0, help='Maximum SNR in dB')
    parser.add_argument('--keep-samples', action='store_true', help='Keep downloaded noise samples')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        return
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        # Add _noisy suffix before extension
        base, ext = os.path.splitext(args.input_file)
        output_file = f"{base}_noisy{ext}"
    
    # Download noise samples if needed
    noise_dir = download_noise_samples(args.noise_dir)
    
    # Add noise to audio
    add_noise_to_audio(
        args.input_file,
        output_file,
        noise_dir,
        args.min_snr,
        args.max_snr
    )
    
    print(f"\n✅ Done! Original: {args.input_file}")
    print(f"   Noisy version: {output_file}")
    
    if not args.keep_samples:
        print(f"\n💡 Tip: Use --keep-samples to keep the noise_samples directory")

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("  python add_background_noise.py <audio_file.wav>")
        print("  python add_background_noise.py conversation.wav -o noisy_conversation.wav")
        print("  python add_background_noise.py conversation.wav --min-snr 3 --max-snr 15")
        print("\nFor help: python add_background_noise.py --help")
        print("\nExample: processing a file...")
        print("  python add_background_noise.py Recording/test/conversation.wav")
    else:
        main()

