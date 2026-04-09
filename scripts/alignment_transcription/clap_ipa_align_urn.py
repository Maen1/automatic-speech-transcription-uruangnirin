"""
Forced Alignment Script for URN Language Data
Performs phone-level forced alignment using IPA-Aligner pretrained models.
"""

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from librosa.sequence import dtw
from tqdm import tqdm
from transformers import AutoProcessor, DebertaV2Tokenizer

from clap.encoders import PhoneEncoder, SpeechEncoder


def create_phone_mask(lengths):
    """Create a mask for phone sequences."""
    mask = torch.zeros(len(lengths), sum(lengths))
    cumsum = 0
    for i, l in enumerate(lengths):
        mask[i, cumsum : cumsum + l] = 1 / l + 1e-8
        cumsum += l
    return mask


def create_sliding_window(lengths, win_len=10, shift=5):
    """Create sliding window for speech features."""
    L = torch.max(lengths)
    num_win = torch.ceil(L / shift).long()
    sliding_window = torch.zeros(num_win, L)
    for n in range(num_win):
        sliding_window[n, n * shift : min(n * shift + win_len, L)] = 1.0
    return sliding_window


def forced_align(cost, phone_feature):
    """Perform DTW-based forced alignment."""
    D, align = dtw(C=cost.T, step_sizes_sigma=np.array([[1, 1], [0, 1]]))

    align_seq = [-1 for i in range(max(align[:, 0]) + 1)]
    for i in list(align):
        if align_seq[i[0]] < i[1]:
            align_seq[i[0]] = i[1]

    align_id = list(align_seq)
    return [(i, p) for i, p in zip(align_id[:-1], phone_feature[1:-1])]


def load_models(device):
    """Load IPA-Aligner pretrained models."""
    print("Loading IPA-Aligner models...")
    speech_encoder = SpeechEncoder.from_pretrained("anyspeech/ipa-align-base-speech")
    phone_encoder = PhoneEncoder.from_pretrained("anyspeech/ipa-align-base-phone")

    phone_encoder.eval().to(device)
    speech_encoder.eval().to(device)

    print("Loading processors...")
    tokenizer = DebertaV2Tokenizer.from_pretrained("charsiu/IPATokenizer")
    processor = AutoProcessor.from_pretrained("openai/whisper-base")

    return speech_encoder, phone_encoder, tokenizer, processor


def align_single_audio(
    audio_path, text, speech_encoder, phone_encoder, tokenizer, processor, device, win_len=1, frame_shift=1
):
    """
    Perform forced alignment on a single audio file.

    Args:
        audio_path: Path to audio file
        text: Phonetic transcription (IPA string)
        speech_encoder: Speech encoder model
        phone_encoder: Phone encoder model
        tokenizer: IPA tokenizer
        processor: Whisper audio processor
        device: torch device
        win_len: Window length for sliding window
        frame_shift: Frame shift for sliding window

    Returns:
        List of (phone, start_time, end_time) tuples
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Process audio
    input_features = [audio]
    batch = processor(
        input_features,
        sampling_rate=16000,
        return_attention_mask=True,
        return_tensors="pt",
    )

    length = torch.max(torch.sum(batch["attention_mask"], dim=-1))
    batch["attention_mask"] = batch["attention_mask"][:, :length]
    batch["input_features"] = batch["input_features"][:, :, :length]
    speech_lengths = torch.sum(batch["attention_mask"][:, ::2], dim=-1)
    speech_sliding_mask = create_sliding_window(speech_lengths, win_len=win_len, shift=frame_shift)

    # Tokenize phone sequence
    # Split text into individual phones (assuming space-separated or continuous IPA)
    # If your text is space-separated, split by space
    # Otherwise, we need to handle it differently
    phones_list = text.strip().split()
    if len(phones_list) == 1:
        # If no spaces, treat each character as a phone (adjust if needed)
        phones_list = list(text.strip())

    augmented_y_ids = phones_list  # For phone-level alignment

    out = tokenizer(
        augmented_y_ids,
        return_attention_mask=False,
        return_length=True,
        return_token_type_ids=False,
        add_special_tokens=False,
    )
    phones = torch.tensor(list(itertools.chain.from_iterable(out["input_ids"]))).long().unsqueeze(0)
    phone_mask = create_phone_mask(out["length"])

    # Extract features
    with torch.no_grad():
        speech_features = speech_encoder(**batch.to(device)).last_hidden_state.squeeze()
        phone_features = phone_encoder(phones.to(device)).last_hidden_state.squeeze()

    # Compute similarity
    transformed_phone_features = torch.matmul(phone_mask.to(device), phone_features)
    transformed_speech_features = torch.matmul(speech_sliding_mask.to(device), speech_features)
    transformed_speech_features = F.normalize(transformed_speech_features, dim=-1)
    transformed_phone_features = F.normalize(transformed_phone_features, dim=-1)
    pairwise_cos_sim = torch.matmul(transformed_speech_features, transformed_phone_features.t())

    # Perform alignment
    alignment = forced_align(-pairwise_cos_sim.cpu().numpy(), augmented_y_ids)

    # Convert to timestamps (each frame is 0.02 seconds for Whisper)
    y_hat = [float(i * frame_shift * 0.02) for i, p in alignment]
    y_hat_ids = [p for i, p in alignment]

    # Create phone-level timestamps
    results = []
    for idx, (phone, start_time) in enumerate(zip(y_hat_ids, y_hat)):
        end_time = y_hat[idx + 1] if idx + 1 < len(y_hat) else start_time + 0.02
        results.append((phone, start_time, end_time))

    return results


def process_csv(csv_path, output_path, speech_encoder, phone_encoder, tokenizer, processor, device, win_len=1, frame_shift=1):
    """Process entire CSV file and perform forced alignment on all entries."""
    print(f"Reading CSV from {csv_path}...")

    # Read CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} audio files to align")

    # Process each row
    results = []
    for row in tqdm(rows, desc="Aligning audio files"):
        audio_path = row["audio_filepath"]
        text = row["text"]

        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}, skipping...")
            continue

        try:
            # Perform alignment
            alignment = align_single_audio(
                audio_path, text, speech_encoder, phone_encoder, tokenizer, processor, device, win_len, frame_shift
            )

            # Add results
            for phone, start_time, end_time in alignment:
                results.append(
                    {
                        "audio_filepath": audio_path,
                        "text": text,
                        "phone": phone,
                        "start_time": f"{start_time:.3f}",
                        "end_time": f"{end_time:.3f}",
                        "duration": row.get("duration", ""),
                        "language": row.get("language", ""),
                    }
                )

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    # Write output CSV
    print(f"Writing aligned results to {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["audio_filepath", "text", "phone", "start_time", "end_time", "duration", "language"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Aligned {len(results)} phones from {len(rows)} audio files")
    return results


def main():
    parser = argparse.ArgumentParser(description="Forced Alignment for URN Language Data using IPA-Aligner")
    parser.add_argument("--csv", type=str, help="Path to input CSV file (e.g., train.csv)")
    parser.add_argument("--output", type=str, default="aligned_output.csv", help="Path to output aligned CSV file")
    parser.add_argument("--audio", type=str, help="Path to single audio file for alignment")
    parser.add_argument("--text", type=str, help="Phonetic transcription for single audio file")
    parser.add_argument("--win_len", type=int, default=1, help="Window length for sliding window (default: 1)")
    parser.add_argument("--frame_shift", type=int, default=1, help="Frame shift for sliding window (default: 1)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    speech_encoder, phone_encoder, tokenizer, processor = load_models(device)

    # Single audio mode
    if args.audio and args.text:
        print(f"Aligning single audio file: {args.audio}")
        alignment = align_single_audio(
            args.audio, args.text, speech_encoder, phone_encoder, tokenizer, processor, device, args.win_len, args.frame_shift
        )

        print("\n=== Alignment Results ===")
        print(f"{'Phone':<10} {'Start (s)':<12} {'End (s)':<12}")
        print("-" * 35)
        for phone, start_time, end_time in alignment:
            print(f"{phone:<10} {start_time:<12.3f} {end_time:<12.3f}")

    # CSV mode
    elif args.csv:
        process_csv(
            args.csv, args.output, speech_encoder, phone_encoder, tokenizer, processor, device, args.win_len, args.frame_shift
        )

    else:
        print("Error: Please provide either --csv or (--audio and --text)")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
