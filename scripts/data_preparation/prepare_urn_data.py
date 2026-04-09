import json
import os
import re
import pandas as pd
from pydub import AudioSegment
from urllib.parse import unquote
from collections import defaultdict

# ================= CONFIGURATION =================
# Path to your folder containing all .wav and .json files
DATA_DIR = "./urn_data"
# Output folder
OUTPUT_DIR = "./processed_urn"
# We only want tiers tagged with this language
TARGET_LANG = "urn" 
# =================================================

def parse_target_info(url_id):
    """
    Parses the confusing ELAN ID string.
    Input: "file:///C:/Users/.../urn-a-kenari-aks-c-all.wav#t=830.311,835.445"
    Output: ("urn-a-kenari-aks-c-all.wav", 830311, 835445)
    """
    try:
        # 1. Split filename from time
        parts = url_id.split('#t=')
        if len(parts) != 2:
            return None, None, None
            
        url_part = parts[0]
        time_part = parts[1]
        
        # 2. Extract Filename from URL (handle %20 spaces etc)
        # Split by '/' and take the last part
        filename = unquote(url_part.split('/')[-1])
        
        # 3. Parse Time
        start_sec, end_sec = time_part.split(',')
        start_ms = int(float(start_sec) * 1000)
        end_ms = int(float(end_sec) * 1000)
        
        return filename, start_ms, end_ms
    except Exception as e:
        print(f"Error parsing ID {url_id}: {e}")
        return None, None, None

def clean_text(text):
    # Regex to clean text, allowing apostrophes and hyphens
    text = re.sub(r"[^\w\s'\-\?]", '', text).lower().strip()
    return text

def process_data():
    audio_out_dir = os.path.join(OUTPUT_DIR, "audio")
    os.makedirs(audio_out_dir, exist_ok=True)
    
    # Dictionary to group segments by their Source WAV file
    # Key: "filename.wav", Value: List of segment dicts
    segments_by_wav = defaultdict(list)
    
    # ---------------------------------------------------------
    # PHASE 1: Scan JSONs and build the Cut List
    # ---------------------------------------------------------
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    print(f"Phase 1: Scanning {len(json_files)} JSON files for '{TARGET_LANG}' segments...")

    total_segments_found = 0

    for json_file in json_files:
        json_path = os.path.join(DATA_DIR, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            collections = data.get('contains', [])
            
            for collection in collections:
                items = collection.get('first', {}).get('items', [])
                tier_label = collection.get('label', 'unknown')

                for item in items:
                    # 1. Language Filter
                    body = item.get('body', {})
                    if body.get('language') != TARGET_LANG:
                        continue
                    
                    text_value = body.get('value', '').strip()
                    if not text_value: 
                        continue

                    # 2. Target Parsing (Find which WAV and what time)
                    target = item.get('target')
                    if isinstance(target, list): target = target[0] # Handle list case
                    if not target: continue
                        
                    target_id = target.get('id', '')
                    
                    source_wav, start_ms, end_ms = parse_target_info(target_id)
                    
                    if not source_wav: continue

                    # 3. Add to grouping dictionary
                    segments_by_wav[source_wav].append({
                        "json_source": json_file,
                        "text": clean_text(text_value),
                        "start": start_ms,
                        "end": end_ms,
                        "speaker_tier": tier_label
                    })
                    total_segments_found += 1

        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    print(f"Found {total_segments_found} segments across {len(segments_by_wav)} unique audio files.")

    # ---------------------------------------------------------
    # PHASE 2: Load Audio Once, Cut Many Times
    # ---------------------------------------------------------
    print("\nPhase 2: Processing Audio...")
    
    manifest_data = []
    
    for wav_filename, segments in segments_by_wav.items():
        # Check if file exists in our folder
        local_wav_path = os.path.join(DATA_DIR, wav_filename)
        
        if not os.path.exists(local_wav_path):
            # Fallback: sometimes JSON says .mp4 but we have .wav
            base_name = os.path.splitext(wav_filename)[0]
            alt_path = os.path.join(DATA_DIR, base_name + ".wav")
            if os.path.exists(alt_path):
                local_wav_path = alt_path
            else:
                print(f"  [MISSING] Could not find audio file: {wav_filename} (referenced in {segments[0]['json_source']})")
                continue

        print(f"  Processing: {wav_filename} ({len(segments)} clips)...")
        
        try:
            # Load the heavy audio file into RAM once
            audio_source = AudioSegment.from_wav(local_wav_path)
            
            for seg in segments:
                duration_ms = seg['end'] - seg['start']
                if duration_ms < 100: continue # Skip glitches

                # Slice
                audio_slice = audio_source[seg['start']:seg['end']]
                
                # Construct output filename: originalWav_start_end.wav
                # Clean the wav name so it doesn't have spaces or .wav inside the string
                clean_wav_name = os.path.splitext(wav_filename)[0].replace(" ", "_")
                slice_name = f"{clean_wav_name}_{seg['start']}_{seg['end']}.wav"
                slice_path = os.path.join(audio_out_dir, slice_name)
                
                audio_slice.export(slice_path, format="wav")
                
                # Add to manifest
                manifest_data.append({
                    "audio_filepath": slice_path,
                    "text": seg['text'],
                    "duration": duration_ms / 1000.0,
                    "language": TARGET_LANG,
                    "original_file": wav_filename
                })
                
        except Exception as e:
            print(f"  [ERROR] Failed to process {wav_filename}: {e}")

    # ---------------------------------------------------------
    # PHASE 3: Save Metadata
    # ---------------------------------------------------------
    if not manifest_data:
        print("\nNo data processed.")
        return

    df = pd.DataFrame(manifest_data)
    csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    df.to_csv(csv_path, index=False)

    total_seconds = df['duration'].sum()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    print(f"\nProcessing Complete!")
    print(f"Total Clips: {len(df)}")
    print(f"Total Audio: {hours}h {minutes}m")
    print(f"Saved to: {csv_path}")

if __name__ == "__main__":
    process_data()
