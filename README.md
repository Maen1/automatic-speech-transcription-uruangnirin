# Uruangnirin Automatic Speech Transcription System

A complete speech recognition system for the **Uruangnirin language**, including tools for data preparation, model training, forced alignment, and transcription.

## What This Project Does

This project converts **audio recordings of Uruangnirin speech** into **written text** automatically. It's designed for linguists and researchers working with Uruangnirin language documentation.

### Key Features:
- **Automatic Transcription**: Convert audio files to text using AI models
- **Dictionary Extraction**: Build vocabulary from Uruangnirin dictionaries
- **Forced Alignment**: Match spoken sounds to written phonemes with precise timestamps
- **Model Training**: Train custom speech models (Whisper or MMS)
- **Quality Testing**: Measure transcription accuracy

---

## Prerequisites

Before you begin, ensure you have:

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended: 24GB+)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB+ free space for models and data

### Software Requirements
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: Version 3.9 or higher
- **Git**: For cloning repositories
- **CUDA**: 11.7+ (for GPU acceleration)

### External Dependencies
This project uses the **CLAP-IPA** repository for forced alignment:
- Repository: https://github.com/Maen1/clap-ipa/tree/fix-dependency-versions
- See the [Forced Alignment section](#-forced-alignment-with-clap-ipa) for setup instructions

---

## Installation

### Step 1: Clone This Repository

```bash
git clone https://github.com/Maen1/automatic-speech-transcription-uruangnirin.git
cd automatic-speech-transcription-uruangnirin
```

### Step 2: Create a Virtual Environment

**Option A: Using venv (Standard)**
```bash
# Create a new virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

**Option B: Using Conda (Recommended for Python 3.12)**
```bash
# Create conda environment with Python 3.12
conda create -n urn-asr python=3.12
conda activate urn-asr
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support FIRST
# For CUDA 12.8 (PyTorch 2.10.0):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# For CPU-only server (no GPU):
# pip install torch torchaudio

# If using conda with CUDA 12.8:
# conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

# Install all other dependencies
pip install -r requirements.txt
```

### Step 4: Set Up CLAP-IPA (Required for Forced Alignment)

```bash
# Clone the CLAP-IPA repository (sibling directory)
cd ..
git clone https://github.com/Maen1/clap-ipa.git
cd clap-ipa
git checkout fix-dependency-versions

# Install CLAP-IPA dependencies
pip install -e .
cd ../automatic-speech-transcription-uruangnirin
```

**Note**: The `clap_ipa_align_urn.py` script imports from the CLAP-IPA repository, so both repositories must be in the same parent directory.

### Step 5: Verify Installation

```bash
# Check if PyTorch can access GPU
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.10.0+cu128
CUDA available: True
```

If CUDA is available, you're ready to go!

---

## Project Structure

```
automatic-speech-transcription-uruangnirin/
├── scripts/
│ ├── data_preparation/
│ │ ├── prepare_urn_data.py # Process ELAN JSON files into training data
│ │ ├── extract_dictionary.py # Extract words from XHTML dictionary
│ │ └── create_splits.py # Split data into train/val/test sets
│ │
│ ├── vocabulary_corpus/
│ │ ├── create_mms_vocab.py # Create vocabulary for MMS model
│ │ └── build_corpus.py # Build text corpus for language model
│ │
│ ├── model_training/
│ │ ├── train_whisper.py # Train Whisper model
│ │ └── train_mms.py # Train MMS (Massively Multilingual Speech) model
│ │
│ ├── testing_evaluation/
│ │ ├── test_model.py # Test Whisper model
│ │ ├── test_cer.py # Calculate Character Error Rate
│ │ ├── test_mms_simple.py # Test MMS without language model
│ │ ├── test_mms_lm.py # Test MMS with language model
│ │ ├── test_with_full_prompt.py # Test with vocabulary prompt
│ │ └── test_with_mms_lm.py # Alternative MMS+LM test
│ │
│ └── alignment_transcription/
│ ├── clap_ipa_align_urn.py # Forced alignment using CLAP-IPA
│ ├── text_ipa.py # Text to IPA conversion helpers
│ └── transcrip.py # Transcribe new audio files
│
├── requirements.txt # Python dependencies
├── README.md # This file
├── DEPLOYMENT.md # Server deployment guide
└── vocab.json # (Auto-created) MMS vocabulary
```

---

## Quick Start Guide

This guide walks you through the complete workflow from raw audio to transcription.

### Overview of the Pipeline

```
Raw Audio + ELAN Annotations
 ↓
 [1] prepare_urn_data.py
 ↓
 Processed CSV Files
 ↓
 [2] create_splits.py
 ↓
 Train/Val/Test Splits
 ↓
 [3] create_mms_vocab.py (for MMS)
 ↓
 [4] build_corpus.py (optional, for LM)
 ↓
 [5] Train Model (Whisper or MMS)
 ↓
 [6] Test Model
 ↓
 [7] Transcribe New Audio
```

### Step 1: Prepare Your Data

**What you need:**
- Audio files (WAV format, 16kHz preferred)
- ELAN export JSON files with Uruangnirin transcriptions

**Setup:**
```bash
# Create data directory
mkdir -p urn_data
mkdir -p processed_urn

# Copy your WAV files and ELAN JSON files to urn_data/
# Example:
# cp /path/to/your/audio/*.wav urn_data/
# cp /path/to/your/elan/*.json urn_data/
```

**Run the preparation script:**
```bash
python scripts/data_preparation/prepare_urn_data.py
```

**What it does:**
- Reads ELAN JSON files
- Extracts Uruangnirin language segments (tagged with `lang="urn"`)
- Cuts audio files at the correct timestamps
- Creates `processed_urn/metadata.csv` with all segments

**Output:**
- `processed_urn/audio/` - Individual audio clips
- `processed_urn/metadata.csv` - Metadata with text and durations

### Step 2: Create Training Splits

```bash
python scripts/data_preparation/create_splits.py
```

**What it does:**
- Removes unintelligible segments (marked as "xxx")
- Removes very short clips (< 0.6 seconds)
- Splits data: 80% training, 10% validation, 10% testing

**Output:**
- `processed_urn/train.csv` - For training the model
- `processed_urn/val.csv` - For validation during training
- `processed_urn/test.csv` - For final evaluation

### Step 3 (Optional): Extract Dictionary Vocabulary

**What you need:**
- Uruangnirin dictionary in XHTML format

**Setup:**
```bash
# Place your dictionary file in the project root
cp /path/to/urn-dict.xhtml ./urn-dict.xhtml
```

**Run:**
```bash
python scripts/vocabulary_corpus/extract_dictionary.py
```

**What it does:**
- Extracts all Uruangnirin words from `<span lang="urn">` tags
- Cleans and deduplicates words
- Creates `dictionary_vocab.txt`

**Output:**
- `dictionary_vocab.txt` - Comma-separated list of Uruangnirin words

### Step 4: Create Model Vocabulary

#### For MMS Model:
```bash
python scripts/vocabulary_corpus/create_mms_vocab.py
```

**What it does:**
- Analyzes training text to find all unique characters
- Creates character-to-ID mapping
- Handles special tokens (space, unknown, padding)

**Output:**
- `vocab.json` - Vocabulary mapping for MMS training

#### For Language Model (Optional, improves MMS accuracy):
```bash
python scripts/data_preparation/extract_dictionary.py
```

**What it does:**
- Combines training sentences and dictionary words
- Creates a text corpus for language model decoding
- Cleans text (lowercase, removes numbers/symbols)

**Output:**
- `corpus.txt` - Text corpus for language model

### Step 5: Train a Model

You have **two options**: Whisper or MMS.

#### Option A: Train Whisper (Recommended for beginners)

```bash
python scripts/model_training/train_whisper.py
```

**Configuration (edit in script):**
- `MODEL_ID`: Base model to start from (default: `openai/whisper-medium`)
- `OUTPUT_DIR`: Where to save trained model (default: `./whisper-uruangnirin-model-medium`)
- `LANGUAGE`: Proxy language for phonetics (default: `Indonesian`)

**Training time:**
- Small model: ~6-12 hours on A5000 GPU
- Medium model: ~12-24 hours on A5000 GPU

**Output:**
- `whisper-uruangnirin-model-medium/` - Trained model folder

#### Option B: Train MMS (Better for low-resource languages)

```bash
python scripts/model_training/train_mms.py
```

**Configuration (edit in script):**
- `MODEL_ID`: Base model (default: `facebook/mms-1b-all`)
- `OUTPUT_DIR`: Where to save trained model (default: `./mms-1b-uruangnirin-128`)
- `VOCAB_FILE`: Your vocabulary file (default: `./vocab.json`)

**Training time:**
- MMS 1B model: ~24-48 hours on A5000 GPU
- MMS 300M model: ~12-24 hours (faster alternative)

**Output:**
- `mms-1b-uruangnirin-128/` - Trained model folder

### Step 6: Test Your Model

#### Test Whisper Model:
```bash
python scripts/testing_evaluation/test_model.py
```

#### Test MMS Model (without language model):
```bash
python scripts/testing_evaluation/test_mms_simple.py
```

#### Test MMS Model (with language model - better accuracy):
```bash
python scripts/testing_evaluation/test_mms_lm.py
```

#### Test with Vocabulary Prompt (Whisper only):
```bash
python scripts/testing_evaluation/test_with_full_prompt.py
```

**What the tests show:**
- **WER (Word Error Rate)**: Percentage of words transcribed incorrectly
- **CER (Character Error Rate)**: Percentage of characters transcribed incorrectly
- Sample predictions vs. ground truth

**Lower is better!** A WER of 20% means 80% accuracy.

### Step 7: Transcribe New Audio

Once you're happy with your model, transcribe new audio files:

```bash
python scripts/alignment_transcription/transcrip.py /path/to/your/audio.wav
```

**What it does:**
- Loads your trained model
- Processes the audio file
- Outputs transcription to console
- Saves result to `audio.wav.txt`

**Features:**
- Handles long audio files (splits into 30-second chunks)
- Automatic batching for speed

---

## Detailed Script Reference

### Data Preparation Scripts

#### `scripts/data_preparation/prepare_urn_data.py`

**Purpose:** Convert ELAN annotations into machine-readable format

**Input:**
- `urn_data/` folder containing:
 - WAV audio files
 - ELAN export JSON files

**Output:**
- `processed_urn/metadata.csv` with columns:
 - `audio_filepath`: Path to audio clip
 - `text`: Transcription
 - `duration`: Length in seconds
 - `language`: Language code ("urn")
 - `original_file`: Source audio filename

**How it works:**
1. Scans all JSON files in `urn_data/`
2. Finds segments tagged with Uruangnirin language (`lang="urn"`)
3. Parses timestamps from ELAN target IDs
4. Cuts audio clips at exact boundaries
5. Creates manifest CSV

**Common issues:**
- **Missing audio files**: Script will print warnings and skip
- **Parsing errors**: Check ELAN export format

---

#### `scripts/data_preparation/extract_dictionary.py`

**Purpose:** Build vocabulary from Uruangnirin dictionary

**Input:**
- `urn-dict.xhtml`: XHTML dictionary file

**Output:**
- `dictionary_vocab.txt`: Comma-separated word list

**Configuration:**
```python
INPUT_FILE = "urn-dict.xhtml" # Your dictionary file
OUTPUT_FILE = "dictionary_vocab.txt"
```

**How it works:**
1. Finds all `<span lang="urn">` elements
2. Extracts text content
3. Removes numbers and special characters
4. Deduplicates and sorts alphabetically

---

#### `scripts/data_preparation/create_splits.py`

**Purpose:** Divide data into training, validation, and test sets

**Input:**
- `processed_urn/metadata.csv`

**Output:**
- `processed_urn/train.csv` (80%)
- `processed_urn/val.csv` (10%)
- `processed_urn/test.csv` (10%)

**Cleaning rules:**
- Removes segments containing "xxx" (unintelligible)
- Removes clips shorter than 0.6 seconds
- Removes single-character transcriptions

**Why three splits?**
- **Train**: Model learns from this data
- **Val**: Monitor overfitting during training
- **Test**: Final unbiased accuracy measurement

---

### Vocabulary & Corpus Scripts

#### `scripts/vocabulary_corpus/create_mms_vocab.py`

**Purpose:** Create character vocabulary for MMS model

**Input:**
- `processed_urn/train.csv`

**Output:**
- `vocab.json`: Character-to-ID mapping

**Vocabulary includes:**
- All unique characters in training data
- Space character (mapped to `|` delimiter)
- Special tokens: `[UNK]` (unknown), `[PAD]` (padding)

**Important:** If your language uses apostrophes (like `ta'biri`), verify they appear in the vocab!

---

#### `scripts/vocabulary_corpus/build_corpus.py`

**Purpose:** Create text corpus for language model decoding

**Input:**
- `processed_urn/train.csv`: Training sentences
- `dictionary_vocab.txt`: Dictionary words

**Output:**
- `corpus.txt`: Space-separated text

**Usage:**
- Required by `test_mms_lm.py` for language model decoding
- Helps model predict likely word sequences

---

### Training Scripts

#### `scripts/model_training/train_whisper.py`

**Purpose:** Fine-tune OpenAI Whisper for Uruangnirin

**Key parameters:**
```python
MODEL_ID = "openai/whisper-medium" # Base model
OUTPUT_DIR = "./whisper-uruangnirin-model-medium"
LANGUAGE = "Indonesian" # Proxy for Austronesian phonetics
```

**Training settings:**
- Batch size: 4 per GPU
- Gradient accumulation: 4 (effective batch size: 16)
- Learning rate: 1e-5
- Max steps: 8000
- Evaluation: Every 500 steps

**VRAM usage:**
- Small model: ~8GB
- Medium model: ~12GB
- Large model: ~20GB

**Tips:**
- Start with `whisper-small` for faster iteration
- Monitor validation WER - stop when it plateaus
- Use `gradient_checkpointing=True` to save memory

---

#### `scripts/model_training/train_mms.py`

**Purpose:** Fine-tune Facebook's MMS model for Uruangnirin

**Key parameters:**
```python
MODEL_ID = "facebook/mms-1b-all" # Base model (or mms-300m)
OUTPUT_DIR = "./mms-1b-uruangnirin-128"
VOCAB_FILE = "./vocab.json"
```

**Training settings:**
- Batch size: 2 per GPU (MMS 1B is large!)
- Gradient accumulation: 8 (effective batch size: 16)
- Learning rate: 1e-4
- Epochs: 128
- Evaluation: Every 500 steps

**Why MMS?**
- Better for low-resource languages
- Supports 1000+ languages out of the box
- Often achieves lower WER than Whisper

**Memory optimization:**
- Uses `gradient_checkpointing` to reduce VRAM by ~40%
- Freezes feature encoder (not updated during training)

---

### Testing Scripts

#### `scripts/testing_evaluation/test_model.py`

**Purpose:** Evaluate Whisper model on test set

**Input:**
- `./whisper-uruangnirin-model-medium/`: Trained model
- `./processed_urn/test.csv`: Test data

**Output:**
- Word Error Rate (WER)
- Sample transcriptions (first 10 examples)

**Settings:**
- Forces Indonesian language tokens
- Transcription mode (not translation)

---

#### `scripts/testing_evaluation/test_mms_simple.py`

**Purpose:** Evaluate MMS model without language model

**Input:**
- `./mms-1b-uruangnirin-128/`: Trained model
- `./processed_urn/test.csv`: Test data

**Output:**
- WER and CER
- Sample transcriptions (first 5 examples)

**Decoding:**
- Uses greedy decoding (simplest, fastest)
- No language model assistance

---

#### `scripts/testing_evaluation/test_mms_lm.py`

**Purpose:** Evaluate MMS model with language model (better accuracy)

**Input:**
- `./mms-1b-uruangnirin-128/`: Trained model
- `./processed_urn/test.csv`: Test data
- `corpus.txt`: Language model corpus

**Output:**
- WER and CER (should be lower than simple test)

**How language model helps:**
- Uses dictionary words to guide decoding
- Prefers valid Uruangnirin words over nonsense

---

#### `scripts/testing_evaluation/test_with_full_prompt.py`

**Purpose:** Test Whisper with vocabulary prompt

**Input:**
- `./whisper-uruangnirin-model-medium/`: Trained model
- `dictionary_vocab.txt`: Dictionary words
- `vocab_prompt.txt`: Training vocabulary (optional)

**Output:**
- WER and CER
- Sample transcriptions

**How prompting works:**
- Combines dictionary and training words into prompt
- Gives model hints about expected vocabulary
- Limited to ~800 characters (224 tokens)

---

### Alignment & Transcription

#### `scripts/alignment_transcription/clap_ipa_align_urn.py`

**Purpose:** Perform forced alignment (match audio to phonemes with timestamps)

** Requires CLAP-IPA repository!** See [Forced Alignment section](#-forced-alignment-with-clap-ipa)

**Usage - Single Audio:**
```bash
python clap_ipa_align_urn.py \
 --audio /path/to/audio.wav \
 --text "p h o n e m e s" \
 --device cuda
```

**Usage - CSV Batch:**
```bash
python clap_ipa_align_urn.py \
 --csv processed_urn/train.csv \
 --output aligned_output.csv \
 --device cuda
```

**Input CSV format:**
```csv
audio_filepath,text,duration,language
/path/to/audio.wav,"p h o n e m e s",3.456,urn
```

**Output CSV format:**
```csv
audio_filepath,text,phone,start_time,end_time,duration,language
/path/to/audio.wav,"p h o n e m e s",p,0.000,0.120,3.456,urn
/path/to/audio.wav,"p h o n e m e s",h,0.120,0.240,3.456,urn
...
```

**Parameters:**
- `--win_len`: Window length for sliding window (default: 1)
- `--frame_shift`: Frame shift for sliding window (default: 1)
- `--device`: `cuda` or `cpu`

**What it does:**
1. Loads IPA-Aligner models
2. Extracts speech features from audio
3. Tokenizes IPA phoneme sequence
4. Uses Dynamic Time Warping (DTW) to align
5. Outputs phone-level timestamps

**Applications:**
- Phonetics research
- Speech segmentation
- Training data refinement

---

#### `scripts/alignment_transcription/text_ipa.py`

**Purpose:** Helper script for text-to-IPA conversion

**Note:** This is a template script. You need to implement:
- `text_to_ipa()`: Convert text to IPA phonemes
- `get_speech_boundaries()`: Call CLAP-IPA for alignment

**Dependencies:**
- Requires CLAP-IPA models
- May need Epitran or CharsiuG2P for G2P conversion

---

#### `scripts/alignment_transcription/transcrip.py`

**Purpose:** Transcribe new audio files with trained model

**Usage:**
```bash
python transcrip.py /path/to/audio.wav
```

**Output:**
- Prints transcription to console
- Saves to `audio.wav.txt`

**Features:**
- Handles long files (30-second chunks)
- Batch processing (batch_size=4)
- Forces Indonesian language tokens

**For multiple files:**
```bash
# Bash loop
for file in /path/to/audio/*.wav; do
 python transcrip.py "$file"
done
```

---

## Forced Alignment with CLAP-IPA

### What is Forced Alignment?

Forced alignment matches **what was said** (audio) to **how it's written** (text) at the phoneme level, producing precise timestamps for each sound.

### Step 1: Clone CLAP-IPA Repository

```bash
# Go to parent directory
cd ..

# Clone the repository
git clone https://github.com/Maen1/clap-ipa.git

# Switch to the correct branch
cd clap-ipa
git checkout fix-dependency-versions

# Install it
pip install -e .

# Go back to main project
cd ../automatic-speech-transcription-uruangnirin
```

**Important:** Both repositories must be in the same parent directory because `clap_ipa_align_urn.py` imports from `clap.encoders`.

### Step 2: Prepare IPA Transcriptions

Your text must be in IPA (International Phonetic Alphabet) format, with phonemes separated by spaces.

**Example:**
```
English "hello" → IPA: "h ə l oʊ"
```

For Uruangnirin, you'll need a grapheme-to-phoneme (G2P) tool to convert text to IPA. You can:
- Use Epitran library
- Use CharsiuG2P
- Create a custom dictionary

### Step 3: Run Alignment

**Single file:**
```bash
python scripts/alignment_transcription/clap_ipa_align_urn.py \
 --audio processed_urn/audio/sample_clip.wav \
 --text "p h o n e m e s" \
 --device cuda
```

**Batch processing:**
```bash
python scripts/alignment_transcription/clap_ipa_align_urn.py \
 --csv processed_urn/train.csv \
 --output aligned_output.csv \
 --device cuda
```

### Step 4: Use Aligned Data

The output CSV contains precise timestamps for each phoneme, useful for:
- Phonetic analysis
- Speech segmentation visualization
- Improving training data quality
- Linguistic research

---

## Model Comparison: Whisper vs MMS

| Feature | Whisper | MMS |
|---------|---------|-----|
| **Developer** | OpenAI | Facebook/Meta |
| **Best for** | High-resource languages | Low-resource languages |
| **Training speed** | Faster (smaller models) | Slower (1B params) |
| **Accuracy** | Good | Often better for rare languages |
| **Language support** | 100 languages | 1000+ languages |
| **VRAM required** | 8-20GB | 12-24GB |
| **Language model** | Built-in | Optional external LM |

### Which Should You Choose?

**Use Whisper if:**
- You want faster training
- You have limited GPU resources
- You need a quick prototype

**Use MMS if:**
- You want maximum accuracy
- Uruangnirin is very different from major languages
- You have sufficient GPU resources

**Pro tip:** Train both and compare! Use the one with lower WER on your test set.

---

## Troubleshooting

### GPU Memory Errors

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size in training script:
 ```python
 per_device_train_batch_size = 2 # or even 1
 ```

2. Increase gradient accumulation:
 ```python
 gradient_accumulation_steps = 8 # or higher
 ```

3. Enable gradient checkpointing (already enabled):
 ```python
 gradient_checkpointing = True
 ```

4. Use a smaller base model:
 - Whisper: `whisper-small` instead of `whisper-medium`
 - MMS: `mms-300m` instead of `mms-1b-all`

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'clap'`

**Solution:**
- Ensure CLAP-IPA repository is cloned in parent directory
- Run `pip install -e .` inside `clap-ipa/`
- Verify both repos are at same level:
 ```
 parent_folder/
 ├── clap-ipa/
 └── automatic-speech-transcription-uruangnirin/
 ```

### Missing Files

**Error:** `FileNotFoundError: processed_urn/train.csv`

**Solution:**
- Run `python scripts/data_preparation/prepare_urn_data.py` first
- Run `python scripts/data_preparation/create_splits.py` to create splits
- Verify folder structure exists

### Path Issues

**Error:** `Audio file not found`

**Solution:**
- Use absolute paths instead of relative paths
- Check that file paths in CSV match actual locations
- Run scripts from project root directory

### Slow Training

**Solutions:**
1. Use `num_proc=4` or higher for data preprocessing
2. Increase `per_device_train_batch_size` if VRAM allows
3. Reduce `max_steps` if you have less data
4. Use mixed precision (`fp16=True`, already enabled)

### High Error Rate (WER)

**Possible causes:**
1. **Not enough training data**: Aim for 5+ hours
2. **Poor audio quality**: Noise, overlapping speech
3. **Inconsistent transcriptions**: Check for typos in training data
4. **Wrong language setting**: Ensure `language="indonesian"` in test scripts

**Solutions:**
- Add more training data
- Clean your dataset (remove noisy clips)
- Use language model decoding (`python scripts/testing_evaluation/test_mms_lm.py`)
- Try vocabulary prompting (`python scripts/testing_evaluation/test_with_full_prompt.py`)

---

## File Format Specifications

### CSV Format

All CSV files should have these columns:

**metadata.csv:**
```csv
audio_filepath,text,duration,language,original_file
processed_urn/audio/clip_001.wav,"ta'biri",3.456,urn,original_file.wav
```

**train.csv / val.csv / test.csv:**
```csv
audio_filepath,text,duration,language
processed_urn/audio/clip_001.wav,"ta'biri",3.456,urn
```

**aligned_output.csv:**
```csv
audio_filepath,text,phone,start_time,end_time,duration,language
processed_urn/audio/clip_001.wav,"ta'biri",t,0.000,0.080,3.456,urn
processed_urn/audio/clip_001.wav,"ta'biri",a,0.080,0.160,3.456,urn
```

### Audio Requirements

**Format:**
- WAV files (uncompressed)
- 16,000 Hz sample rate (16kHz)
- Mono (single channel)
- 16-bit depth

**Duration:**
- Minimum: 0.6 seconds
- Recommended: 1-15 seconds per clip
- Maximum: No hard limit, but longer files use more memory

**Quality:**
- Clear speech (minimal background noise)
- Single speaker per clip (preferred)
- No clipping or distortion

### ELAN JSON Format

Expected structure from ELAN export:
```json
{
 "contains": [
 {
 "label": "Speaker A",
 "first": {
 "items": [
 {
 "body": {
 "language": "urn",
 "value": "ta'biri"
 },
 "target": {
 "id": "file:///path/to/audio.wav#t=830.311,835.445"
 }
 }
 ]
 }
 }
 ]
}
```

**Key fields:**
- `body.language`: Must be `"urn"` for Uruangnirin
- `body.value`: Transcription text
- `target.id`: Contains filename and timestamps

---

## Glossary

**ASR (Automatic Speech Recognition):** Technology that converts spoken language to text.

**WER (Word Error Rate):** Percentage of words transcribed incorrectly. Lower is better.
- Formula: `(Substitutions + Insertions + Deletions) / Total Words`
- Example: 20% WER = 80% word accuracy

**CER (Character Error Rate):** Percentage of characters transcribed incorrectly. More granular than WER.
- Example: 10% CER = 90% character accuracy

**CTC (Connectionist Temporal Classification):** Training method that aligns audio to text without explicit timestamps.

**G2P (Grapheme-to-Phoneme):** Conversion from written characters to speech sounds.

**IPA (International Phonetic Alphabet):** Standardized system for representing speech sounds.

**Forced Alignment:** Process of matching audio to known text at the phoneme level with timestamps.

**Language Model (LM):** System that predicts likely word sequences to improve transcription.

**VRAM (Video RAM):** Memory on your GPU. More VRAM allows larger batch sizes and models.

**Epoch:** One complete pass through all training data.

**Batch Size:** Number of audio clips processed at once.

**Learning Rate:** How quickly the model adapts during training. Too high = unstable, too low = slow.

**Validation Set:** Data used to monitor training progress (not used for learning).

**Test Set:** Hidden data used for final accuracy measurement (never seen during training).

**Fine-tuning:** Taking a pre-trained model and adapting it to a specific language or task.

**Vocabulary:** Complete set of characters/words the model knows.

**Corpus:** Collection of text used to train language models.

**DTW (Dynamic Time Warping):** Algorithm for aligning sequences of different lengths.

---

## Additional Resources

### CLAP-IPA Repository
- GitHub: https://github.com/Maen1/clap-ipa
- Branch: `fix-dependency-versions`
- Documentation: See their `forced_alignment_example.ipynb`

### Model Documentation
- Whisper: https://github.com/openai/whisper
- MMS: https://github.com/facebookresearch/fairseq/tree/main/examples/mms

### Hugging Face Transformers
- Documentation: https://huggingface.co/docs/transformers
- MMS Models: https://huggingface.co/facebook

### ELAN Software
- Download: https://archive.mpi.nl/tla/elan
- Tutorial: https://www.mpi.nl/corpus/html/elan/

---

## Tips for Best Results

1. **Data Quality > Quantity:** 5 hours of clean data beats 20 hours of noisy data
2. **Consistent Transcriptions:** Use the same spelling conventions throughout
3. **Start Small:** Test with 30 minutes of data before training on everything
4. **Monitor Validation:** Stop training when validation WER stops improving
5. **Use Language Models:** They significantly improve accuracy for rare words
6. **Check Audio:** Listen to random clips to verify quality
7. **Backup Models:** Save checkpoints regularly during training
8. **Document Everything:** Keep notes on what worked and what didn't

---

## Contributing

Found a bug or want to improve the project?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## License

This project is provided for research and linguistic documentation purposes. Please respect the rights of Uruangnirin language speakers and communities.

---

## Support

If you encounter issues:
1. Check the [Troubleshooting section](#-troubleshooting)
2. Review error messages carefully
3. Verify all dependencies are installed
4. Ensure data formats match specifications

---

**Happy transcribing! **
