# Server Deployment Guide

# For Python 3.12 and PyTorch 2.10.0+cu128

## Project Structure

```
automatic-speech-transcription-uruangnirin/
├── scripts/
│ ├── data_preparation/ # Data processing scripts
│ ├── vocabulary_corpus/ # Vocabulary and corpus builders
│ ├── model_training/ # Training scripts
│ ├── testing_evaluation/ # Testing and evaluation scripts
│ └── alignment_transcription/ # Alignment and transcription
├── requirements.txt
├── README.md
└── DEPLOYMENT.md
```

## Quick Setup

### 1. Install PyTorch (CUDA 12.8)

```bash
# Option A: Using pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Option B: Using conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
```

### 2. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up CLAP-IPA

```bash
cd ..
git clone https://github.com/Maen1/clap-ipa.git
cd clap-ipa
git checkout fix-dependency-versions
pip install -e .
cd ../automatic-speech-transcription-uruangnirin
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:

```
PyTorch: 2.10.0+cu128
CUDA: True
```

## System Dependencies

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 build-essential cmake
```

### CentOS/RHEL

```bash
sudo yum install -y ffmpeg libsndfile
sudo yum groupinstall "Development Tools"
sudo yum install -y cmake
```

## Full Automated Setup Script

```bash
#!/bin/bash
set -e

echo " Setting up Uruangnirin ASR..."

# Create project directory
mkdir -p ~/projects
cd ~/projects

# Clone repository
git clone https://github.com/YOUR_USERNAME/automatic-speech-transcription-uruangnirin.git
cd automatic-speech-transcription-uruangnirin

# Create conda environment
conda create -n urn-asr python=3.12 -y
conda activate urn-asr

# Install PyTorch with CUDA 12.8
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt

# Set up CLAP-IPA
cd ..
git clone https://github.com/Maen1/clap-ipa.git
cd clap-ipa
git checkout fix-dependency-versions
pip install -e .

# Install system dependencies
echo " Installing system dependencies..."
sudo apt update
sudo apt install -y ffmpeg libsndfile1

cd ../automatic-speech-transcription-uruangnirin

echo " Setup complete!"
echo "Activate with: conda activate urn-asr"
echo "Test with: python -c 'import torch; print(torch.cuda.is_available())'"
```

Save as `setup.sh` and run:

```bash
chmod +x setup.sh
./setup.sh
```
