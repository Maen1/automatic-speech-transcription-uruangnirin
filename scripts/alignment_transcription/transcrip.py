import torch
import sys
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ================= CONFIGURATION =================
# Path to your trained model folder
MODEL_PATH = "./whisper-uruangnirin-model-medium"
# =================================================

def transcribe(audio_path):
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Load Model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    ).to("cuda:0")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 2. Reset Config to avoid English translation loops
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # 3. Create Pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda:0",
    )

    print(f"Transcribing {audio_path}...")
    
    # 4. Run Inference (Chunking allows long files)
    result = pipe(
        audio_path,
        chunk_length_s=30, # Breaks long files into 30s chunks
        batch_size=4,
        generate_kwargs={
            "language": "indonesian", 
            "task": "transcribe"
        }
    )
    
    return result["text"]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_new_file.py <path_to_wav>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    text = transcribe(file_path)
    
    print("\n=== TRANSCRIPTION ===\n")
    print(text)
    print("\n=====================")
    
    # Optional: Save to text file
    with open(file_path + ".txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved to {file_path}.txt")
