import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio
import evaluate

# ==========================================
# 1. Load Model and Sanitize Configuration
# ==========================================
model_id = "./whisper-uruangnirin-model-medium"

print(f"Loading model from {model_id}...")

# Load model manually to adjust configuration
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
).to("cuda:0")

processor = AutoProcessor.from_pretrained(model_id)

# --- CONFIGURATION FIX ---
# Clear any forced tokens saved during training to prevent conflicts
# and allow us to manually specify language/task below.
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
# -------------------------

# Create the pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device="cuda:0",
)

# ==========================================
# 2. Load Data
# ==========================================
print("Loading Test Set...")
dataset = load_dataset("csv", data_files={"test": "./processed_urn/test.csv"})
dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
test_data = dataset["test"]

# ==========================================
# 3. Evaluate
# ==========================================
wer_metric = evaluate.load("wer")
print(f"Evaluating on {len(test_data)} test segments...")

predictions = []
references = []

for i, item in enumerate(test_data):
    audio = item["audio_filepath"]["array"]
    text = item["text"]
    
    # --- THE FIX FOR HALLUCINATIONS ---
    # We force the model to use the "Indonesian" slots (where Uruangnirin was learned)
    # and strictly perform Transcription, not Translation.
    result = pipe(
        audio, 
        generate_kwargs={
            "language": "indonesian", 
            "task": "transcribe"
        }
    )
    pred_text = result["text"]
    
    predictions.append(pred_text)
    references.append(text)
    
    # Print first 10 examples to check quality
    if i < 10:
        print(f"\nRef:  {text}")
        print(f"Pred: {pred_text}")
    
    # Progress indicator
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {len(test_data)}")

# ==========================================
# 4. Final Score
# ==========================================
wer = wer_metric.compute(predictions=predictions, references=references)
print(f"\n------------------------------------------")
print(f"FINAL TEST RESULT (Word Error Rate): {wer * 100:.2f}%")
print(f"------------------------------------------")
