from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio
import evaluate
import torch

model_id = "./whisper-uruangnirin-model-medium"
print(f"Loading model: {model_id}")

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16, use_safetensors=True
).to("cuda:0")
processor = AutoProcessor.from_pretrained(model_id)

# Fix config
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device="cuda:0",
)

# Load Test Data
dataset = load_dataset("csv", data_files={"test": "./processed_urn/test.csv"})
dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
test_data = dataset["test"]

# Metric: CER instead of WER
cer_metric = evaluate.load("cer")

print("Running inference...")
predictions = []
references = []

for i, item in enumerate(test_data):
    audio = item["audio_filepath"]["array"]
    text = item["text"]
    
    # Force Indonesian to keep phonetic mapping
    result = pipe(audio, generate_kwargs={"language": "indonesian", "task": "transcribe"})
    
    predictions.append(result["text"])
    references.append(text)

cer = cer_metric.compute(predictions=predictions, references=references)

print(f"\n------------------------------------------")
print(f"FINAL Character Error Rate (CER): {cer * 100:.2f}%")
print(f"------------------------------------------")
