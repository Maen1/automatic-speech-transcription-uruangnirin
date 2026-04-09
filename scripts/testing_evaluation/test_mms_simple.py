import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, Audio
import evaluate

# ================= CONFIGURATION =================
# Path where your training saved the model
MODEL_PATH = "./mms-1b-uruangnirin-128" 
TEST_DATA = "./processed_urn/test.csv"
# =================================================

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Load Processor & Model
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to("cuda:0")

    # 2. Load Data
    print("Loading Test Data...")
    dataset = load_dataset("csv", data_files={"test": TEST_DATA})
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    test_data = dataset["test"]

    # 3. Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    print(f"Evaluating {len(test_data)} files...")
    
    predictions = []
    references = []

    # 4. Inference Loop
    for i, item in enumerate(test_data):
        audio = item["audio_filepath"]["array"]
        text = item["text"]
        
        # Prepare input
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to("cuda:0")
        
        # Forward pass (no gradient needed)
        with torch.no_grad():
            logits = model(input_values).logits
        
        # GREEDY DECODING (Take the highest probability character at each step)
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        
        predictions.append(pred_str)
        references.append(text)
        
        if i < 5:
            print(f"\nRef:  {text}")
            print(f"Pred: {pred_str}")

    # 5. Results
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    print(f"\n==========================================")
    print(f"MMS RESULTS (No Language Model)")
    print(f"Word Error Rate (WER):      {wer * 100:.2f}%")
    print(f"Character Error Rate (CER): {cer * 100:.2f}%")
    print(f"==========================================")

if __name__ == "__main__":
    main()
