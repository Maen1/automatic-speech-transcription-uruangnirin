import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio
import evaluate

# ================= CONFIGURATION =================
MODEL_PATH = "./whisper-uruangnirin-model-medium"
TEST_DATA_PATH = "./processed_urn/test.csv"
DICT_FILE = "dictionary_vocab.txt"
TRAIN_VOCAB_FILE = "vocab_prompt.txt"
# =================================================

def load_prompt():
    """
    Combines dictionary words and training words into a single prompt string.
    """
    prompt_parts = []
    
    # 1. Add Dictionary Words
    if os.path.exists(DICT_FILE):
        with open(DICT_FILE, "r", encoding="utf-8") as f:
            prompt_parts.append(f.read().strip())
        print(f"-> Loaded dictionary vocabulary from {DICT_FILE}")
    else:
        print(f"-> Warning: {DICT_FILE} not found. Skipping.")

    # 2. Add Common Training Words
    if os.path.exists(TRAIN_VOCAB_FILE):
        with open(TRAIN_VOCAB_FILE, "r", encoding="utf-8") as f:
            prompt_parts.append(f.read().strip())
        print(f"-> Loaded training vocabulary from {TRAIN_VOCAB_FILE}")
    else:
        pass

    # Combine
    full_prompt = "Uruangnirin vocabulary: " + ", ".join(prompt_parts)
    
    # Truncate (Whisper max prompt is approx 224 tokens)
    # We keep it to 800 characters to be safe.
    if len(full_prompt) > 800:
        full_prompt = full_prompt[:800]
    
    return full_prompt

def main():
    # 1. PREPARE THE PROMPT
    vocab_prompt_text = load_prompt()
    print(f"\nFinal Prompt Text (Truncated): {vocab_prompt_text[:150]}...\n")

    # 2. LOAD MODEL
    print(f"Loading model from {MODEL_PATH}...")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    ).to("cuda:0")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # Clean config
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # --- THE FIX: Convert Text Prompt to 1D Tensor on GPU ---
    # 1. Get integer IDs
    ids_list = processor.tokenizer.get_prompt_ids(vocab_prompt_text)
    
    # 2. Convert to Tensor, keep it 1D, and move to GPU
    # REMOVED .unsqueeze(0) - this was the cause of the RuntimeError
    prompt_ids = torch.tensor(ids_list, dtype=torch.long).to("cuda:0")
    
    print(f"-> Converted prompt text into {prompt_ids.shape[0]} token IDs.")
    # --------------------------------------------------

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda:0",
    )

    # 3. LOAD DATA
    print("Loading Test Data...")
    dataset = load_dataset("csv", data_files={"test": TEST_DATA_PATH})
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    test_data = dataset["test"]

    # 4. RUN INFERENCE
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    print(f"Evaluating on {len(test_data)} test segments...")

    predictions = []
    references = []

    for i, item in enumerate(test_data):
        audio = item["audio_filepath"]["array"]
        text = item["text"]
        
        # --- GENERATE WITH PROMPT IDS ---
        result = pipe(
            audio, 
            generate_kwargs={
                "language": "indonesian", 
                "task": "transcribe",
                "prompt_ids": prompt_ids # Now this is a valid 1D GPU Tensor
            }
        )
        pred_text = result["text"]
        
        predictions.append(pred_text)
        references.append(text)
        
        if i < 5:
            print(f"\nRef:  {text}")
            print(f"Pred: {pred_text}")
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} / {len(test_data)}")

    # 5. CALCULATE SCORES
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    print(f"\n==========================================")
    print(f"FINAL RESULTS (Model: {MODEL_PATH})")
    print(f"------------------------------------------")
    print(f"Word Error Rate (WER):      {wer * 100:.2f}%")
    print(f"Character Error Rate (CER): {cer * 100:.2f}%")
    print(f"==========================================")

if __name__ == "__main__":
    main()
