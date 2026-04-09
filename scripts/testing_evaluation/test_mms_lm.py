import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, Audio
import evaluate
from pyctcdecode import build_ctcdecoder

# ================= CONFIGURATION =================
MODEL_PATH = "./mms-1b-uruangnirin-128" 
TEST_DATA = "./processed_urn/test.csv"
CORPUS_FILE = "corpus.txt" 
# =================================================

def main():
    print("Loading Model and Processor...")
    
    # 1. Load Model FIRST to get the true output size
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to("cuda:0")
    correct_vocab_size = model.config.vocab_size
    print(f"-> Model Output Size: {correct_vocab_size}")

    # 2. Load Base Processor
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    
    # --- MANUAL LABEL CONSTRUCTION ---
    # We build the list of labels the Decoder needs, matching the Model's size exactly.
    
    # 1. Create list large enough for the Tokenizer's max ID
    max_id = max(vocab.values())
    labels = [f"<unused_{i}>" for i in range(max_id + 1)]
    
    # 2. Fill in the known vocabulary
    for token, id in vocab.items():
        labels[id] = token

    print(f"-> Tokenizer Label Size: {len(labels)}")

    # 3. Truncate labels to match the Model (Fixing the 32 vs 31 issue)
    if len(labels) > correct_vocab_size:
        print(f"-> ADJUSTMENT: Truncating labels from {len(labels)} to {correct_vocab_size}.")
        labels = labels[:correct_vocab_size]

    # 4. Handle Special Tokens in the Label List
    # Replace "|" with space
    if tokenizer.word_delimiter_token_id is not None:
        if tokenizer.word_delimiter_token_id < len(labels):
            labels[tokenizer.word_delimiter_token_id] = " "
    
    # Handle CTC Blank Token (Set to empty string "")
    if tokenizer.pad_token_id is not None:
        blank_id = tokenizer.pad_token_id
        if blank_id < len(labels):
            print(f"-> CTC Blank Token ID: {blank_id} (Was: '{labels[blank_id]}')")
            labels[blank_id] = ""
        else:
            # If blank_id was cut off, assume the last remaining token is blank
            labels[-1] = ""
            print(f"-> Warning: Pad token was outside model range. Setting last token (ID {len(labels)-1}) as blank.")

    # 3. Load Dictionary Text (Language Model)
    try:
        print("Building Language Model...")
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            text_data = f.read().lower()
            unigrams = list(set(text_data.split())) 
            print(f"-> Loaded {len(unigrams)} unique words from dictionary.")
    except FileNotFoundError:
        print("Error: corpus.txt not found. Please create it first.")
        return

    # 4. Build Decoder directly
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=None, 
        unigrams=unigrams,
    )
    
    # 5. Inference
    print("Loading Test Data...")
    dataset = load_dataset("csv", data_files={"test": TEST_DATA})
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    test_data = dataset["test"]
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    predictions = []
    references = []

    print(f"Running Inference on {len(test_data)} files...")
    
    for i, item in enumerate(test_data):
        audio = item["audio_filepath"]["array"]
        
        # 1. Feature Extraction
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to("cuda:0")
        
        # 2. Acoustic Model Forward Pass
        with torch.no_grad():
            logits = model(input_values).logits
            
        # 3. Manual Decoding (Bypassing ProcessorWithLM wrapper)
        # We perform the decode directly on the numpy logits
        logits_numpy = logits.cpu().numpy()[0] # Take first item in batch
        transcription = decoder.decode(logits_numpy)
        
        predictions.append(transcription)
        references.append(item["text"])
        
        if i < 5:
            print(f"\nRef:  {item['text']}")
            print(f"Pred: {transcription}")

    # 6. Calculate Final Scores
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    
    print(f"\n==========================================")
    print(f"MMS RESULTS (+ Dictionary LM)")
    print(f"Word Error Rate (WER):      {wer * 100:.2f}%")
    print(f"Character Error Rate (CER): {cer * 100:.2f}%")
    print(f"==========================================")

if __name__ == "__main__":
    main()
