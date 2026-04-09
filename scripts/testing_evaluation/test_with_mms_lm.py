from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
import torch
from datasets import load_dataset, Audio
import evaluate

MODEL_DIR = "./mms-1b-uruangnirin"
CORPUS_FILE = "corpus.txt" # The file you created in Phase 4

# 1. Create LM Processor (This is the magic part)
# We build a simple unigram/bigram model on the fly from your text file
print("Building Language Model...")
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor

# Load base processor
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

# Load your corpus text
with open(CORPUS_FILE, "r") as f:
    corpus_text = f.read().split()

# Build Decoder
decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path=None, # We use unigrams from the text list for simplicity if kenlm binary is hard
    unigrams=corpus_text,
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)

# 2. Load Model
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to("cuda:0")

# 3. Test
dataset = load_dataset("csv", data_files={"test": "./processed_urn/test.csv"})
dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
test_data = dataset["test"]

wer_metric = evaluate.load("wer")
preds = []
refs = []

print("Running Inference...")
for i, item in enumerate(test_data):
    audio = item["audio_filepath"]["array"]
    inputs = processor_with_lm(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to("cuda:0")
    
    with torch.no_grad():
        logits = model(input_values).logits
        
    # Decode with LM
    transcription = processor_with_lm.batch_decode(logits.cpu().numpy()).text[0]
    
    preds.append(transcription)
    refs.append(item["text"])
    
    if i < 5:
        print(f"Ref:  {item['text']}")
        print(f"Pred: {transcription}")

wer = wer_metric.compute(predictions=preds, references=refs)
print(f"WER with Language Model: {wer * 100:.2f}%")
