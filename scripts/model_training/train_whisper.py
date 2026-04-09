import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

# ================= CONFIGURATION =================
# We use "small" as a balance between speed and accuracy. 
# With A5000s, you could try "medium" or "large-v3" later.
MODEL_ID = "openai/whisper-medium" 
OUTPUT_DIR = "./whisper-uruangnirin-model-medium"
# We initialize the tokenizer with Indonesian as a proxy for Austronesian phonetics
LANGUAGE = "Indonesian" 
TASK = "transcribe"
# =================================================

def main():
    # 1. Load Data
    print("Loading Dataset...")
    data_files = {
        "train": "./processed_urn/train.csv",
        "validation": "./processed_urn/val.csv",
        "test": "./processed_urn/test.csv"
    }
    dataset = load_dataset("csv", data_files=data_files)

    # 2. Prepare Audio (Resample to 16kHz)
    # Whisper requires 16000Hz. dataset.cast_column handles the conversion on the fly.
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

    # 3. Load Processor (Feature Extractor + Tokenizer)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)

    # 4. Preprocessing Function
    def prepare_dataset(batch):
        # Load audio data
        audio = batch["audio_filepath"]

        # Compute log-Mel input features from audio
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Encode target text to label ids
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    print("Preprocessing dataset (this might take a moment)...")
    # We drop columns we don't need for training to save RAM
    dataset = dataset.map(
        prepare_dataset, 
        remove_columns=dataset.column_names["train"], 
        num_proc=4  # Use CPU cores for faster processing
    )

    # 5. Data Collator (Handles padding dynamically)
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Treat inputs and labels differently (audio vs text)
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # If start of sequence token is in labels, cut it (Whisper generates it automatically)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 6. Evaluation Metric (WER)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode back to text
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 7. Load Model
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    
    # Enable gradient checkpointing to save VRAM (optional with A5000s but good practice)
    model.config.use_cache = False 

    # Force language and task tokens
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # 8. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, #16, # Increase to 32 if VRAM allows
        gradient_accumulation_steps=4, #1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=8000, # 4000 steps is usually enough for ~8 hours of data. Adjust based on Val Loss.
        gradient_checkpointing=True,
        fp16=True, # Use mixed precision for A5000
        eval_strategy="steps",
        per_device_eval_batch_size=4, #8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False, # Set True if you want to upload to HuggingFace
        dataloader_num_workers=4
    )

    # 9. Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # 10. TRAIN!
    print("Starting Training...")
    trainer.train()
    
    # 11. Final Save
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
