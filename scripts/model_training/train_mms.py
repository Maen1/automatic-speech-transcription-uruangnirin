import os
import torch
import json
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Union, Any
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)
import evaluate
import re
# ================= CONFIGURATION =================
# You are loading the 1B model based on logs. 
# If you want faster training, change to "facebook/mms-300m"
MODEL_ID = "facebook/mms-1b-all" 
OUTPUT_DIR = "./mms-1b-uruangnirin-128"
VOCAB_FILE = "./vocab.json"
# =================================================

def main():
    # 1. Load Data
    data_files = {
        "train": "./processed_urn/train.csv", 
        "test": "./processed_urn/test.csv"
    }
    dataset = load_dataset("csv", data_files=data_files)
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

    # 2. Processors
    tokenizer = Wav2Vec2CTCTokenizer(
        VOCAB_FILE, 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        word_delimiter_token="|"
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=True
    )
    
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 3. Prepare Data
    def prepare_dataset(batch):
        audio = batch["audio_filepath"]
        batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
        text = batch["text"].lower()                     # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)              # Remove all punctuation
        text = text.strip()
        # Encode text to IDs
        with processor.as_target_processor():
            batch["labels"] = processor(text).input_ids
        return batch

    print("Preprocessing data...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)

    # --- NEW: Filter out bad data that crashes CTC ---
    # The audio length (in samples) divided by 320 must be greater than text length
    def filter_inputs(batch):
        # input_values length / 320 > labels length
        # We add a buffer of 10 to be safe
        audio_len = len(batch["input_values"])
        text_len = len(batch["labels"])
        return (audio_len / 320) > (text_len + 5)

    print(f"Original Train Size: {len(dataset['train'])}")
    dataset["train"] = dataset["train"].filter(filter_inputs, num_proc=4)
    print(f"Filtered Train Size: {len(dataset['train'])} (Removed risky files)")
    # -------------------------------------------------

    # 4. Data Collator
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding=True)
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = pred_logits.argmax(-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 5. Load Model
    print(f"Loading Model: {MODEL_ID}")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID, 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        layerdrop=0.1,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    
    model.freeze_feature_encoder()

    # 6. Training Args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        # --- STABILITY CHANGES ---
        per_device_train_batch_size=2, # Reduced to prevent OOM on 1B model
        gradient_accumulation_steps=8, # Increased to keep effective batch size high
        # -------------------------
        eval_strategy="steps",
        num_train_epochs=128,
        fp16=True,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Keep this fix!
        ddp_find_unused_parameters=True, # Keep this fix!
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        learning_rate=1e-4, 
        warmup_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"], 
        tokenizer=processor.feature_extractor,
    )

    print("Starting Training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
