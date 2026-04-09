import pandas as pd
import os
import re

# ================= CONFIGURATION =================
TRAIN_CSV = "./processed_urn/train.csv"
DICT_FILE = "dictionary_vocab.txt"
OUTPUT_FILE = "corpus.txt"
# =================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove anything that isn't a letter, space, apostrophe, or hyphen
    # This ensures the LM doesn't get confused by weird symbols
    text = re.sub(r"[^a-z'\-\s]", "", text)
    return text.strip()

def main():
    full_text = []

    # 1. Read from Training Data (Sentences)
    if os.path.exists(TRAIN_CSV):
        print(f"Reading training data from {TRAIN_CSV}...")
        df = pd.read_csv(TRAIN_CSV)
        # Apply cleaning to every sentence
        train_sentences = df['text'].apply(clean_text).tolist()
        full_text.extend(train_sentences)
        print(f"-> Added {len(train_sentences)} sentences from training data.")
    else:
        print("Warning: Training CSV not found.")

    # 2. Read from Dictionary (Word List)
    if os.path.exists(DICT_FILE):
        print(f"Reading dictionary from {DICT_FILE}...")
        with open(DICT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            # The dictionary file we made earlier was comma-separated
            # We replace commas with newlines to make it a list of words
            dict_words = content.replace(",", "\n")
            
            # Clean these words too just in case
            cleaned_dict_words = [clean_text(w) for w in dict_words.split()]
            full_text.extend(cleaned_dict_words)
            print(f"-> Added {len(cleaned_dict_words)} words from dictionary.")
    else:
        print("Warning: Dictionary file not found.")

    # 3. Save to Corpus File
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Join everything with a space so it looks like one giant text stream
        # This is what KenLM/pyctcdecode expects for unigram estimation
        f.write(" ".join(full_text))

    print("\nDone! You can now run test_mms_lm.py")

if __name__ == "__main__":
    main()
