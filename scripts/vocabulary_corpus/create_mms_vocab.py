import pandas as pd
import json
import re

# Load your training data
df = pd.read_csv("./processed_urn/train.csv")

def clean_text_for_vocab(text):
    text = text.lower()
    
    # 1. Remove numbers explicitly (fixes the "2")
    text = re.sub(r'\d+', '', text)
    
    # 2. Remove underscore explicitly (fixes the "_")
    text = text.replace("_", " ")
    
    # 3. Allow: a-z, apostrophe ('), hyphen (-), and question mark (?)
    #    Remove everything else.
    #    Note: We keep ' because Uruangnirin seems to use it (ta'biri)
    text = re.sub(r"[^a-z'\-\?]", " ", text)
    
    # 4. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 1. Extract all unique characters from the CLEANED text
# We apply the cleaning function here to see exactly what the model will see
all_text = " ".join(df["text"].apply(clean_text_for_vocab).tolist())
unique_chars = sorted(list(set(all_text)))

# 2. Create Mapping
vocab_dict = {char: i for i, char in enumerate(unique_chars)}

# 3. Handling the Special CTC Tokens
# IMPORTANT: The model needs the space to be mapped to a delimiter, usually "|"
if " " in vocab_dict:
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
else:
    # If no spaces found (unlikely), assign next available ID
    vocab_dict["|"] = len(vocab_dict)

# Add special tokens for the model
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict) + 1

# 4. Print Checks
print(f"Vocabulary Size: {len(vocab_dict)}")
print("Chars found:", sorted(vocab_dict.keys()))

if "'" not in vocab_dict:
    print("\n[WARNING] Apostrophe (') is NOT in the vocab. If your language uses glottal stops (e.g. 'ta'biri'), checking your cleaning logic.")
else:
    print("\n[OK] Apostrophe is present.")

# 5. Save
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f)
    
print("Saved cleaned vocab.json")
