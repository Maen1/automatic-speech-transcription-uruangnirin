from bs4 import BeautifulSoup
import re

# ================= CONFIGURATION =================
INPUT_FILE = "urn-dict.xhtml"  # Name of your XHTML file
OUTPUT_FILE = "dictionary_vocab.txt"
# =================================================

def clean_word(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove numbers (like "a1", "a2")
    text = re.sub(r'\d+', '', text)
    # 3. Remove non-alphabetic chars (keep apostrophes/hyphens if needed)
    text = re.sub(r"[^\w\s'\-]", '', text)
    return text.strip()

def extract_vocab():
    print(f"Reading {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Use a set to avoid duplicates
    urn_words = set()

    # Find all elements tagged with Uruangnirin language
    # Based on your file: <span lang="urn">
    spans = soup.find_all(attrs={"lang": "urn"})

    print(f"Found {len(spans)} potential entries. Cleaning...")

    for span in spans:
        # Get text
        raw_text = span.get_text()
        
        # Clean it
        word = clean_word(raw_text)
        
        # Filter out tiny garbage (single letters like 'A' used for headers)
        if len(word) > 1:
            urn_words.add(word)

    # Sort them alphabetically
    sorted_words = sorted(list(urn_words))

    # Save to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Join with commas for the Whisper prompt
        prompt_string = ", ".join(sorted_words)
        f.write(prompt_string)

    print(f"\nSuccess! Extracted {len(sorted_words)} unique words.")
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Preview: {sorted_words[:20]}...")

if __name__ == "__main__":
    extract_vocab()
