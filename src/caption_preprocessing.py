"""
src/caption_preprocessing.py
==============================
STEP 1 of the Image Captioning Pipeline:
    Load raw captions → Clean text → Filter rare words (MIN_WORD_COUNT=5) →
    Add sequence tokens → Fit Keras Tokenizer → Save tokenizer + cleaned captions

WHY FILTER RARE WORDS?
────────────────────────
Without filtering, the Flickr8k vocabulary balloons to ~8769 words.
Most of these are hapax legomena (appearing once or twice) — the model cannot
learn meaningful representations for them because there simply isn't enough
context. They add noise to the softmax output, inflate the Embedding matrix,
and increase the probability that the model's gradient signal is wasted on
words it will never reliably predict.

By keeping only words with count >= MIN_WORD_COUNT = 5, the vocabulary shrinks
to ~3000–4000 words. The softmax layer is smaller, training is faster, and
each remaining word has enough examples to be learned properly.

Rare words in a caption are replaced with the special <unk> token so the
sequence structure is preserved (rather than just deleting the word).
startseq/endseq are always kept regardless of count.

Syllabus Alignment:
    - Unit V  : Word Embeddings — the Tokenizer builds the vocabulary
                that feeds into the Embedding layer during training.
    - Unit II : Understanding the input data is prerequisite to
                defining Cross-Entropy loss over the vocabulary.

Outputs:
    data/processed/clean_captions.json  — {image_id: [caption, ...]}
    models/tokenizer.pkl                — Fitted Keras Tokenizer object
"""

import os
import re
import json
import pickle
import string
from collections import Counter

from tensorflow.keras.preprocessing.text import Tokenizer   # Unit V: builds vocabulary index

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TOKEN_FILE   = os.path.join(PROJECT_ROOT, "data", "Flickr8k_text", "Flickr8k.token.txt")
FALLBACK_CSV = os.path.join(PROJECT_ROOT, "data", "captions.txt")

PROCESSED_DIR       = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR          = os.path.join(PROJECT_ROOT, "models")
CLEAN_CAPTIONS_PATH = os.path.join(PROCESSED_DIR, "clean_captions.json")
TOKENIZER_PATH      = os.path.join(MODELS_DIR,    "tokenizer.pkl")

START_TOKEN = "startseq"
END_TOKEN   = "endseq"
UNK_TOKEN   = "unk"          # replaces rare words in captions

# Minimum word frequency — words appearing fewer than this many times are
# replaced with UNK_TOKEN. Value 5 is standard for Flickr8k vocabulary pruning.
MIN_WORD_COUNT = 5


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — load_captions_from_token_file
# ──────────────────────────────────────────────────────────────────────────────

def load_captions_from_token_file(filepath):
    """
    Parse Flickr8k.token.txt into a raw mapping.

    File format (tab-separated, one caption per line):
        1000268201_693b08cb0e.jpg#0  <TAB>  A child in a pink dress ...
        1000268201_693b08cb0e.jpg#1  <TAB>  A girl going into a wooden building .

    Each image has exactly 5 captions (index #0 through #4).

    Args:
        filepath (str): Absolute path to Flickr8k.token.txt

    Returns:
        dict: { "1000268201_693b08cb0e.jpg" : ["caption1", "caption2", ...] }
    """
    captions_dict = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue

            image_tag, caption = parts
            image_id = image_tag.split("#")[0]

            if image_id not in captions_dict:
                captions_dict[image_id] = []

            captions_dict[image_id].append(caption)

    print(f"[load] Loaded raw captions for {len(captions_dict)} images from token file.")
    return captions_dict


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — load_captions_from_csv
# ──────────────────────────────────────────────────────────────────────────────

def load_captions_from_csv(filepath):
    """
    Parse a Kaggle-style captions.txt (CSV with header "image,caption").

    Args:
        filepath (str): Absolute path to captions.txt

    Returns:
        dict: { "image_filename.jpg" : ["caption1", "caption2", ...] }
    """
    captions_dict = {}

    with open(filepath, "r", encoding="utf-8") as f:
        next(f)   # skip header row
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",", 1)
            if len(parts) != 2:
                continue

            image_id, caption = parts

            if image_id not in captions_dict:
                captions_dict[image_id] = []

            captions_dict[image_id].append(caption)

    print(f"[load] Loaded raw captions for {len(captions_dict)} images from CSV file.")
    return captions_dict


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — clean_caption
# ──────────────────────────────────────────────────────────────────────────────

def clean_caption(caption):
    """
    Apply text normalisation rules to a single caption string.

    Cleaning pipeline:
        1. Lowercase          — "Dog" and "dog" → same token
        2. Remove digits      — numbers add noise for captioning
        3. Remove punctuation — periods, commas etc. are noise for an LSTM vocab
        4. Collapse whitespace
        5. Strip short tokens  — single-char tokens (except article "a") removed

    Args:
        caption (str): A raw caption string.

    Returns:
        str: Cleaned caption WITHOUT boundary tokens.

    Example:
        "A child, 2 years old, climbing up stairs!"
        → "a child years old climbing up stairs"
    """
    caption = caption.lower()
    caption = re.sub(r'\d+', '', caption)
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = re.sub(r'\s+', ' ', caption)
    caption = caption.strip()

    tokens  = caption.split()
    tokens  = [t for t in tokens if len(t) > 1 or t in ('a',)]
    caption = ' '.join(tokens)

    return caption


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — add_sequence_tokens
# ──────────────────────────────────────────────────────────────────────────────

def add_sequence_tokens(caption):
    """
    Wrap a cleaned caption with START and END boundary tokens.

    Example:
        "a dog runs on the grass"
        → "startseq a dog runs on the grass endseq"
    """
    return f"{START_TOKEN} {caption} {END_TOKEN}"


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — clean_all_captions
# ──────────────────────────────────────────────────────────────────────────────

def clean_all_captions(raw_captions_dict):
    """
    Apply clean_caption() and add_sequence_tokens() to the entire dataset.

    NOTE: boundary tokens (startseq / endseq) are added BEFORE frequency
    filtering so they are always counted and kept in the vocabulary.

    Args:
        raw_captions_dict (dict): { image_id: [raw_caption, ...] }

    Returns:
        dict: { image_id: [cleaned_caption_with_tokens, ...] }
    """
    clean_dict = {}

    for image_id, captions_list in raw_captions_dict.items():
        clean_dict[image_id] = []
        for raw_cap in captions_list:
            cleaned   = clean_caption(raw_cap)
            tokenised = add_sequence_tokens(cleaned)
            clean_dict[image_id].append(tokenised)

    total_captions = sum(len(v) for v in clean_dict.values())
    print(f"[clean] Cleaned {total_captions} captions across {len(clean_dict)} images.")
    return clean_dict


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 6 — build_word_frequency
# ──────────────────────────────────────────────────────────────────────────────

def build_word_frequency(clean_captions_dict):
    """
    Count how many times each word appears across ALL captions.

    Args:
        clean_captions_dict (dict): { image_id: [cleaned_caption, ...] }

    Returns:
        Counter: { word: count }
    """
    freq = Counter()
    for caps in clean_captions_dict.values():
        for cap in caps:
            freq.update(cap.split())
    return freq


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 7 — filter_rare_words
# ──────────────────────────────────────────────────────────────────────────────

def filter_rare_words(clean_captions_dict, min_count=MIN_WORD_COUNT):
    """
    Replace words that appear fewer than min_count times with UNK_TOKEN.

    WHY REPLACE (not drop)?
        Dropping the word would shift word positions and shorten the sequence,
        disrupting the model's ability to learn positional context.
        Replacing with <unk> preserves sequence length and lets the model
        learn that some positions carry uncertain/rare information.

    Always-kept words: startseq, endseq (regardless of frequency).

    Args:
        clean_captions_dict (dict): { image_id: [caption_with_tokens, ...] }
        min_count (int)           : Minimum frequency threshold (default 5).

    Returns:
        dict: { image_id: [filtered_caption, ...] } with rare words → unk
    """
    freq   = build_word_frequency(clean_captions_dict)
    always_keep = {START_TOKEN, END_TOKEN}

    # Build set of words that meet the frequency threshold
    vocab = {w for w, c in freq.items() if c >= min_count} | always_keep

    rare_replaced = 0
    filtered_dict = {}

    for image_id, caps in clean_captions_dict.items():
        filtered_dict[image_id] = []
        for cap in caps:
            tokens   = cap.split()
            new_toks = []
            for tok in tokens:
                if tok in vocab:
                    new_toks.append(tok)
                else:
                    new_toks.append(UNK_TOKEN)
                    rare_replaced += 1
            filtered_dict[image_id].append(' '.join(new_toks))

    kept_vocab_size = len(vocab)
    total_words     = sum(c for c in freq.values())
    print(f"[filter] Rare-word replacement (min_count={min_count}):")
    print(f"         Original vocabulary : {len(freq):,} unique words")
    print(f"         Filtered vocabulary : {kept_vocab_size:,} unique words")
    print(f"         Rare words replaced : {rare_replaced:,} / {total_words:,} tokens")

    return filtered_dict


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 8 — fit_tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def fit_tokenizer(clean_captions_dict):
    """
    Build and fit a Keras Tokenizer on the filtered caption corpus.

    After rare-word filtering the vocabulary should be ~3000-4000 words.
    This smaller softmax makes training faster and the gradient signal cleaner.

    Args:
        clean_captions_dict (dict): { image_id: [filtered_caption, ...] }

    Returns:
        tokenizer (Tokenizer): A fitted Keras Tokenizer object.
        vocab_size (int)     : len(word_index) + 1  (index 0 reserved by Keras)
        max_length (int)     : Length of the longest caption (in tokens).
    """
    all_captions = []
    for caps in clean_captions_dict.values():
        all_captions.extend(caps)

    tokenizer = Tokenizer(
        oov_token=None,  # unk is already in captions as a string token
        filters=''       # our clean_caption() already handled normalisation
    )
    tokenizer.fit_on_texts(all_captions)

    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(cap.split()) for cap in all_captions)

    print(f"[tokenizer] Vocabulary size    : {vocab_size} unique tokens")
    print(f"[tokenizer] Max caption length : {max_length} tokens")
    print(f"            (Expected vocab ~3000-4000 after MIN_WORD_COUNT={MIN_WORD_COUNT} filter)")

    return tokenizer, vocab_size, max_length


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 9 — save_tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def save_tokenizer(tokenizer, path):
    """
    Persist the fitted Keras Tokenizer to disk using pickle.

    This saved file is loaded again by train.py, predict.py, and app/app.py.

    Args:
        tokenizer (Tokenizer): The fitted Keras Tokenizer.
        path (str)           : Absolute file path to save the .pkl file.
    """
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[save] Tokenizer saved to: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 10 — save_clean_captions
# ──────────────────────────────────────────────────────────────────────────────

def save_clean_captions(clean_captions_dict, path):
    """
    Save the cleaned (and filtered) captions dictionary to a JSON file.

    Args:
        clean_captions_dict (dict): { image_id: [filtered_caption, ...] }
        path (str)               : Absolute file path to save the .json file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_captions_dict, f, indent=2, ensure_ascii=False)
    print(f"[save] Clean captions saved to: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Full caption preprocessing pipeline:

        1. Ensure output directories exist
        2. Load raw captions (prefer token file, fall back to CSV)
        3. Clean + add startseq / endseq boundary tokens
        4. Filter rare words (count < MIN_WORD_COUNT) → replace with unk
        5. Fit the Keras Tokenizer on the filtered corpus
        6. Save tokenizer.pkl → models/
        7. Save clean_captions.json → data/processed/
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,    exist_ok=True)
    print("[init] Output directories confirmed.")

    # -- Step 1: Load raw captions --------------------------------------------
    if os.path.exists(TOKEN_FILE):
        print(f"[load] Using token file: {TOKEN_FILE}")
        raw_captions = load_captions_from_token_file(TOKEN_FILE)
    elif os.path.exists(FALLBACK_CSV):
        print(f"[load] Token file not found. Using fallback CSV: {FALLBACK_CSV}")
        raw_captions = load_captions_from_csv(FALLBACK_CSV)
    else:
        raise FileNotFoundError(
            "No caption file found!\n"
            f"  Expected: {TOKEN_FILE}\n"
            f"  Fallback: {FALLBACK_CSV}\n"
            "Please place Flickr8k.token.txt in data/Flickr8k_text/ and re-run."
        )

    # -- Step 2: Clean captions and add startseq / endseq --------------------
    clean_captions = clean_all_captions(raw_captions)

    # -- Step 3: Filter rare words (min count = MIN_WORD_COUNT = 5) ----------
    print(f"\n[filter] Filtering words with count < {MIN_WORD_COUNT}...")
    clean_captions = filter_rare_words(clean_captions, min_count=MIN_WORD_COUNT)

    # -- Step 4: Fit the Keras Tokenizer on filtered captions -----------------
    tokenizer, vocab_size, max_length = fit_tokenizer(clean_captions)

    # -- Step 5: Save the tokenizer -------------------------------------------
    save_tokenizer(tokenizer, TOKENIZER_PATH)

    # -- Step 6: Save clean captions as JSON ----------------------------------
    save_clean_captions(clean_captions, CLEAN_CAPTIONS_PATH)

    print("\n" + "="*60)
    print("  CAPTION PREPROCESSING COMPLETE")
    print("="*60)
    print(f"  Images processed   : {len(clean_captions)}")
    print(f"  Vocabulary size    : {vocab_size} tokens  (after MIN_WORD_COUNT={MIN_WORD_COUNT} filter)")
    print(f"  Max caption length : {max_length} tokens")
    print(f"  Tokenizer saved    : {TOKENIZER_PATH}")
    print(f"  Clean captions     : {CLEAN_CAPTIONS_PATH}")
    print("="*60)
    print("\nNext step -> run src/feature_extraction.py")


if __name__ == "__main__":
    main()
