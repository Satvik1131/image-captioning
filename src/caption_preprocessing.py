"""
src/caption_preprocessing.py
==============================
STEP 1 of the Image Captioning Pipeline:
    Load raw captions → Clean text → Add sequence tokens →
    Fit Keras Tokenizer → Save tokenizer + cleaned captions

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

from tensorflow.keras.preprocessing.text import Tokenizer  # Unit V: builds vocabulary index

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (change these at the top rather than hunting through the code)
# ──────────────────────────────────────────────────────────────────────────────

# Absolute path to project root — all other paths are derived from here
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Primary caption source: Flickr8k.token.txt  (preferred — richer, full dataset)
TOKEN_FILE   = os.path.join(PROJECT_ROOT, "data", "Flickr8k_text", "Flickr8k.token.txt")

# Fallback: data/captions.txt (kaggle-style CSV with header "image,caption")
FALLBACK_CSV = os.path.join(PROJECT_ROOT, "data", "captions.txt")

# Output paths
PROCESSED_DIR       = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR          = os.path.join(PROJECT_ROOT, "models")
CLEAN_CAPTIONS_PATH = os.path.join(PROCESSED_DIR, "clean_captions.json")
TOKENIZER_PATH      = os.path.join(MODELS_DIR,    "tokenizer.pkl")

# Special boundary tokens — the model learns to START and STOP generation here
START_TOKEN = "startseq"   # prepended to every caption
END_TOKEN   = "endseq"     # appended  to every caption


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — load_captions_from_token_file
# ──────────────────────────────────────────────────────────────────────────────

def load_captions_from_token_file(filepath):
    """
    Parse Flickr8k.token.txt into a raw mapping.

    File format (tab-separated, one caption per line):
        1000268201_693b08cb0e.jpg#0  <TAB>  A child in a pink dress ...
        1000268201_693b08cb0e.jpg#1  <TAB>  A girl going into a wooden building .
        ...

    Each image has exactly 5 captions (index #0 through #4).

    Args:
        filepath (str): Absolute path to Flickr8k.token.txt

    Returns:
        dict: { "1000268201_693b08cb0e.jpg" : ["caption1", "caption2", ...] }
    """
    captions_dict = {}  # will hold image_id -> list of raw captions

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines

            # Split on the TAB character that separates image_id#index from caption
            parts = line.split("\t")
            if len(parts) != 2:
                continue  # skip malformed lines

            image_tag, caption = parts  # e.g. "1000268201_693b08cb0e.jpg#0", "A child..."

            # Remove the "#0" / "#1" ... "#4" index suffix to get the plain filename
            image_id = image_tag.split("#")[0]  # -> "1000268201_693b08cb0e.jpg"

            # Initialise the list on first encounter of this image
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
    Parse a Kaggle-style captions.txt that has a CSV header.

    File format:
        image,caption
        1000268201_693b08cb0e.jpg,A child in a pink dress ...
        ...

    Args:
        filepath (str): Absolute path to captions.txt

    Returns:
        dict: { "image_filename.jpg" : ["caption1", "caption2", ...] }
    """
    captions_dict = {}

    with open(filepath, "r", encoding="utf-8") as f:
        next(f)  # skip the header row "image,caption"
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split only on the FIRST comma so captions with commas stay intact
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

    Cleaning pipeline (order matters):
        1. Lowercase          — so "Dog" and "dog" map to the same token
        2. Remove digits      — numbers carry little semantic meaning for captioning
        3. Remove punctuation — periods, commas etc. are noise for an LSTM vocab
        4. Collapse whitespace — normalise to single spaces
        5. Strip short tokens  — single-char tokens (except article "a") removed

    Args:
        caption (str): A raw caption string.

    Returns:
        str: A cleaned caption string (WITHOUT boundary tokens yet).

    Example:
        Input  -> "A child, 2 years old, climbing up stairs!"
        Output -> "a child years old climbing up stairs"
    """
    # Step 1: Lowercase everything
    caption = caption.lower()

    # Step 2: Remove digits (e.g. "2 dogs" -> "dogs")
    caption = re.sub(r'\d+', '', caption)

    # Step 3: Remove punctuation using str.translate
    #   string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    caption = caption.translate(str.maketrans('', '', string.punctuation))

    # Step 4: Collapse multiple whitespace characters into a single space
    caption = re.sub(r'\s+', ' ', caption)

    # Step 5: Strip leading/trailing whitespace
    caption = caption.strip()

    # Step 6: Remove single-character tokens that are NOT meaningful articles
    #   (e.g. stray letters left after removing punctuation like "s" from "dog's")
    tokens = caption.split()
    tokens = [t for t in tokens if len(t) > 1 or t in ('a',)]  # keep 'a' (article)
    caption = ' '.join(tokens)

    return caption


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — add_sequence_tokens
# ──────────────────────────────────────────────────────────────────────────────

def add_sequence_tokens(caption):
    """
    Wrap a cleaned caption with START and END boundary tokens.

    Why boundary tokens?
    ---------------------
    During training the LSTM receives the image feature + a PARTIAL caption
    and predicts the NEXT word.  The model needs to learn:
        * When to START generating  -> signalled by startseq
        * When to STOP  generating  -> the model predicts endseq and we halt

    This converts the learning task into a "given context -> predict next token"
    sequence-to-sequence problem (Unit V: LSTMs for text generation).

    Args:
        caption (str): A cleaned caption string.

    Returns:
        str: Caption wrapped with start/end tokens.

    Example:
        Input  -> "a dog runs on the grass"
        Output -> "startseq a dog runs on the grass endseq"
    """
    return f"{START_TOKEN} {caption} {END_TOKEN}"


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — clean_all_captions
# ──────────────────────────────────────────────────────────────────────────────

def clean_all_captions(raw_captions_dict):
    """
    Apply clean_caption() and add_sequence_tokens() to the entire dataset.

    Iterates over every image and every one of its 5 captions.

    Args:
        raw_captions_dict (dict): { image_id: [raw_caption, ...] }

    Returns:
        dict: { image_id: [cleaned_caption_with_tokens, ...] }
    """
    clean_dict = {}

    for image_id, captions_list in raw_captions_dict.items():
        clean_dict[image_id] = []

        for raw_cap in captions_list:
            cleaned   = clean_caption(raw_cap)          # normalise text
            tokenised = add_sequence_tokens(cleaned)    # add startseq / endseq
            clean_dict[image_id].append(tokenised)

    total_captions = sum(len(v) for v in clean_dict.values())
    print(f"[clean] Cleaned {total_captions} captions across {len(clean_dict)} images.")
    return clean_dict


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 6 — fit_tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def fit_tokenizer(clean_captions_dict):
    """
    Build and fit a Keras Tokenizer on the full cleaned caption corpus.

    The Tokenizer:
        * Assigns a unique integer index to every word in the vocabulary.
        * This index is what the Embedding layer (Unit V) looks up to convert
          words into dense vectors of fixed dimension.

    Example mapping after fitting:
        { "startseq": 1, "dog": 2, "runs": 3, ..., "endseq": N }

    The Embedding layer input shape = (batch_size, max_sequence_length)
    Each integer in that sequence is an index into the Embedding weight matrix
    of shape (vocab_size, embedding_dim).

    Args:
        clean_captions_dict (dict): { image_id: [cleaned_caption, ...] }

    Returns:
        tokenizer (Tokenizer): A fitted Keras Tokenizer object.
        vocab_size (int)      : Total number of unique tokens + 1
                                (the +1 accounts for index 0 reserved by Keras).
        max_length (int)      : Length of the longest caption (in tokens).
                                Used to pad sequences to a fixed length.
    """
    # Flatten all captions into a single list for fitting
    all_captions = []
    for captions_list in clean_captions_dict.values():
        all_captions.extend(captions_list)

    # Create the tokenizer — no filter on punctuation since we already cleaned
    tokenizer = Tokenizer(
        oov_token=None,  # no out-of-vocab token needed; we control the vocabulary
        filters=''       # disable default filters — our clean_caption() already handled it
    )

    # fit_on_texts scans every caption and builds the word->index mapping
    # After this call:  tokenizer.word_index = { "startseq":1, "a":2, "dog":3, ... }
    tokenizer.fit_on_texts(all_captions)

    # vocab_size = number of unique words + 1 (index 0 is reserved/unused by Keras)
    vocab_size = len(tokenizer.word_index) + 1

    # max_length = length of the longest caption sequence (in words/tokens)
    # We use this to pad all shorter sequences to the same length for batch training
    max_length = max(len(cap.split()) for cap in all_captions)

    print(f"[tokenizer] Vocabulary size    : {vocab_size} unique tokens")
    print(f"[tokenizer] Max caption length : {max_length} tokens")

    return tokenizer, vocab_size, max_length


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 7 — save_tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def save_tokenizer(tokenizer, path):
    """
    Persist the fitted Keras Tokenizer to disk using Python's pickle module.

    We use pickle (binary serialisation) rather than JSON because the Tokenizer
    object contains internal state (word counts, index maps) that is easiest
    to restore exactly via pickle.

    This saved file is loaded again by:
        - train.py       (to convert caption strings to integer sequences)
        - predict.py     (to convert predicted integer indices back to words)
        - app/app.py     (for the Streamlit web interface)

    Args:
        tokenizer (Tokenizer): The fitted Keras Tokenizer.
        path (str)           : Absolute file path to save the .pkl file.
    """
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[save] Tokenizer saved to: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 8 — save_clean_captions
# ──────────────────────────────────────────────────────────────────────────────

def save_clean_captions(clean_captions_dict, path):
    """
    Save the cleaned captions dictionary to a JSON file.

    JSON is used (instead of pickle) here because:
        * Human-readable — you can open it in any text editor and inspect it
        * Language-agnostic — useful for debugging or exploration in notebooks
        * dict of lists of strings maps naturally to JSON structure

    File structure:
        {
          "1000268201_693b08cb0e.jpg": [
            "startseq a child in a pink dress is climbing up stairs endseq",
            "startseq a girl going into a wooden building endseq",
            ...
          ],
          ...
        }

    Args:
        clean_captions_dict (dict): { image_id: [cleaned_caption, ...] }
        path (str)               : Absolute file path to save the .json file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_captions_dict, f, indent=2, ensure_ascii=False)
    print(f"[save] Clean captions saved to: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — run the full preprocessing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Orchestrates the full caption preprocessing pipeline:

        1. Ensure all required output directories exist
        2. Load raw captions (prefer token file, fall back to CSV)
        3. Clean + add boundary tokens to all captions
        4. Fit the Keras Tokenizer on the full corpus
        5. Save tokenizer.pkl  -> models/
        6. Save clean_captions.json -> data/processed/

    After this script runs, the next steps can proceed:
        - feature_extraction.py  uses image IDs from clean_captions.json
        - train.py               loads both tokenizer.pkl and clean_captions.json
    """

    # -- Step 0: Create output directories (won't fail if they already exist) --
    os.makedirs(PROCESSED_DIR, exist_ok=True)  # data/processed/
    os.makedirs(MODELS_DIR,    exist_ok=True)  # models/
    print("[init] Output directories confirmed.")

    # -- Step 1: Load raw captions --------------------------------------------
    if os.path.exists(TOKEN_FILE):
        # Preferred: Flickr8k.token.txt — official Flickr8k caption file
        print(f"[load] Using token file: {TOKEN_FILE}")
        raw_captions = load_captions_from_token_file(TOKEN_FILE)
    elif os.path.exists(FALLBACK_CSV):
        # Fallback: data/captions.txt — Kaggle-style CSV
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

    # -- Step 3: Fit the Keras Tokenizer on all cleaned captions --------------
    tokenizer, vocab_size, max_length = fit_tokenizer(clean_captions)

    # -- Step 4: Save the tokenizer -------------------------------------------
    save_tokenizer(tokenizer, TOKENIZER_PATH)

    # -- Step 5: Save clean captions as JSON ----------------------------------
    save_clean_captions(clean_captions, CLEAN_CAPTIONS_PATH)

    # -- Step 6: Print a summary for the user ---------------------------------
    print("\n" + "="*60)
    print("  CAPTION PREPROCESSING COMPLETE")
    print("="*60)
    print(f"  Images processed   : {len(clean_captions)}")
    print(f"  Vocabulary size    : {vocab_size} tokens")
    print(f"  Max caption length : {max_length} tokens")
    print(f"  Tokenizer saved    : {TOKENIZER_PATH}")
    print(f"  Clean captions     : {CLEAN_CAPTIONS_PATH}")
    print("="*60)
    print("\nNext step -> run src/feature_extraction.py")


if __name__ == "__main__":
    main()
