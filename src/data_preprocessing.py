"""
src/data_preprocessing.py
===========================
STEP 3 of the Image Captioning Pipeline:
    Build (image_feature, input_sequence) -> target_word training pairs
    and yield them in memory-efficient batches via a Python generator.

HOW SEQUENCE GENERATION WORKS (Viva Explanation):
──────────────────────────────────────────────────
Image captioning is trained as a NEXT-WORD PREDICTION task.

Given a caption like:  "startseq a dog runs endseq"
The tokenizer converts it to integer indices:
                        [1, 5, 42, 87, 2]

We then generate ONE training sample per word step:

  Step | Image Feature | Input Sequence (padded)    | Target Word (one-hot)
  ─────┼───────────────┼────────────────────────────┼──────────────────────
   1   | VGG16 (512,)  | [1, 0, 0, 0, 0, ...]       | "a"     → index 5
   2   | VGG16 (512,)  | [1, 5, 0, 0, 0, ...]       | "dog"   → index 42
   3   | VGG16 (512,)  | [1, 5, 42, 0, 0, ...]      | "runs"  → index 87
   4   | VGG16 (512,)  | [1, 5, 42, 87, 0, ...]     | "endseq"→ index 2

This is called "teacher forcing": at each step the model sees the TRUE
previous words (not its own predictions), which stabilises training.

The target word is converted to a ONE-HOT vector of length vocab_size.
The model's softmax output is compared to this one-hot via Cross-Entropy loss.

For 8091 images × 5 captions × ~10 words each ≈ 400,000 training samples.
Loading all of these into RAM at once would require several GB.
Instead, a GENERATOR yields batches lazily — only batch_size samples live
in memory at any moment.

Syllabus Alignment:
    - Unit II  : One-hot target vectors feed into Categorical Cross-Entropy
    - Unit V   : Sequence-to-sequence modelling, teacher forcing
    - Unit III : Batch-wise training via generator (memory management)
"""

import os
import json
import pickle
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences  # pads integer sequences
from tensorflow.keras.utils import to_categorical                   # integer → one-hot vector

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEAN_CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clean_captions.json")
FEATURES_PATH       = os.path.join(PROJECT_ROOT, "data", "processed", "features.pkl")
TOKENIZER_PATH      = os.path.join(PROJECT_ROOT, "models", "tokenizer.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — load_clean_captions
# ──────────────────────────────────────────────────────────────────────────────

def load_clean_captions(path):
    """
    Load the cleaned captions dictionary from JSON.

    Returns:
        dict: { "image_filename.jpg": ["startseq ... endseq", ...] }
              5 cleaned captions per image, all with boundary tokens.
    """
    with open(path, "r", encoding="utf-8") as f:
        captions_dict = json.load(f)
    print(f"[load] Loaded captions for {len(captions_dict)} images.")
    return captions_dict


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — load_features
# ──────────────────────────────────────────────────────────────────────────────

def load_features(path):
    """
    Load the pre-extracted VGG16 feature dictionary from pickle.

    Returns:
        dict: { "image_filename.jpg": np.array of shape (512,) }
    """
    with open(path, "rb") as f:
        features = pickle.load(f)
    print(f"[load] Loaded features for {len(features)} images.")
    return features


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — load_tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(path):
    """
    Load the fitted Keras Tokenizer from pickle.

    The tokenizer provides:
        tokenizer.word_index  — { "startseq":1, "a":2, "dog":3, ... }
        tokenizer.texts_to_sequences(["startseq a dog"]) → [[1, 2, 3]]

    Returns:
        tokenizer: Fitted Keras Tokenizer object.
    """
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"[load] Tokenizer loaded. Vocabulary size: {len(tokenizer.word_index) + 1}")
    return tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — get_max_length
# ──────────────────────────────────────────────────────────────────────────────

def get_max_length(captions_dict):
    """
    Compute the maximum caption length (in tokens) across all captions.

    This is used to determine the padding length for all input sequences.
    Every input sequence is padded (or truncated) to exactly this length
    so that all training samples have a uniform shape for batching.

    Why do we need a fixed length?
        Numpy arrays and Keras models require fixed-size tensors.
        pad_sequences() adds zeros (index 0) to shorter sequences
        from the LEFT by default (post-padding also available).

    Args:
        captions_dict (dict): { image_id: [caption_string, ...] }

    Returns:
        int: The maximum number of tokens in any single caption.

    Example:
        "startseq a dog runs endseq" → 5 tokens
        max_length = max over all such lengths
    """
    all_lengths = []
    for captions_list in captions_dict.values():
        for caption in captions_list:
            all_lengths.append(len(caption.split()))  # count space-separated tokens

    max_len = max(all_lengths)
    print(f"[utils] Computed max caption length: {max_len} tokens")
    return max_len


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — create_sequences
# ──────────────────────────────────────────────────────────────────────────────

def create_sequences(tokenizer, max_length, captions_list, feature_vector, vocab_size):
    """
    Generate all (image_feature, input_sequence, target_word) triples
    for ONE image and its list of captions.

    This is the core of teacher-forcing sequence generation:

    For each caption in captions_list:
        1. Convert the caption string to an integer sequence using the tokenizer.
           e.g. "startseq a dog runs endseq"  →  [1, 5, 42, 87, 2]

        2. For each position i in the sequence (except the last):
              input_seq  = all tokens UP TO position i (the "context")
              target_idx = the token AT position i+1 (the "next word to predict")

           So for [1, 5, 42, 87, 2]:
              Step 1: in=[1],          target=5   ("a")
              Step 2: in=[1,5],        target=42  ("dog")
              Step 3: in=[1,5,42],     target=87  ("runs")
              Step 4: in=[1,5,42,87],  target=2   ("endseq")

        3. Pad input_seq to max_length using POST-padding (zeros on the right).
           Turns [1, 5] → [1, 5, 0, 0, 0, ..., 0]  (length = max_length)

        4. Convert target_idx to a one-hot vector of length vocab_size.
           target=5 → [0,0,0,0,0,1,0,...,0]  (1 at position 5, 0 elsewhere)

    Args:
        tokenizer       : Fitted Keras Tokenizer with word_index populated.
        max_length (int): All input sequences padded to this fixed length.
        captions_list   : List of cleaned caption strings for one image.
                          e.g. ["startseq a dog runs endseq", ...]
        feature_vector  : np.array of shape (512,) — VGG16 features for this image.
        vocab_size (int): Total vocabulary size (for one-hot encoding).

    Returns:
        X1 (np.array): Image features,    shape (num_samples, 512)
        X2 (np.array): Input sequences,   shape (num_samples, max_length)
        y  (np.array): One-hot targets,   shape (num_samples, vocab_size)

    where num_samples = total word-steps across all captions for this image
    (roughly 5 captions × avg_caption_length words).
    """
    X1 = []  # image feature vectors (one copy per word-step)
    X2 = []  # padded partial caption sequences (growing prefix)
    y  = []  # one-hot encoded target word (next word to predict)

    for caption in captions_list:
        # Step 1: Convert caption string → list of integer indices
        # texts_to_sequences wraps input in a list, returns list of lists
        # e.g. "startseq a dog" → [[1, 5, 42]]  → we take [0] to get [1, 5, 42]
        seq = tokenizer.texts_to_sequences([caption])[0]
        # seq is now a list of ints, e.g. [1, 5, 42, 87, 2]

        # Step 2: Generate one training sample per word position
        # We iterate from index 1 to len(seq) so that:
        #   - input  = seq[0 : i]     (everything seen so far)
        #   - target = seq[i]         (the next word to predict)
        for i in range(1, len(seq)):

            # Input: the partial caption seen so far (prefix of length i)
            in_seq = seq[:i]   # e.g. i=2 → [1, 5]  ("startseq a")

            # Target: the next word (single integer index)
            target_idx = seq[i]    # e.g. seq[2] = 42  ("dog")

            # Step 3: Pad the input sequence to max_length
            # pad_sequences expects a list of sequences; we pass a list of one
            # maxlen=max_length: if len(in_seq) < max_length → pad with 0s
            # padding='post': zeros go at the END → [1, 5, 0, 0, ..., 0]
            # (post-padding works better with mask_zero=True in the Embedding layer)
            in_seq_padded = pad_sequences(
                [in_seq],           # list of one sequence
                maxlen=max_length,  # target fixed length
                padding='post'      # zeros appended at the end
            )[0]
            # in_seq_padded shape: (max_length,)  e.g. (37,)

            # Step 4: One-hot encode the target word index
            # to_categorical(5, num_classes=8769) → array of 8769 zeros with a 1 at index 5
            # Shape: (vocab_size,)
            target_one_hot = to_categorical(
                [target_idx],       # single integer index
                num_classes=vocab_size
            )[0]
            # target_one_hot shape: (vocab_size,)  e.g. (8769,)

            # Accumulate this training sample
            X1.append(feature_vector)    # same image feature repeated for each word step
            X2.append(in_seq_padded)     # growing prefix sequence, padded
            y.append(target_one_hot)     # one-hot next word

    # Convert lists to numpy arrays for Keras
    X1 = np.array(X1)   # shape: (num_samples, 512)
    X2 = np.array(X2)   # shape: (num_samples, max_length)
    y  = np.array(y)    # shape: (num_samples, vocab_size)

    return X1, X2, y


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 6 — data_generator
# ──────────────────────────────────────────────────────────────────────────────

def data_generator(captions_dict, features_dict, tokenizer, max_length,
                   vocab_size, batch_size=32):
    """
    A Python GENERATOR that yields training batches on-the-fly.

    WHY A GENERATOR? (Important for viva)
    ────────────────────────────────────────
    If we called create_sequences() for ALL 8091 images at once and stored
    the results in RAM, we'd need approximately:

        8091 images × 5 captions × ~10 word steps = ~400,000 samples
        Per sample: 512 (X1) + 37 (X2) + 8769 (y) floats
        Total: ~400,000 × 9,318 × 4 bytes ≈ 14.9 GB of RAM

    That would crash any normal laptop/desktop.

    A generator instead:
        1. Processes ONE IMAGE at a time.
        2. Accumulates its samples into a growing batch buffer.
        3. Once the buffer reaches batch_size samples, it YIELDS the batch.
        4. Execution pauses at 'yield', Keras uses the batch for one gradient update,
           then resumes — the buffer is cleared and we continue with the next image.

    This keeps memory usage constant at:
        batch_size × (512 + 37 + 8769) × 4 bytes ≈ a few MB regardless of dataset size.

    Keras model.fit() accepts generators directly via the 'generator' argument
    and the 'steps_per_epoch' parameter (how many yield calls = 1 epoch).

    Args:
        captions_dict (dict): { image_id: [caption_string, ...] }
        features_dict (dict): { image_id: np.array(512,) }
        tokenizer           : Fitted Keras Tokenizer
        max_length (int)    : Fixed sequence length for padding
        vocab_size (int)    : Vocabulary size for one-hot encoding
        batch_size (int)    : Number of training samples per yielded batch

    Yields:
        ([X1_batch, X2_batch], y_batch) — exactly what Keras model.fit() expects
            X1_batch shape: (batch_size, 512)
            X2_batch shape: (batch_size, max_length)
            y_batch  shape: (batch_size, vocab_size)
    """
    # Accumulators for the current batch
    X1_batch = []
    X2_batch = []
    y_batch  = []

    while True:
        # Keras generators must loop FOREVER (while True) —
        # model.fit() controls when to stop via steps_per_epoch and epochs.
        for image_id, captions_list in captions_dict.items():

            # Skip images that have no extracted features
            # (e.g. if an image file was missing during feature extraction)
            if image_id not in features_dict:
                continue

            feature_vector = features_dict[image_id]  # shape: (512,)

            # Generate all (X1, X2, y) triples for this one image
            X1, X2, y = create_sequences(
                tokenizer, max_length, captions_list, feature_vector, vocab_size
            )

            # Add each sample from this image into the current batch buffer
            for j in range(len(X1)):
                X1_batch.append(X1[j])   # (512,)
                X2_batch.append(X2[j])   # (max_length,)
                y_batch.append(y[j])     # (vocab_size,)

                # Once we've accumulated batch_size samples — YIELD the batch
                if len(X1_batch) == batch_size:

                    # Convert to numpy and yield
                    yield (
                        [np.array(X1_batch),   # shape: (batch_size, 512)
                         np.array(X2_batch)],  # shape: (batch_size, max_length)
                        np.array(y_batch)       # shape: (batch_size, vocab_size)
                    )

                    # Clear the buffer for the next batch
                    X1_batch = []
                    X2_batch = []
                    y_batch  = []


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 7 — get_train_val_split
# ──────────────────────────────────────────────────────────────────────────────

def get_train_val_split(captions_dict, features_dict, val_split=0.1):
    """
    Split image IDs into training and validation sets.

    We split on IMAGE IDs (not individual captions) to ensure that all 5
    captions of a given image are either ALL in train or ALL in val.
    This prevents data leakage (where the model sees test images during training).

    Args:
        captions_dict (dict): Full { image_id: [captions] } dict.
        features_dict (dict): Full { image_id: feature_vector } dict.
        val_split     (float): Fraction of images to use for validation.
                               Default 0.1 = 10% val, 90% train.

    Returns:
        train_captions (dict): { image_id: [captions] } for training
        val_captions   (dict): { image_id: [captions] } for validation
        train_features (dict): { image_id: feature } for training
        val_features   (dict): { image_id: feature } for validation
    """
    # Only use image IDs that have BOTH captions AND features
    common_ids = [
        img_id for img_id in captions_dict
        if img_id in features_dict
    ]

    # Shuffle deterministically for reproducibility
    np.random.seed(42)
    np.random.shuffle(common_ids)

    # Split point: e.g. 10% of 8091 = ~809 validation images
    n_val = int(len(common_ids) * val_split)
    val_ids   = set(common_ids[:n_val])
    train_ids = set(common_ids[n_val:])

    # Build split dictionaries
    train_captions = {k: captions_dict[k] for k in train_ids}
    val_captions   = {k: captions_dict[k] for k in val_ids}
    train_features = {k: features_dict[k] for k in train_ids}
    val_features   = {k: features_dict[k] for k in val_ids}

    print(f"[split] Train images : {len(train_captions)}")
    print(f"[split] Val   images : {len(val_captions)}")

    return train_captions, val_captions, train_features, val_features


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 8 — calculate_steps
# ──────────────────────────────────────────────────────────────────────────────

def calculate_steps(captions_dict, features_dict, tokenizer, batch_size):
    """
    Calculate steps_per_epoch for model.fit().

    steps_per_epoch tells Keras how many generator.yield() calls
    constitute one full pass over the training data (one epoch).

    Formula:
        total_samples = sum of word-steps across all images
        steps = ceil(total_samples / batch_size)

    We count word-steps by summing (len(seq) - 1) for each caption,
    where -1 is because the last token has no "next word" to predict.

    Args:
        captions_dict (dict): { image_id: [captions] }
        features_dict (dict): { image_id: feature }
        tokenizer           : Fitted Keras Tokenizer
        batch_size (int)    : Batch size used in the generator

    Returns:
        int: Number of steps per epoch.
    """
    total_samples = 0

    for image_id, captions_list in captions_dict.items():
        if image_id not in features_dict:
            continue
        for caption in captions_list:
            seq = tokenizer.texts_to_sequences([caption])[0]
            # Each sequence of length n generates (n-1) training samples
            total_samples += len(seq) - 1

    steps = int(np.ceil(total_samples / batch_size))
    print(f"[steps] Total word-step samples : {total_samples:,}")
    print(f"[steps] Steps per epoch (bs={batch_size}) : {steps:,}")
    return steps


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Load all data, build a small demonstration batch, and print shapes.
    This verifies the full data pipeline before training starts.
    """
    BATCH_SIZE = 32

    # Load all data artifacts produced by previous pipeline steps
    captions_dict = load_clean_captions(CLEAN_CAPTIONS_PATH)
    features_dict = load_features(FEATURES_PATH)
    tokenizer     = load_tokenizer(TOKENIZER_PATH)

    # Compute derived constants
    vocab_size = len(tokenizer.word_index) + 1
    max_length = get_max_length(captions_dict)

    print(f"\n[config] vocab_size = {vocab_size}")
    print(f"[config] max_length = {max_length}")
    print(f"[config] batch_size = {BATCH_SIZE}")

    # Train/Val split
    train_caps, val_caps, train_feats, val_feats = get_train_val_split(
        captions_dict, features_dict, val_split=0.1
    )

    # Calculate steps
    print("\n--- Training set ---")
    train_steps = calculate_steps(train_caps, train_feats, tokenizer, BATCH_SIZE)
    print("\n--- Validation set ---")
    val_steps = calculate_steps(val_caps, val_feats, tokenizer, BATCH_SIZE)

    # Build the generator and pull ONE batch to verify shapes
    print("\n[verify] Pulling one batch from data_generator...")
    gen = data_generator(
        train_caps, train_feats, tokenizer, max_length, vocab_size, BATCH_SIZE
    )
    [X1_batch, X2_batch], y_batch = next(gen)

    print(f"\n  X1_batch shape (image features) : {X1_batch.shape}")
    # Expected: (32, 512)
    print(f"  X2_batch shape (input sequences): {X2_batch.shape}")
    # Expected: (32, 37)
    print(f"  y_batch  shape (one-hot targets): {y_batch.shape}")
    # Expected: (32, 8769)

    print("\n" + "="*60)
    print("  DATA PREPROCESSING VERIFIED")
    print("="*60)
    print(f"  Train images       : {len(train_caps)}")
    print(f"  Val   images       : {len(val_caps)}")
    print(f"  Train steps/epoch  : {train_steps}")
    print(f"  Val   steps/epoch  : {val_steps}")
    print(f"  Batch shape [X1]   : {X1_batch.shape}")
    print(f"  Batch shape [X2]   : {X2_batch.shape}")
    print(f"  Batch shape [y ]   : {y_batch.shape}")
    print("="*60)
    print("\nNext step -> run src/train.py")


if __name__ == "__main__":
    main()
