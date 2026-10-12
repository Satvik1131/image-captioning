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

  Step | Image Feature  | Input Sequence (padded)     | Target (integer)
  ─────┼────────────────┼─────────────────────────────┼──────────────────
   1   | VGG16 (4096,)  | [1, 0, 0, 0, 0, ...]        | 5  ("a")
   2   | VGG16 (4096,)  | [1, 5, 0, 0, 0, ...]        | 42 ("dog")
   3   | VGG16 (4096,)  | [1, 5, 42, 0, 0, ...]       | 87 ("runs")
   4   | VGG16 (4096,)  | [1, 5, 42, 87, 0, ...]      | 2  ("endseq")

WHY SPARSE INTEGER TARGETS (not one-hot)?
──────────────────────────────────────────
The original code used to_categorical() to create one-hot vectors of shape
(vocab_size,). With vocab_size ~3000-4000 and ~400k samples, storing all
one-hot targets would require:

    400,000 × 4000 × 4 bytes ≈ 6.4 GB

Sparse targets store just the integer index (1 value per sample):

    400,000 × 1 × 4 bytes ≈ 1.6 MB

Keras sparse_categorical_crossentropy accepts integer targets directly and
computes the same loss as categorical_crossentropy(one_hot). No information
is lost — this is purely a memory optimisation.

WHY SHUFFLE IN THE GENERATOR?
───────────────────────────────
Without shuffling, each batch of 64 samples comes from 1–2 images (because
one image generates ~50 samples). Correlated batches mean the gradient updates
are biased toward whatever patterns the first few images have, and the model
quickly learns to ignore the image features and output the most common language
patterns. Shuffling across a buffer of ~100 images ensures that every batch
is a diverse mixture of images → gradients contain varied visual signals →
the model is forced to condition its predictions on the image.

Syllabus Alignment:
    - Unit II  : Sparse Cross-Entropy loss (vs. one-hot Categorical)
    - Unit V   : Sequence-to-sequence modelling, teacher forcing
    - Unit III : Batch-wise training via generator (memory management)
"""

import os
import json
import pickle
import random
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEAN_CAPTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clean_captions.json")
FEATURES_PATH       = os.path.join(PROJECT_ROOT, "data", "processed", "features.pkl")
TOKENIZER_PATH      = os.path.join(PROJECT_ROOT, "models", "tokenizer.pkl")

BATCH_SIZE = 64     # Larger than original 32 — amortises per-batch overhead
                    # and provides more diverse image mixing per gradient step.


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
    Load the pre-extracted VGG16 fc2 feature dictionary from pickle.

    Returns:
        dict: { "image_filename.jpg": np.array of shape (4096,) }
              All vectors are L2-normalised (unit sphere).
    """
    with open(path, "rb") as f:
        features = pickle.load(f)
    sample_shape = next(iter(features.values())).shape
    print(f"[load] Loaded features for {len(features)} images. Shape: {sample_shape}")
    return features


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — load_tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(path):
    """
    Load the fitted Keras Tokenizer from pickle.

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

    Used to determine the padding length for all input sequences.

    Args:
        captions_dict (dict): { image_id: [caption_string, ...] }

    Returns:
        int: The maximum number of tokens in any single caption.
    """
    max_len = max(
        len(cap.split())
        for caps in captions_dict.values()
        for cap in caps
    )
    print(f"[utils] Computed max caption length: {max_len} tokens")
    return max_len


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — create_sequences
# ──────────────────────────────────────────────────────────────────────────────

def create_sequences(tokenizer, max_length, captions_list, feature_vector):
    """
    Generate all (image_feature, input_sequence, target_word) triples
    for ONE image and its list of captions.

    SPARSE TARGET FORMAT:
        target is a plain integer index, NOT a one-hot vector.
        This is used with loss='sparse_categorical_crossentropy'.

    For each caption, for each word step i:
        in_seq  = seq[0:i]     padded post → (max_length,)
        target  = seq[i]       integer index (e.g. 42 for "dog")

    Args:
        tokenizer       : Fitted Keras Tokenizer.
        max_length (int): All input sequences padded to this length.
        captions_list   : List of cleaned caption strings for one image.
        feature_vector  : np.array of shape (4096,) — VGG16 fc2 features.

    Returns:
        X1 (np.array): Image features,    shape (num_samples, 4096)
        X2 (np.array): Input sequences,   shape (num_samples, max_length)
        y  (np.array): Sparse int targets, shape (num_samples,)
    """
    X1 = []
    X2 = []
    y  = []

    for caption in captions_list:
        seq = tokenizer.texts_to_sequences([caption])[0]

        for i in range(1, len(seq)):
            in_seq     = seq[:i]
            target_idx = seq[i]

            # Post-pad to max_length (must match inference padding)
            in_seq_padded = pad_sequences(
                [in_seq],
                maxlen=max_length,
                padding='post'
            )[0]

            X1.append(feature_vector)    # (4096,)
            X2.append(in_seq_padded)     # (max_length,)
            y.append(target_idx)         # integer — sparse label

    X1 = np.array(X1)           # shape: (num_samples, 4096)
    X2 = np.array(X2)           # shape: (num_samples, max_length)
    y  = np.array(y,  dtype=np.int32)  # shape: (num_samples,) — sparse int

    return X1, X2, y


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 6 — data_generator  (SHUFFLED + BUFFERED)
# ──────────────────────────────────────────────────────────────────────────────

def data_generator(captions_dict, features_dict, tokenizer, max_length,
                   vocab_size, batch_size=BATCH_SIZE, buffer_images=100):
    """
    A Python GENERATOR that yields training batches on-the-fly,
    with per-epoch shuffling and within-buffer sample shuffling.

    SHUFFLING STRATEGY:
    ────────────────────
    1. EPOCH-LEVEL SHUFFLE:
       At the start of every pass through the image list, shuffle the list
       of image IDs. This ensures that different random orderings of images
       are seen each epoch, preventing the model from memorising the fixed
       ordering.

    2. BUFFER SHUFFLE:
       Collect samples from `buffer_images` images into a buffer, then
       shuffle that buffer before yielding batches from it. Because one
       image contributes ~50 samples, a buffer of 100 images contains
       ~5000 samples from ~100 different images. Each batch of 64 drawn
       from this buffer is therefore guaranteed to contain samples from
       many different images → diverse gradient signal.

    MEMORY:
       Only buffer_images × ~50 samples live in RAM at once.
       (~100 images × 50 × (4096 + max_len + 1) × 4 bytes ≈ 80 MB)

    SPARSE TARGETS:
       y_batch has shape (batch_size,) with dtype int32 — compatible with
       sparse_categorical_crossentropy. No one-hot expansion needed.

    Args:
        captions_dict  (dict) : { image_id: [captions] }
        features_dict  (dict) : { image_id: np.array(4096,) }
        tokenizer             : Fitted Keras Tokenizer
        max_length (int)      : Fixed sequence length for padding
        vocab_size (int)      : Vocabulary size (kept for API compatibility)
        batch_size (int)      : Samples per yielded batch
        buffer_images (int)   : Number of images to buffer before shuffling

    Yields:
        ({'image_input': X1, 'seq_input': X2}, y) batches, where:
            X1 shape : (batch_size, 4096)
            X2 shape : (batch_size, max_length)
            y  shape : (batch_size,)  — sparse integer labels
    """
    # Build a mutable list of image IDs that have both captions and features
    all_ids = [img_id for img_id in captions_dict if img_id in features_dict]

    while True:
        # ── Epoch-level shuffle: randomise image order every pass ─────────────
        random.shuffle(all_ids)

        # ── Buffer: collect samples from buffer_images images, then shuffle ───
        buf_X1, buf_X2, buf_y = [], [], []

        for image_id in all_ids:
            feature_vector = features_dict[image_id]   # (4096,)
            captions_list  = captions_dict[image_id]

            X1, X2, y = create_sequences(
                tokenizer, max_length, captions_list, feature_vector
            )

            buf_X1.extend(X1)
            buf_X2.extend(X2)
            buf_y.extend(y)

            # Once the buffer is large enough, shuffle and yield batches
            if len(buf_X1) >= buffer_images * 50:
                # Shuffle indices within the buffer for within-buffer mixing
                indices = list(range(len(buf_X1)))
                random.shuffle(indices)

                buf_X1_arr = np.array(buf_X1)[indices]
                buf_X2_arr = np.array(buf_X2)[indices]
                buf_y_arr  = np.array(buf_y,  dtype=np.int32)[indices]

                # Yield batches from the shuffled buffer
                for start in range(0, len(buf_X1_arr) - batch_size + 1, batch_size):
                    end = start + batch_size
                    yield (
                        (buf_X1_arr[start:end],    # (batch_size, 4096)
                         buf_X2_arr[start:end]),   # (batch_size, max_length)
                        buf_y_arr[start:end]        # (batch_size,) — sparse int
                    )

                # Clear buffer
                buf_X1, buf_X2, buf_y = [], [], []

        # ── Flush any remaining samples in the buffer ─────────────────────────
        if buf_X1:
            indices = list(range(len(buf_X1)))
            random.shuffle(indices)

            buf_X1_arr = np.array(buf_X1)[indices]
            buf_X2_arr = np.array(buf_X2)[indices]
            buf_y_arr  = np.array(buf_y, dtype=np.int32)[indices]

            for start in range(0, len(buf_X1_arr) - batch_size + 1, batch_size):
                end = start + batch_size
                yield (
                    (buf_X1_arr[start:end],
                     buf_X2_arr[start:end]),
                    buf_y_arr[start:end]
                )

            buf_X1, buf_X2, buf_y = [], [], []


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 7 — get_train_val_split
# ──────────────────────────────────────────────────────────────────────────────

def get_train_val_split(captions_dict, features_dict, val_split=0.1):
    """
    Split image IDs into training and validation sets.

    We split on IMAGE IDs (not individual captions) to ensure that all 5
    captions of a given image are either ALL in train or ALL in val.
    This prevents data leakage.

    Args:
        captions_dict (dict): Full { image_id: [captions] } dict.
        features_dict (dict): Full { image_id: feature_vector } dict.
        val_split (float)   : Fraction of images for validation. Default 0.1.

    Returns:
        train_captions, val_captions, train_features, val_features (dicts)
    """
    common_ids = [
        img_id for img_id in captions_dict
        if img_id in features_dict
    ]

    np.random.seed(42)
    np.random.shuffle(common_ids)

    n_val     = int(len(common_ids) * val_split)
    val_ids   = set(common_ids[:n_val])
    train_ids = set(common_ids[n_val:])

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

    steps_per_epoch = ceil(total_word_step_samples / batch_size)

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
            total_samples += len(seq) - 1   # each sequence of length n → n-1 samples

    steps = int(np.ceil(total_samples / batch_size))
    print(f"[steps] Total word-step samples : {total_samples:,}")
    print(f"[steps] Steps per epoch (bs={batch_size}) : {steps:,}")
    return steps


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — sanity check
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Load all data, build a small demonstration batch, and print shapes.
    Verifies the full data pipeline before training starts.
    """
    # Load all data artifacts
    captions_dict = load_clean_captions(CLEAN_CAPTIONS_PATH)
    features_dict = load_features(FEATURES_PATH)
    tokenizer     = load_tokenizer(TOKENIZER_PATH)

    vocab_size = len(tokenizer.word_index) + 1
    max_length = get_max_length(captions_dict)

    print(f"\n[config] vocab_size = {vocab_size}")
    print(f"[config] max_length = {max_length}")
    print(f"[config] batch_size = {BATCH_SIZE}")

    train_caps, val_caps, train_feats, val_feats = get_train_val_split(
        captions_dict, features_dict, val_split=0.1
    )

    print("\n--- Training set ---")
    train_steps = calculate_steps(train_caps, train_feats, tokenizer, BATCH_SIZE)
    print("\n--- Validation set ---")
    val_steps = calculate_steps(val_caps, val_feats, tokenizer, BATCH_SIZE)

    # Pull one batch from the generator and verify shapes
    print("\n[verify] Pulling one batch from data_generator...")
    gen = data_generator(
        train_caps, train_feats, tokenizer, max_length, vocab_size, BATCH_SIZE
    )
    [X1_batch, X2_batch], y_batch = next(gen)

    # Count unique image sources in the batch (diversity check)
    unique_feats = len(set(tuple(x) for x in X1_batch[:, :10]))  # first 10 dims as proxy

    print(f"\n  X1_batch shape (image features) : {X1_batch.shape}")
    # Expected: (64, 4096)
    print(f"  X2_batch shape (input sequences): {X2_batch.shape}")
    # Expected: (64, max_length)
    print(f"  y_batch  shape (sparse targets) : {y_batch.shape}")
    # Expected: (64,)  — integer indices, not one-hot
    print(f"  y_batch  dtype                  : {y_batch.dtype}")
    # Expected: int32
    print(f"  Unique feature vectors in batch : {unique_feats} (higher = more diverse)")

    print("\n" + "="*60)
    print("  DATA PREPROCESSING VERIFIED")
    print("="*60)
    print(f"  Train images       : {len(train_caps)}")
    print(f"  Val   images       : {len(val_caps)}")
    print(f"  Train steps/epoch  : {train_steps}")
    print(f"  Val   steps/epoch  : {val_steps}")
    print(f"  Batch shape [X1]   : {X1_batch.shape}")   # (64, 4096)
    print(f"  Batch shape [X2]   : {X2_batch.shape}")   # (64, max_length)
    print(f"  Batch shape [y ]   : {y_batch.shape}")    # (64,)
    print("="*60)
    print("\nNext step -> run src/train.py")


if __name__ == "__main__":
    main()
