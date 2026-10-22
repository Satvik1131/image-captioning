"""
src/model.py
=============
STEP 4 of the Image Captioning Pipeline:
    Define the CNN + LSTM image captioning model using Keras Functional API.

Architecture Overview (Merge Model):
    The model has TWO input branches that are merged before the output:

    Branch 1 — Image Feature Branch:
        Input:   (4096,)         — pre-extracted VGG16 fc2 feature (L2-normalised)
        Dense:   (256,) relu     — project image features into shared 256-dim space
        Dropout: (256,)          — regularisation AFTER the dense projection
                                   (dropout on raw 4096-d input is wasteful and
                                    disrupts the already-normalised features)

    Branch 2 — Sequence (Caption) Decoder:
        Input:     (max_length,) — integer-encoded partial caption (padded, post)
        Embedding: (max_length, 256) — Unit V: maps word indices → dense vectors
        Dropout:   (max_length, 256) — regularisation on embeddings
        LSTM:      (256,)        — processes sequence, returns final hidden state

    Merge:
        add([image_out, lstm_out])  — element-wise addition (both 256-dim)
        Dense(256, relu)            — combined representation
        Dense(vocab_size, softmax)  — probability over vocabulary (next word)

    Loss:
        sparse_categorical_crossentropy — Unit II: target is an integer index,
                                          NOT a one-hot vector. Mathematically
                                          identical to categorical_crossentropy
                                          but avoids allocating huge one-hot arrays.

    Optimizer:
        Adam(learning_rate=1e-3)    — Unit II: adaptive learning rate

DESIGN DECISION — Dropout AFTER Dense (not before):
    Applying Dropout(0.5) directly to the raw 4096-d fc2 input (as in the
    previous version) randomly zeroed half of a rich, high-dimensional signal
    BEFORE it could be projected. The Dense layer then had to learn from a
    crippled input every step. Moving Dropout to AFTER Dense(256) means the
    model first compresses the 4096-d signal into a meaningful 256-d space,
    then regularises that compressed representation. This is the standard
    pattern used in ResNet / VGG fine-tuning literature.

Syllabus Alignment:
    - Unit II  : Adam Optimizer, Sparse Categorical Cross-Entropy Loss
    - Unit III : Dropout regularisation (after Dense projection, on embeddings)
    - Unit IV  : VGG16 fc2 feature vector (4096,) is the input to Branch 1
    - Unit V   : Embedding layer (Word Embeddings) + LSTM (Sequence Modelling)
"""

import os
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    add,
    Rescaling,
    LayerNormalization,
)
from tensorflow.keras.optimizers import Adam

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoder import FEATURE_DIM


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION — build_model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(vocab_size, max_length):
    """
    Build and compile the image captioning model using Keras Functional API.

    Why Functional API (not Sequential)?
        We have TWO separate input branches (image features + caption sequence)
        that need to be merged. Sequential API only supports a single linear
        stack — the Functional API supports arbitrary computation graphs.

    Args:
        vocab_size  (int): Total vocabulary size = len(word_index) + 1.
                           Dynamically read from the tokenizer; NOT hard-coded.
                           Example: ~3500 after MIN_WORD_COUNT=5 filtering.

        max_length  (int): Padded sequence length from caption_preprocessing.py.
                           Example: ~37 for Flickr8k.

    Returns:
        model (tf.keras.Model): Compiled Keras model ready for training.

    Input/Output:
        Inputs  : [image_input (batch, 4096), seq_input (batch, max_length)]
        Output  : (batch, vocab_size) — softmax next-word probabilities
    """

    # ──────────────────────────────────────────────────────────────────────────
    # BRANCH 1 — Image Feature Branch
    # ──────────────────────────────────────────────────────────────────────────
    # VGG16 fc2 output: 4096-dim L2-normalised vector per image.
    # VGG16 itself is NOT part of this model; we use pre-extracted pkl features.

    image_input = Input(shape=(FEATURE_DIM,), name='image_input')
    # shape: (batch, FEATURE_DIM)  -- L2-normalised, so each value is only ~1/sqrt(dim)

    # RESCALE: an L2-normalised vector has tiny per-element values, which made the
    # image branch ~10x weaker than the LSTM branch and let the language prior win.
    # Multiplying by sqrt(dim) brings values back to ~1.
    image_scaled = Rescaling(float(FEATURE_DIM) ** 0.5, name='image_rescale')(image_input)

    image_dense = Dense(256, activation='relu', name='image_dense')(image_scaled)
    # LayerNorm keeps both branches on the same scale before they are added
    image_norm = LayerNormalization(name='image_ln')(image_dense)
    image_out = Dropout(0.3, name='image_dropout')(image_norm)
    # shape: (batch, 256)

    # ──────────────────────────────────────────────────────────────────────────
    # BRANCH 2 — Sequence (Caption) Decoder
    # ──────────────────────────────────────────────────────────────────────────
    # Integer-encoded partial caption, post-padded to max_length.

    seq_input = Input(shape=(max_length,), name='seq_input')
    # shape: (batch, max_length)

    # Embedding — Unit V: learnable lookup table (vocab_size, 256)
    # mask_zero=True: tells LSTM to ignore padding (index 0) positions.
    seq_embed = Embedding(
        input_dim=vocab_size,
        output_dim=256,
        mask_zero=True,
        name='word_embedding'
    )(seq_input)
    # shape: (batch, max_length, 256)

    # Dropout on embeddings — Unit III: prevents over-reliance on specific vectors
    seq_drop = Dropout(0.3, name='seq_dropout')(seq_embed)
    # shape: (batch, max_length, 256)

    # LSTM — Unit V: Long Short-Term Memory
    # Processes word embeddings sequentially; returns the FINAL hidden state h_T
    # which summarises the entire partial caption in a 256-dim vector.
    # units=256 matches the image Dense output for direct addition.
    seq_out = LSTM(256, name='caption_lstm')(seq_drop)
    seq_out = LayerNormalization(name='seq_ln')(seq_out)
    # shape: (batch, 256)

    # ──────────────────────────────────────────────────────────────────────────
    # MERGE — Combine image and sequence branches
    # ──────────────────────────────────────────────────────────────────────────
    # Element-wise addition: forces complementary 256-d representations.
    # Both image_out and seq_out are (batch, 256).
    merged = add([image_out, seq_out], name='merge_add')
    # shape: (batch, 256)

    # Post-merge Dense — learn complex interactions between visual + language context
    decoder_out = Dense(256, activation='relu', name='decoder_dense')(merged)
    # shape: (batch, 256)

    # ──────────────────────────────────────────────────────────────────────────
    # OUTPUT — Next-word prediction
    # ──────────────────────────────────────────────────────────────────────────
    # Softmax over vocab_size words → probability distribution.
    # During inference: argmax (greedy) or beam search over this distribution.
    output = Dense(vocab_size, activation='softmax', name='output_softmax')(decoder_out)
    # shape: (batch, vocab_size)

    # ──────────────────────────────────────────────────────────────────────────
    # ASSEMBLE & COMPILE
    # ──────────────────────────────────────────────────────────────────────────
    model = Model(
        inputs=[image_input, seq_input],
        outputs=output,
        name='image_captioning_model'
    )

    # sparse_categorical_crossentropy:
    #   - target y is a plain integer index (e.g. 42), not a one-hot vector
    #   - loss = -log(y_pred[true_class])   (same as categorical_crossentropy)
    #   - avoids materialising giant one-hot arrays in memory
    model.compile(
        optimizer=Adam(learning_rate=1e-3),   # Unit II: Adam, lr=0.001
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — print_model_summary
# ──────────────────────────────────────────────────────────────────────────────

def print_model_summary(vocab_size, max_length):
    """
    Build the model and print a full layer-by-layer summary.

    Args:
        vocab_size (int): Vocabulary size (from tokenizer).
        max_length (int): Max caption length (from tokenizer fitting).
    """
    model = build_model(vocab_size, max_length)
    print("\n" + "="*60)
    print("  MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    model.summary()
    print("="*60)

    total_params     = model.count_params()
    trainable_params = sum(
        layer.count_params() for layer in model.layers if layer.trainable
    )
    print(f"\n  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print("="*60)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — demonstrate the architecture; reads vocab from saved tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Build the model using the actual saved tokenizer (not hard-coded numbers).

    Reads tokenizer.pkl from models/ to get the true vocab_size and a
    representative max_length. Falls back to safe defaults if not found.
    """
    PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tokenizer.pkl")

    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        VOCAB_SIZE = len(tokenizer.word_index) + 1
        print(f"[model] Loaded tokenizer. vocab_size = {VOCAB_SIZE}")
    else:
        # Safe fallback if preprocessing hasn't been run yet
        VOCAB_SIZE = 3500
        print(f"[model] tokenizer.pkl not found — using default vocab_size={VOCAB_SIZE}")

    MAX_LENGTH = 37   # typical Flickr8k value; train.py uses the real computed value

    print(f"[model] Building model for vocab_size={VOCAB_SIZE}, max_length={MAX_LENGTH}")
    model = print_model_summary(VOCAB_SIZE, MAX_LENGTH)

    print("\n[model] Architecture check:")
    print(f"  Input  1 (image)  : {model.input[0].shape}")   # (None, 4096)
    print(f"  Input  2 (seq)    : {model.input[1].shape}")   # (None, max_length)
    print(f"  Output (softmax)  : {model.output.shape}")     # (None, vocab_size)
    print("\nModel is ready. Next step -> run src/train.py")


if __name__ == "__main__":
    main()
