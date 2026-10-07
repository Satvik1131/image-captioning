"""
src/model.py
=============
STEP 4 of the Image Captioning Pipeline:
    Define the CNN + LSTM image captioning model using Keras Functional API.

Architecture Overview (Merge Model):
    The model has TWO input branches that are merged before the output:

    Branch 1 — Image Feature Extractor:
        Input:   (512,)          — pre-extracted VGG16 feature vector
        Dropout: (512,)          — Unit III: regularisation to prevent overfitting
        Dense:   (256,) relu     — project image features into shared 256-dim space

    Branch 2 — Sequence (Caption) Decoder:
        Input:     (max_length,) — integer-encoded partial caption (padded)
        Embedding: (max_length, 256) — Unit V: maps word indices → dense vectors
        Dropout:   (max_length, 256) — Unit III: regularisation on embeddings
        LSTM:      (256,)        — Unit V: processes word sequence, returns final hidden state

    Merge:
        add([image_out, lstm_out])  — element-wise addition (both are 256-dim)
        Dense(256, relu)            — combined representation
        Dense(vocab_size, softmax)  — probability over entire vocabulary (next word)

    Loss:
        categorical_crossentropy    — Unit II: measures difference between predicted
                                      word probability distribution and true one-hot label

    Optimizer:
        Adam                        — Unit II: adaptive learning rate, combines
                                      momentum + RMSProp for robust convergence

Syllabus Alignment:
    - Unit II  : Adam Optimizer, Categorical Cross-Entropy Loss
    - Unit III : Dropout regularisation (on image branch AND embedding branch)
    - Unit IV  : VGG16 feature vector (512,) is the input to Branch 1
    - Unit V   : Embedding layer (Word Embeddings) + LSTM (Sequence Modelling)
"""

import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,       # defines an input tensor with a fixed shape
    Dense,       # fully connected layer: y = activation(Wx + b)
    Dropout,     # Unit III: randomly zeros fraction of neurons during training
    Embedding,   # Unit V: learnable lookup table, index → dense vector
    LSTM,        # Unit V: Long Short-Term Memory recurrent layer
    add,         # element-wise addition of two tensors (merge operation)
)


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION — build_model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(vocab_size, max_length):
    """
    Build and compile the image captioning model using Keras Functional API.

    Why Functional API (not Sequential)?
        We have TWO separate input branches (image features + caption sequence)
        that need to be merged. Sequential API only supports a single linear
        stack of layers — it cannot represent this forked/merged graph.
        The Functional API lets us define arbitrary DAG (directed acyclic graph)
        computation graphs.

    Args:
        vocab_size  (int): Total number of unique tokens in the vocabulary + 1.
                           This was computed in caption_preprocessing.py.
                           Example: 8769
                           The output Dense layer has vocab_size neurons —
                           one probability per possible next word.

        max_length  (int): Length of the longest caption sequence (in tokens).
                           This was computed in caption_preprocessing.py.
                           Example: 37
                           All input sequences are padded to this length.

    Returns:
        model (tf.keras.Model): Compiled Keras model ready for training.

    Model Input/Output summary:
        Inputs  : [image_input (batch, 512), seq_input (batch, max_length)]
        Output  : (batch, vocab_size)  — softmax probability over vocabulary
    """

    # ──────────────────────────────────────────────────────────────────────────
    # BRANCH 1 — Image Feature Extractor
    # ──────────────────────────────────────────────────────────────────────────
    # The image features come from VGG16's Global Average Pooling output.
    # Shape: (512,) — one 512-dim float vector per image.
    # VGG16 is NOT part of THIS model; we use pre-extracted pkl features.

    # Input layer: tells Keras the shape of the image feature tensor
    # (512,) means a flat vector of 512 floats — no batch dimension here,
    #  Keras automatically adds the batch axis: (None, 512)
    image_input = Input(shape=(512,), name='image_input')
    # Current shape: (batch, 512)

    # Dropout — Unit III: Regularisation
    # During training, randomly sets 50% of the 512 values to 0 each forward pass.
    # This prevents the model from memorising specific image features and
    # forces it to learn more robust, distributed representations.
    # At inference time (predict), Dropout is automatically disabled.
    image_drop = Dropout(0.5, name='image_dropout')(image_input)
    # Current shape: (batch, 512)  — same shape, some values zeroed during training

    # Dense(256, relu) — project from 512-dim VGG16 space → 256-dim shared space
    # relu activation: f(x) = max(0, x) — introduces non-linearity, kills negatives
    # This 256-dim vector is what gets ADDED to the LSTM output
    image_features = Dense(256, activation='relu', name='image_dense')(image_drop)
    # Current shape: (batch, 256)

    # ──────────────────────────────────────────────────────────────────────────
    # BRANCH 2 — Sequence (Caption) Decoder
    # ──────────────────────────────────────────────────────────────────────────
    # The caption input is a PARTIAL caption represented as an integer sequence.
    # Example: "startseq a dog" → [1, 5, 42, 0, 0, ..., 0] (padded to max_length)

    # Input layer: integer sequence of length max_length
    # Each integer is a word index from the tokenizer vocabulary
    seq_input = Input(shape=(max_length,), name='seq_input')
    # Current shape: (batch, max_length)  — e.g. (batch, 37)

    # Embedding layer — Unit V: Word Embeddings
    # This is a learnable lookup table of shape (vocab_size, 256).
    # For each integer index in the sequence, it retrieves the corresponding
    # 256-dimensional embedding vector.
    #
    # Parameters:
    #   vocab_size  : number of rows in the embedding matrix (one per word)
    #   256         : embedding dimension — each word maps to a 256-dim vector
    #   mask_zero=True : tells LSTM to IGNORE padding (index 0) positions
    #                    so they don't contribute to the learned sequence context
    #
    # Why 256-dim? It matches the image Dense output so we can add() them directly.
    #
    # Weight shape: (vocab_size, 256) — e.g. (8769, 256) = ~2.25M trainable params
    seq_embed = Embedding(
        input_dim=vocab_size,   # vocabulary size (number of unique tokens)
        output_dim=256,         # each word → a 256-dim dense vector
        mask_zero=True,         # ignore padding tokens (index 0) in the LSTM
        name='word_embedding'
    )(seq_input)
    # Current shape: (batch, max_length, 256)
    # Each of the max_length positions now has a 256-dim word vector

    # Dropout on embeddings — Unit III
    # Randomly zeros entire word embedding vectors during training.
    # Helps prevent the model from overfitting to specific word-vector combinations.
    seq_drop = Dropout(0.5, name='seq_dropout')(seq_embed)
    # Current shape: (batch, max_length, 256)

    # LSTM layer — Unit V: Long Short-Term Memory
    # Processes the sequence of word embeddings one step at a time.
    # At each time step t, the LSTM cell:
    #   1. Takes the current word embedding x_t and previous hidden state h_{t-1}
    #   2. Applies forget gate, input gate, cell gate, output gate
    #   3. Outputs a new hidden state h_t
    #
    # After processing all max_length tokens, the FINAL hidden state h_T
    # summarises the entire partial caption context into a 256-dim vector.
    #
    # units=256: the LSTM has 256 memory units (matches image Dense output)
    # return_sequences=False (default): only return the FINAL hidden state h_T,
    #   not all intermediate states h_1, h_2, ..., h_T.
    #   This gives us a single 256-dim summary of the sequence.
    #
    # LSTM trainable parameters:
    #   4 gates × (256 input + 256 hidden + 1 bias) × 256 units = ~525K params
    seq_features = LSTM(256, name='caption_lstm')(seq_drop)
    # Current shape: (batch, 256)  — final hidden state only

    # ──────────────────────────────────────────────────────────────────────────
    # MERGE — Combine image and sequence branches
    # ──────────────────────────────────────────────────────────────────────────
    # Both branches now output a (batch, 256) tensor.
    # We merge them with element-wise addition (add()).
    #
    # Why addition (not concatenation)?
    #   - Concatenation would give (batch, 512), doubling parameters downstream.
    #   - Addition forces the model to learn complementary representations:
    #     the image Dense and the LSTM must produce COMPATIBLE 256-dim vectors
    #     that make sense when summed. This is a form of implicit regularisation.
    #   - Addition is the approach used in the original "Merge" captioning model
    #     (Tanti et al., 2017) that this architecture is based on.
    decoder_input = add([image_features, seq_features], name='merge_add')
    # Current shape: (batch, 256)

    # Dense(256, relu) — process the merged representation
    # Adds non-linearity after the addition, allowing the model to learn
    # complex interactions between image and language features.
    decoder_out = Dense(256, activation='relu', name='decoder_dense')(decoder_input)
    # Current shape: (batch, 256)

    # ──────────────────────────────────────────────────────────────────────────
    # OUTPUT — Predict next word
    # ──────────────────────────────────────────────────────────────────────────
    # Final Dense layer with softmax activation.
    # vocab_size neurons, each outputting the probability that the next word
    # is the word at that vocabulary index.
    #
    # softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
    # Output sums to 1.0 — a valid probability distribution over the vocabulary.
    #
    # During training:   we compare this distribution to the one-hot true label
    #                    using Categorical Cross-Entropy loss (Unit II).
    # During inference:  we take argmax (greedy) or sample (beam search) from
    #                    this distribution to pick the next word.
    output = Dense(vocab_size, activation='softmax', name='output_softmax')(decoder_out)
    # Current shape: (batch, vocab_size)  — e.g. (batch, 8769)

    # ──────────────────────────────────────────────────────────────────────────
    # ASSEMBLE & COMPILE
    # ──────────────────────────────────────────────────────────────────────────
    # Create the Model by specifying inputs and outputs
    model = Model(
        inputs=[image_input, seq_input],   # two inputs: image vector + partial caption
        outputs=output,                    # one output: next-word probability distribution
        name='image_captioning_model'
    )

    # Compile the model:
    #   optimizer='adam'                    — Unit II: Adam optimizer
    #       Adam (Adaptive Moment Estimation) maintains per-parameter learning rates.
    #       It combines Momentum (exponential moving avg of gradients) and
    #       RMSProp (exponential moving avg of squared gradients).
    #       Default lr=0.001 works well for this task.
    #
    #   loss='categorical_crossentropy'     — Unit II: Cross-Entropy Loss
    #       For multi-class classification (predict 1 word from vocab_size words).
    #       L = -sum(y_true * log(y_pred))
    #       Where y_true is a one-hot vector (1 at the true word index, 0 elsewhere)
    #       and y_pred is our softmax output distribution.
    #       Penalises low confidence in the correct word heavily.
    #
    #   metrics=['accuracy']               — word-level prediction accuracy
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — print_model_summary
# ──────────────────────────────────────────────────────────────────────────────

def print_model_summary(vocab_size, max_length):
    """
    Build the model and print a full layer-by-layer summary.

    Useful for verifying the architecture before starting training.
    Call this from train.py or run this file directly.

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

    # Count trainable parameters explicitly for clarity
    total_params     = model.count_params()
    trainable_params = sum(
        layer.count_params() for layer in model.layers if layer.trainable
    )
    print(f"\n  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")
    print("="*60)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — demonstrate the architecture with values from caption_preprocessing.py
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Build the model with the known Flickr8k vocabulary stats and print summary.

    Values from caption_preprocessing.py output:
        vocab_size = 8769   (8768 unique tokens + 1 for index-0 padding)
        max_length = 37     (longest caption has 37 tokens)
    """
    # These constants match the output from caption_preprocessing.py exactly.
    # They will be loaded dynamically from the tokenizer in train.py.
    VOCAB_SIZE  = 8769
    MAX_LENGTH  = 37

    print(f"[model] Building model for vocab_size={VOCAB_SIZE}, max_length={MAX_LENGTH}")
    model = print_model_summary(VOCAB_SIZE, MAX_LENGTH)

    print("\n[model] Architecture check:")
    print(f"  Input  1 (image)  : {model.input[0].shape}")  # (None, 512)
    print(f"  Input  2 (seq)    : {model.input[1].shape}")  # (None, 37)
    print(f"  Output (softmax)  : {model.output.shape}")    # (None, 8769)
    print("\nModel is ready. Next step -> run src/train.py")


if __name__ == "__main__":
    main()
