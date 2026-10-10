"""
src/train.py
=============
STEP 5 of the Image Captioning Pipeline:
    Load all preprocessed data → Build model → Train with generator →
    Save best weights via ModelCheckpoint → Save final model

HOW TRAINING WORKS (Viva Explanation):
────────────────────────────────────────
At each training step, the model receives a BATCH of:
    X1 : image feature vectors,  shape (batch, 512)   — from VGG16
    X2 : partial caption sequences, shape (batch, 37)  — padded integer sequences
    y  : one-hot next-word targets,  shape (batch, 8769) — what word comes next

The model predicts a probability distribution over 8769 words via softmax.
Adam optimizer computes gradients of Categorical Cross-Entropy loss and
updates all 5.2M trainable weights (Embedding + LSTM + Dense layers).

After every epoch the ModelCheckpoint callback compares validation loss.
If it improved, the weights are saved to models/caption_model.h5.
This ensures we always keep the BEST generalising model, not the last one.

Syllabus Alignment:
    - Unit II  : Adam Optimizer, Categorical Cross-Entropy Loss
    - Unit III : Dropout (active during training, disabled at inference)
    - Unit V   : Embedding + LSTM trained end-to-end via backprop through time
"""

import os
import sys
import json
import pickle

import numpy as np

# ── Keras callbacks and utilities ──────────────────────────────────────────────
from tensorflow.keras.callbacks import (
    ModelCheckpoint,   # saves model weights whenever validation loss improves
    EarlyStopping,     # stops training if val_loss stops improving (prevents overfitting)
    ReduceLROnPlateau, # halves learning rate when val_loss plateaus (Unit II: adaptive LR)
)

# Add project root to sys.path so we can import from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our own modules
from src.model import build_model                            # CNN + LSTM architecture
from src.data_preprocessing import (
    load_clean_captions,       # loads data/processed/clean_captions.json
    load_features,             # loads data/processed/features.pkl
    load_tokenizer,            # loads models/tokenizer.pkl
    get_max_length,            # computes max caption length from captions dict
    get_train_val_split,       # splits images 90% train / 10% val
    data_generator,            # memory-efficient batch generator
    calculate_steps,           # computes steps_per_epoch for model.fit()
)

# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS  (all in one place — easy to change and explain in viva)
# ──────────────────────────────────────────────────────────────────────────────

EPOCHS      = 10    # Number of full passes over the training data
                    # 10 epochs is a reasonable starting point for Flickr8k on CPU.
                    # Each epoch ≈ 13,403 gradient updates with batch_size=32.

BATCH_SIZE  = 32    # Samples per gradient update.
                    # Smaller batch → noisier gradients but faster updates.
                    # 32 is the standard default (fits in CPU RAM easily).

VAL_SPLIT   = 0.1  # 10% of images held out for validation (not seen during training)

# ── File paths ─────────────────────────────────────────────────────────────────
PROCESSED_DIR       = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR          = os.path.join(PROJECT_ROOT, "models")

CLEAN_CAPTIONS_PATH = os.path.join(PROCESSED_DIR, "clean_captions.json")
FEATURES_PATH       = os.path.join(PROCESSED_DIR, "features.pkl")
TOKENIZER_PATH      = os.path.join(MODELS_DIR,    "tokenizer.pkl")

# ModelCheckpoint saves the BEST weights here (lowest val_loss across all epochs)
CHECKPOINT_PATH     = os.path.join(MODELS_DIR, "caption_model.h5")

# Final model saved here after all epochs complete
FINAL_MODEL_PATH    = os.path.join(MODELS_DIR, "caption_model_final.h5")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — configure_callbacks
# ──────────────────────────────────────────────────────────────────────────────

def configure_callbacks(checkpoint_path):
    """
    Set up Keras training callbacks.

    Callbacks are hooks that Keras calls at the end of each epoch
    (or each batch) to perform automatic actions during training.

    Returns a list of three callbacks:

    1. ModelCheckpoint
       ─────────────────
       Monitors validation loss after every epoch.
       If val_loss improved (decreased), saves the full model weights to .h5.
       save_best_only=True — only overwrites the file if the model got better.
       This ensures models/caption_model.h5 always holds the best generalising
       weights, even if training diverges later.

    2. EarlyStopping
       ─────────────────
       If validation loss does NOT improve for 'patience' consecutive epochs,
       training stops automatically.
       restore_best_weights=True — rolls back to the best weights found so far.
       This prevents wasting time training a model that has already peaked.

    3. ReduceLROnPlateau  (Unit II: Adaptive Learning Rate)
       ─────────────────
       If val_loss hasn't improved for 'patience' epochs, multiplies the
       current Adam learning rate by 'factor' (halves it).
       This gives the optimiser a better chance to find a local minimum
       when it's oscillating around a plateau.
       min_lr = lower bound — we never let LR shrink below this.

    Args:
        checkpoint_path (str): File path for ModelCheckpoint output.

    Returns:
        list: [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau]
    """
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',    # watch validation loss (not training loss)
        verbose=1,             # print a message each time the model is saved
        save_best_only=True,   # only save if val_loss improved
        mode='min'             # 'min' because lower loss = better
    )

    early_stop = EarlyStopping(
        monitor='val_loss',    # watch validation loss
        patience=3,            # stop if no improvement for 3 consecutive epochs
        verbose=1,
        restore_best_weights=True  # revert to best weights when stopping
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,            # new_lr = current_lr * 0.5 (halve the learning rate)
        patience=2,            # trigger after 2 epochs of no improvement
        verbose=1,
        min_lr=1e-6            # never go below this learning rate
    )

    return [checkpoint, early_stop, reduce_lr]


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — train
# ──────────────────────────────────────────────────────────────────────────────

def train():
    """
    Full training pipeline:

        1. Ensure output directories exist
        2. Load clean captions, VGG16 features, and tokenizer
        3. Derive vocab_size and max_length
        4. Split into train / validation sets (90/10)
        5. Build the CNN + LSTM model (from model.py)
        6. Print model summary
        7. Set up callbacks (checkpoint, early stop, reduce LR)
        8. Train using the data generator (memory-efficient)
        9. Save the final model weights
       10. Print training summary

    All data is loaded from the paths defined in CONSTANTS above.
    Model weights are saved to models/caption_model.h5 (best) and
    models/caption_model_final.h5 (last epoch).
    """

    # ── Step 0: Ensure models/ directory exists ───────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("[init] Output directories confirmed.\n")

    # ── Step 1: Load all preprocessed artifacts ───────────────────────────────
    print("=" * 60)
    print("  LOADING DATA ARTIFACTS")
    print("=" * 60)

    captions_dict = load_clean_captions(CLEAN_CAPTIONS_PATH)
    # captions_dict: { "image_id.jpg": ["startseq ... endseq", ...] }
    # 8092 images, 5 captions each

    features_dict = load_features(FEATURES_PATH)
    # features_dict: { "image_id.jpg": np.array(512,) }
    # 8091 pre-extracted VGG16 feature vectors

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    # tokenizer.word_index: { "startseq": 1, "a": 2, ..., "endseq": N }

    # ── Step 2: Compute derived constants ─────────────────────────────────────
    # vocab_size: number of unique tokens + 1 (Keras reserves index 0 for padding)
    vocab_size = len(tokenizer.word_index) + 1

    # max_length: length of the longest caption — all sequences padded to this
    max_length = get_max_length(captions_dict)

    print(f"\n[config] vocab_size = {vocab_size}")
    print(f"[config] max_length = {max_length}")
    print(f"[config] epochs     = {EPOCHS}")
    print(f"[config] batch_size = {BATCH_SIZE}")
    print(f"[config] val_split  = {VAL_SPLIT} ({int(VAL_SPLIT*100)}% validation)")

    # ── Step 3: Train / Validation split ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAIN / VALIDATION SPLIT")
    print("=" * 60)

    train_caps, val_caps, train_feats, val_feats = get_train_val_split(
        captions_dict, features_dict, val_split=VAL_SPLIT
    )
    # Approx: 7282 train images, 809 val images

    # ── Step 4: Calculate steps per epoch ─────────────────────────────────────
    # steps_per_epoch = ceil(total_word_step_samples / batch_size)
    # Tells Keras how many generator.yield() calls = 1 complete epoch
    print("\n[steps] Calculating training steps...")
    train_steps = calculate_steps(train_caps, train_feats, tokenizer, BATCH_SIZE)

    print("\n[steps] Calculating validation steps...")
    val_steps = calculate_steps(val_caps, val_feats, tokenizer, BATCH_SIZE)

    # ── Step 5: Build model ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BUILDING MODEL")
    print("=" * 60)

    model = build_model(vocab_size, max_length)
    # Model is already compiled with:
    #   optimizer = Adam(lr=0.001)         — Unit II
    #   loss      = categorical_crossentropy — Unit II
    #   metrics   = ['accuracy']

    model.summary()
    print(f"\n[model] Total trainable parameters: {model.count_params():,}")

    # ── Step 6: Configure callbacks ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CONFIGURING CALLBACKS")
    print("=" * 60)

    callbacks = configure_callbacks(CHECKPOINT_PATH)
    print(f"  [✓] ModelCheckpoint  → {CHECKPOINT_PATH}")
    print(f"  [✓] EarlyStopping    → patience=3 epochs")
    print(f"  [✓] ReduceLROnPlateau → factor=0.5, patience=2 epochs")

    # ── Step 7: Create data generators ────────────────────────────────────────
    # Generators yield batches lazily — only batch_size samples in RAM at once.
    # The training generator loops forever (while True); Keras stops it after
    # train_steps yields per epoch.
    print("\n[generator] Creating train generator...")
    train_gen = data_generator(
        train_caps, train_feats, tokenizer, max_length, vocab_size, BATCH_SIZE
    )

    print("[generator] Creating validation generator...")
    val_gen = data_generator(
        val_caps, val_feats, tokenizer, max_length, vocab_size, BATCH_SIZE
    )

    # ── Step 8: TRAIN ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STARTING TRAINING")
    print("=" * 60)
    print(f"  Epochs          : {EPOCHS}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Train steps/epoch: {train_steps:,}")
    print(f"  Val   steps/epoch: {val_steps:,}")
    print(f"  Best model path : {CHECKPOINT_PATH}")
    print("=" * 60 + "\n")

    # model.fit() with a generator:
    #   - 'generator'         : our data_generator() (yields batches indefinitely)
    #   - 'steps_per_epoch'   : how many batches = 1 epoch
    #   - 'validation_data'   : separate val generator
    #   - 'validation_steps'  : how many val batches to evaluate at epoch end
    #   - 'epochs'            : total number of full passes over the data
    #   - 'callbacks'         : checkpoint + early stop + reduce LR
    #   - 'verbose=1'         : print a progress bar per epoch

    history = model.fit(
        train_gen,                         # training data generator
        steps_per_epoch=train_steps,       # yield calls per epoch
        validation_data=val_gen,           # validation data generator
        validation_steps=val_steps,        # val yield calls per epoch
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1                          # show epoch progress bar
    )

    # ── Step 9: Save final model weights ──────────────────────────────────────
    # The ModelCheckpoint already saved the BEST weights mid-training.
    # Here we additionally save the FINAL state (last epoch) for comparison.
    model.save(FINAL_MODEL_PATH)
    print(f"\n[save] Final model saved to: {FINAL_MODEL_PATH}")

    # ── Step 10: Training Summary ──────────────────────────────────────────────
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    best_val_loss = min(history.history['val_loss'])
    best_val_acc  = history.history['val_accuracy'][best_epoch - 1]
    final_train_loss = history.history['loss'][-1]
    final_train_acc  = history.history['accuracy'][-1]

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Epochs run        : {len(history.history['loss'])} / {EPOCHS}")
    print(f"  Best epoch        : {best_epoch}")
    print(f"  Best val_loss     : {best_val_loss:.4f}")
    print(f"  Best val_accuracy : {best_val_acc:.4f}")
    print(f"  Final train_loss  : {final_train_loss:.4f}")
    print(f"  Final train_acc   : {final_train_acc:.4f}")
    print(f"  Best model saved  : {CHECKPOINT_PATH}")
    print(f"  Final model saved : {FINAL_MODEL_PATH}")
    print("=" * 60)
    print("\nNext step -> run src/predict.py")

    return model, history, vocab_size, max_length


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
