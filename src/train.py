"""
src/train.py
=============
STEP 5 of the Image Captioning Pipeline:
    Load all preprocessed data → Build model → Train with shuffled generator →
    Save best weights via ModelCheckpoint → Plot and save training curves

HOW TRAINING WORKS (Viva Explanation):
────────────────────────────────────────
At each training step, the model receives a BATCH of:
    X1 : image feature vectors,      shape (batch, 4096)  — from VGG16 fc2
    X2 : partial caption sequences,  shape (batch, max_length) — padded integers
    y  : sparse integer next-word,   shape (batch,)  — target word index

The model predicts a probability distribution over vocab_size words via softmax.
Adam optimizer computes gradients of Sparse Categorical Cross-Entropy loss and
updates all trainable weights (Embedding + LSTM + Dense layers).

CALLBACKS:
    EarlyStopping(patience=4)   — stop if val_loss stops improving for 4 epochs
    ReduceLROnPlateau(factor=0.5, patience=2) — halve LR on plateau
    ModelCheckpoint(save_best_only=True)      — always keep best val_loss weights

Syllabus Alignment:
    - Unit II  : Adam Optimizer, Sparse Categorical Cross-Entropy Loss
    - Unit III : Dropout (active during training, disabled at inference)
    - Unit V   : Embedding + LSTM trained end-to-end via backprop through time
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for server/headless runs)
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import build_model
from src.data_preprocessing import (
    load_clean_captions,
    load_features,
    load_tokenizer,
    get_max_length,
    get_train_val_split,
    data_generator,
    calculate_steps,
    BATCH_SIZE,
)

# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

EPOCHS    = 30    # Max epochs; EarlyStopping will typically stop before this.
                  # 30 is enough for Flickr8k to converge with patience=4.

VAL_SPLIT = 0.1   # 10% validation images

# ── File paths ─────────────────────────────────────────────────────────────────
PROCESSED_DIR       = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR          = os.path.join(PROJECT_ROOT, "models")

CLEAN_CAPTIONS_PATH = os.path.join(PROCESSED_DIR, "clean_captions.json")
FEATURES_PATH       = os.path.join(PROCESSED_DIR, "features.pkl")
TOKENIZER_PATH      = os.path.join(MODELS_DIR,    "tokenizer.pkl")

# Best-checkpoint path (monitored by ModelCheckpoint, save_best_only=True)
CHECKPOINT_PATH     = os.path.join(MODELS_DIR, "caption_model.h5")

# Final model path (last epoch — kept for comparison)
FINAL_MODEL_PATH    = os.path.join(MODELS_DIR, "caption_model_final.h5")

# Training curve plot path
CURVES_PATH         = os.path.join(MODELS_DIR, "training_curves.png")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — configure_callbacks
# ──────────────────────────────────────────────────────────────────────────────

def configure_callbacks(checkpoint_path):
    """
    Set up Keras training callbacks.

    Returns a list of three callbacks:

    1. ModelCheckpoint
       ─────────────────
       Monitors val_loss after every epoch.
       Saves the full model to .h5 only when val_loss improved.
       save_best_only=True → models/caption_model.h5 always holds the best weights.

    2. EarlyStopping
       ─────────────────
       Stops training if val_loss does NOT improve for 4 consecutive epochs.
       restore_best_weights=True → reverts to the best checkpoint on stop.
       patience=4 (increased from original 3) gives the model more chances to
       recover from temporary plateaus.

    3. ReduceLROnPlateau  (Unit II: Adaptive Learning Rate)
       ─────────────────
       If val_loss hasn't improved for 2 epochs, multiplies Adam LR by 0.5.
       min_lr=1e-6 prevents the LR from shrinking to zero.

    Args:
        checkpoint_path (str): File path for ModelCheckpoint output.

    Returns:
        list: [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau]
    """
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,    # only save if val_loss improved
        mode='min'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=4,             # increased from 3 → allows more recovery time
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,             # halve the learning rate
        patience=2,
        verbose=1,
        min_lr=1e-6
    )

    return [checkpoint, early_stop, reduce_lr]


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — plot_training_curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history, save_path):
    """
    Save a 2-panel plot of training/validation loss and accuracy.

    Panel 1: Loss curves (lower is better)
    Panel 2: Accuracy curves (higher is better)

    The plots help diagnose:
        - Overfitting (train loss falls, val loss rises)
        - Underfitting (both losses remain high)
        - Good convergence (both losses fall together)

    Args:
        history   : Keras History object from model.fit()
        save_path : Absolute path to save the .png file
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History — Image Captioning Model', fontsize=14)

    epochs_ran = range(1, len(history.history['loss']) + 1)

    # Panel 1: Loss
    axes[0].plot(epochs_ran, history.history['loss'],     'b-o', label='Train Loss')
    axes[0].plot(epochs_ran, history.history['val_loss'], 'r-o', label='Val Loss')
    axes[0].set_title('Loss (Sparse Categorical Cross-Entropy)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Accuracy
    axes[1].plot(epochs_ran, history.history['accuracy'],     'b-o', label='Train Accuracy')
    axes[1].plot(epochs_ran, history.history['val_accuracy'], 'r-o', label='Val Accuracy')
    axes[1].set_title('Word-Level Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Training curves saved to: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — train
# ──────────────────────────────────────────────────────────────────────────────

def train():
    """
    Full training pipeline:

        1. Ensure output directories exist
        2. Load clean captions, VGG16 fc2 features, and tokenizer
        3. Derive vocab_size (dynamic) and max_length
        4. Split into train / validation sets (90/10)
        5. Build the CNN + LSTM model (from model.py)
        6. Print model summary
        7. Set up callbacks (checkpoint, early stop, reduce LR)
        8. Train using the shuffled data generator
        9. Save the final model weights
       10. Plot and save training curves
       11. Print training summary
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("[init] Output directories confirmed.\n")

    # ── Step 1: Load all preprocessed artifacts ───────────────────────────────
    print("=" * 60)
    print("  LOADING DATA ARTIFACTS")
    print("=" * 60)

    captions_dict = load_clean_captions(CLEAN_CAPTIONS_PATH)
    features_dict = load_features(FEATURES_PATH)
    tokenizer     = load_tokenizer(TOKENIZER_PATH)

    # ── Step 2: Compute derived constants (dynamically from tokenizer) ─────────
    # NEVER hard-code vocab_size — always read from the actual fitted tokenizer
    vocab_size = len(tokenizer.word_index) + 1
    max_length = get_max_length(captions_dict)

    print(f"\n[config] vocab_size = {vocab_size}")
    print(f"[config] max_length = {max_length}")
    print(f"[config] epochs     = {EPOCHS}  (EarlyStopping may stop earlier)")
    print(f"[config] batch_size = {BATCH_SIZE}")
    print(f"[config] val_split  = {VAL_SPLIT} ({int(VAL_SPLIT*100)}% validation)")

    # ── Step 3: Train / Validation split ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAIN / VALIDATION SPLIT")
    print("=" * 60)

    train_caps, val_caps, train_feats, val_feats = get_train_val_split(
        captions_dict, features_dict, val_split=VAL_SPLIT
    )

    # ── Step 4: Calculate steps per epoch ─────────────────────────────────────
    print("\n[steps] Calculating training steps...")
    train_steps = calculate_steps(train_caps, train_feats, tokenizer, BATCH_SIZE)

    print("\n[steps] Calculating validation steps...")
    val_steps = calculate_steps(val_caps, val_feats, tokenizer, BATCH_SIZE)

    # ── Step 5: Build model ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BUILDING MODEL")
    print("=" * 60)

    model = build_model(vocab_size, max_length)
    model.summary()
    print(f"\n[model] Total trainable parameters: {model.count_params():,}")

    # ── Step 6: Configure callbacks ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CONFIGURING CALLBACKS")
    print("=" * 60)

    callbacks = configure_callbacks(CHECKPOINT_PATH)
    print(f"  [✓] ModelCheckpoint   → {CHECKPOINT_PATH}")
    print(f"  [✓] EarlyStopping     → patience=4 epochs")
    print(f"  [✓] ReduceLROnPlateau → factor=0.5, patience=2 epochs")

    # ── Step 7: Create data generators ────────────────────────────────────────
    print("\n[generator] Creating shuffled train generator...")
    train_gen = data_generator(
        train_caps, train_feats, tokenizer, max_length, vocab_size, BATCH_SIZE
    )

    print("[generator] Creating shuffled validation generator...")
    val_gen = data_generator(
        val_caps, val_feats, tokenizer, max_length, vocab_size, BATCH_SIZE
    )

    # ── Step 8: TRAIN ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STARTING TRAINING")
    print("=" * 60)
    print(f"  Max epochs       : {EPOCHS}  (EarlyStopping patience=4)")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Train steps/epoch: {train_steps:,}")
    print(f"  Val   steps/epoch: {val_steps:,}")
    print(f"  Best model path  : {CHECKPOINT_PATH}")
    print("=" * 60 + "\n")

    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # ── Step 9: Save final model weights ──────────────────────────────────────
    model.save(FINAL_MODEL_PATH)
    print(f"\n[save] Final model saved to: {FINAL_MODEL_PATH}")

    # ── Step 10: Plot and save training curves ─────────────────────────────────
    plot_training_curves(history, CURVES_PATH)

    # ── Step 11: Training Summary ──────────────────────────────────────────────
    best_epoch    = int(np.argmin(history.history['val_loss'])) + 1
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
    print(f"  Training curves   : {CURVES_PATH}")
    print("=" * 60)
    print("\nNext step -> run src/predict.py")

    return model, history, vocab_size, max_length


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
