"""
src/predict.py
==============
STEP 6 of the Image Captioning Pipeline:
    Load trained caption model + tokenizer → extract VGG16 features from any
    image → generate a caption using greedy decoding → print clean output.

HOW GREEDY DECODING WORKS (Viva Explanation):
──────────────────────────────────────────────
At inference time there is no "next word" label to compare against.
Instead we generate one word at a time in a loop:

    Step 0 : Start with the seed text "startseq"
    Step 1 : Convert current text → padded integer sequence
    Step 2 : Run model.predict([image_feature, sequence])
             → probability distribution over vocab_size words
    Step 3 : Pick argmax — the single most probable next word (greedy)
    Step 4 : Convert that index back to a word via index_to_word
    Step 5 : Append the word to the growing caption
    Step 6 : Stop when "endseq" is generated OR max_length is reached

This is "greedy" because at every step we always commit to the single best
word rather than exploring multiple branches (as beam search would do).
Greedy is simpler and fast; beam search typically produces higher-BLEU captions
at the cost of more compute.

Syllabus Alignment:
    - Unit IV : VGG16 CNN used as a real-time feature extractor
    - Unit V  : LSTM-based sequence generation at inference time
    - Unit II : Softmax → argmax decision for next-word selection
"""

import os
import sys
import json
import pickle

import h5py
import numpy as np
from PIL import Image

# ── Keras / TF imports ───────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Add project root to sys.path so `from src.model import build_model` works
# regardless of the CWD when the script is invoked.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import build_model   # reuse the same architecture definition

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

MODELS_DIR        = os.path.join(PROJECT_ROOT, "models")
CAPTION_MODEL_PATH = os.path.join(MODELS_DIR, "caption_model.h5")
TOKENIZER_PATH     = os.path.join(MODELS_DIR, "tokenizer.pkl")

# VGG16 input dimensions — the network was trained on 224×224 RGB images
IMG_SIZE = (224, 224)

# Candidate image directories (handles Flickr8k dataset folder-name variants)
IMAGE_DIR_OPTIONS = [
    os.path.join(PROJECT_ROOT, "data", "Flickr8k_Dataset", "Flicker8k_Dataset"),
    os.path.join(PROJECT_ROOT, "data", "Flickr8k_Dataset", "Flickr8k_Dataset"),
    os.path.join(PROJECT_ROOT, "data", "images"),
]


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — load_caption_model
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# HELPER — _read_max_length_from_h5
# ──────────────────────────────────────────────────────────────────────────────

def _read_max_length_from_h5(path):
    """
    Read max_length (the padded sequence length) from the H5 model config.

    The H5 file stores the full Keras model JSON config as an attribute.
    We locate the 'seq_input' Input layer and read its batch_input_shape[1].

    This avoids reloading the captions JSON just to compute max_length.

    Returns:
        int | None: The max_length if found, else None.
    """
    try:
        with h5py.File(path, 'r') as f:
            raw = f.attrs.get('model_config', None)
            if raw is None:
                return None
            config = json.loads(raw if isinstance(raw, str) else raw.decode('utf-8'))

        # Walk the layer list looking for the seq_input InputLayer
        layers = config.get('config', {}).get('layers', [])
        for layer in layers:
            lconf = layer.get('config', {})
            # In Keras 2 the name lives directly under 'config'
            if lconf.get('name') == 'seq_input':
                shape = lconf.get('batch_input_shape', [])
                if shape and len(shape) > 1 and shape[1] is not None:
                    return int(shape[1])
            # In some versions it's nested one level deeper
            inner = lconf.get('config', {})
            if inner.get('name') == 'seq_input':
                shape = inner.get('batch_input_shape', [])
                if shape and len(shape) > 1 and shape[1] is not None:
                    return int(shape[1])
    except Exception as exc:
        print(f"[warn] Could not read max_length from H5 config: {exc}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — load_caption_model
# ──────────────────────────────────────────────────────────────────────────────

def load_caption_model(path, vocab_size, max_length):
    """
    Load the trained captioning model by rebuilding the architecture from
    scratch and then loading only the saved weights from the H5 file.

    WHY NOT load_model(path)?
        The H5 format embeds the full Keras JSON config inside the file.
        Keras 2 serialised functional-model node connections differently from
        Keras 3, so load_model() raises errors like:
            'Unknown layer: NotEqual'                  (mask_zero embedding)
            'Only input tensors may be passed as positional arguments'
        These are purely deserialisation artefacts — the WEIGHTS themselves are
        stored correctly in both versions.

    SOLUTION — rebuild + load_weights:
        1. Rebuild the identical architecture using build_model(vocab_size, max_length).
           This is safe because build_model() is the same function used in train.py.
        2. Load ONLY the weight tensors from the H5 file with model.load_weights().
           Keras matches weights by layer name, so the order doesn't matter.
           This completely sidesteps the config deserialisation issue.

    Args:
        path       (str): Absolute path to caption_model.h5.
        vocab_size (int): Total vocabulary size (len(tokenizer.word_index) + 1).
        max_length (int): Padded sequence length used during training.

    Returns:
        tf.keras.Model: Rebuilt model with trained weights loaded, compiled.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Trained model not found at: {path}\n"
            "        Run src/train.py first to produce caption_model.h5"
        )

    print(f"[load] Rebuilding model architecture "
          f"(vocab_size={vocab_size}, max_length={max_length})...")
    model = build_model(vocab_size, max_length)
    # build_model() already compiles with Adam + categorical_crossentropy

    print(f"[load] Loading weights from: {path}")
    # by_name=True: match weight tensors to layers by their name attribute.
    # This is robust against minor architecture changes between save and load.
    model.load_weights(path, by_name=True)

    print(f"[load] Caption model ready. Input shapes: "
          f"{[inp.shape for inp in model.inputs]}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — load_tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(path):
    """
    Load the fitted Keras Tokenizer from pickle.

    The tokenizer provides:
        tokenizer.word_index       — { "startseq": 1, "a": 2, "dog": 3, ... }
        tokenizer.index_word       — { 1: "startseq", 2: "a", 3: "dog", ... }
        tokenizer.texts_to_sequences(["startseq a dog"]) → [[1, 2, 3]]

    Args:
        path (str): Absolute path to tokenizer.pkl.

    Returns:
        tokenizer: Fitted Keras Tokenizer object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Tokenizer not found at: {path}\n"
            "        Run src/caption_preprocessing.py first."
        )
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"[load] Tokenizer loaded. Vocabulary size: {vocab_size}")
    return tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — build_feature_extractor
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_extractor():
    """
    Build the VGG16 feature extractor — identical configuration used in
    src/feature_extraction.py during the offline preprocessing step.

    Configuration:
        weights='imagenet'  — 13 convolutional layers pre-trained on 1.2M images
        include_top=False   — discard the Dense(4096) × 2 + Dense(1000) classifier head
        pooling='avg'       — Global Average Pooling: 7×7×512 → (512,) per image

    Input  shape: (batch, 224, 224, 3)
    Output shape: (batch, 512)

    The model is NOT trained further; weights are frozen via model.trainable=False.
    We re-use it here to extract a feature vector for any arbitrary new image.

    Returns:
        tf.keras.Model: VGG16 feature extractor, output shape (None, 512).
    """
    print("[model] Building VGG16 feature extractor (ImageNet weights)...")
    vgg = VGG16(
        weights='imagenet',   # downloads ~553 MB on first run, cached thereafter
        include_top=False,    # strip the Dense classification head
        pooling='avg'         # Global Average Pooling → (512,)
    )
    vgg.trainable = False     # freeze — we only run forward passes, no training
    print(f"[model] VGG16 ready. Output shape: {vgg.output_shape}")
    return vgg


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — extract_image_feature
# ──────────────────────────────────────────────────────────────────────────────

def extract_image_feature(vgg_model, image_path):
    """
    Run a single image through VGG16 to obtain a (1, 512) feature vector.

    Preprocessing pipeline (must match what was done in feature_extraction.py):
        1. Open with PIL             → RGB image, any size
        2. Resize to (224, 224)      → required VGG16 input size
        3. img_to_array              → (224, 224, 3) numpy float32 array
        4. np.expand_dims            → (1, 224, 224, 3) batch tensor
        5. preprocess_input          → subtract ImageNet per-channel mean
                                       (R: 103.94, G: 116.78, B: 123.68)
                                       + convert RGB → BGR

    Args:
        vgg_model      : VGG16 Keras model (from build_feature_extractor).
        image_path (str): Path to the image file (.jpg / .png / etc.).

    Returns:
        np.ndarray: Feature vector of shape (1, 512).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError       : If the image cannot be decoded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

    # Step 1+2: Open and resize
    img = Image.open(image_path).convert('RGB')  # ensure 3-channel RGB
    img = img.resize(IMG_SIZE)                   # (224, 224)

    # Step 3: PIL Image → numpy array (224, 224, 3)
    img_array = img_to_array(img)

    # Step 4: Add batch axis → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Step 5: VGG16-specific pixel normalisation
    img_array = preprocess_input(img_array)

    # Forward pass through VGG16 → (1, 512)
    feature = vgg_model.predict(img_array, verbose=0)  # verbose=0: no progress bar
    return feature   # shape: (1, 512)


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — generate_caption  (GREEDY DECODING)
# ──────────────────────────────────────────────────────────────────────────────

def generate_caption(caption_model, tokenizer, image_feature, max_length):
    """
    Generate a caption for an image using greedy (argmax) decoding.

    GREEDY DECODING LOOP:
    ─────────────────────
    Iteration 0:
        in_text  = "startseq"
        sequence = [1]  (tokenizer converts "startseq" → index 1)
        padded   = [1, 0, 0, ..., 0]  (length = max_length)
        model predicts probabilities over vocab_size words
        word     = argmax → e.g. "a"

    Iteration 1:
        in_text  = "startseq a"
        sequence = [1, 5]
        ...

    Termination:
        Loop ends when the predicted word is "endseq"
        OR when the sequence length reaches max_length
        (prevents infinite loops if "endseq" is never predicted).

    Why greedy (not beam search)?
        Greedy selects the single most probable word at each step.
        It is O(max_length) forward passes and simple to implement.
        Beam search keeps the top-k sequences at each step, giving
        better BLEU scores but requiring k × more compute.

    Args:
        caption_model    : Trained Keras captioning model (two inputs).
        tokenizer        : Fitted Keras Tokenizer.
        image_feature    : np.ndarray of shape (1, 512) — VGG16 features.
        max_length (int) : Maximum number of tokens to generate.

    Returns:
        str: Generated caption with 'startseq' and 'endseq' tags removed.
             Example: "a dog runs through the grass"
    """
    # Build reverse mapping: integer index → word string
    # tokenizer.index_word was added in Keras 2.x; fall back to inverting word_index
    if hasattr(tokenizer, 'index_word'):
        index_to_word = tokenizer.index_word
    else:
        index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    in_text = 'startseq'   # seed text: the model always starts from here

    for _ in range(max_length):
        # ── Step 1: Convert current partial caption to integer sequence ────────
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # e.g. "startseq a"  →  [1, 5]

        # ── Step 2: Post-pad to max_length ─────────────────────────────────────
        # Must match the padding used during training (padding='post')
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        # shape: (1, max_length)

        # ── Step 3: Model forward pass ─────────────────────────────────────────
        # The captioning model expects two named inputs:
        #   'image_input' : image feature vector (1, 512)
        #   'seq_input'   : padded partial caption (1, max_length)
        #
        # We pass a dict keyed by layer name to be unambiguous.
        y_pred = caption_model.predict(
            {'image_input': image_feature, 'seq_input': sequence},
            verbose=0   # suppress per-call progress bars
        )
        # y_pred shape: (1, vocab_size) — softmax probability distribution

        # ── Step 4: Greedy selection — pick the most probable next word ─────────
        next_word_idx = np.argmax(y_pred[0])
        # y_pred[0] is the first (only) sample in the batch

        # ── Step 5: Convert index → word string ───────────────────────────────
        next_word = index_to_word.get(next_word_idx, None)

        # If the index maps to nothing (shouldn't happen, but defensive):
        if next_word is None:
            break

        # ── Step 6: Append word to growing caption ────────────────────────────
        in_text += ' ' + next_word

        # ── Step 7: Termination check ─────────────────────────────────────────
        # Stop as soon as the model predicts the end-of-sequence token
        if next_word == 'endseq':
            break

    # ── Clean up boundary tokens ───────────────────────────────────────────────
    caption = in_text
    caption = caption.replace('startseq', '').replace('endseq', '')
    caption = caption.strip()
    return caption


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 6 — predict_caption  (high-level helper)
# ──────────────────────────────────────────────────────────────────────────────

def predict_caption(image_path, caption_model, vgg_model, tokenizer, max_length):
    """
    End-to-end caption prediction for a single image.

    Chains:
        1. extract_image_feature  — run image through VGG16 → (1, 512)
        2. generate_caption       — greedy decode → caption string

    Args:
        image_path    (str) : Path to the image file.
        caption_model       : Trained captioning model.
        vgg_model           : VGG16 feature extractor.
        tokenizer           : Fitted Keras Tokenizer.
        max_length    (int) : Max tokens to generate.

    Returns:
        str: Generated caption string.
    """
    feature = extract_image_feature(vgg_model, image_path)
    caption = generate_caption(caption_model, tokenizer, feature, max_length)
    return caption


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 7 — resolve_image_dir  (reused from feature_extraction.py logic)
# ──────────────────────────────────────────────────────────────────────────────

def resolve_image_dir(candidates):
    """
    Find the first existing image directory from a list of candidates.

    Handles the Flickr8k dataset folder-name inconsistency across download
    sources (Kaggle vs official zip — "Flicker" vs "Flickr" typo).

    Returns:
        str: First existing directory path.

    Raises:
        FileNotFoundError: If none of the candidates exist.
    """
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Could not find the Flickr8k image directory.\n"
        "Checked:\n" + "\n".join(f"  {p}" for p in candidates) +
        "\nPlease unzip Flickr8k_Dataset.zip into data/Flickr8k_Dataset/."
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — demo: load models, pick a test image, generate and print caption
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Demonstration: load all artefacts, select an arbitrary test image from the
    Flickr8k dataset directory, and print the generated caption.

    You can also supply a custom image path via the IMAGE_PATH variable below.
    """

    print("=" * 60)
    print("  IMAGE CAPTIONING — INFERENCE DEMO")
    print("=" * 60)

    # ── Step 1: Load tokenizer FIRST — provides vocab_size for model rebuild ───
    tokenizer  = load_tokenizer(TOKENIZER_PATH)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"\n[config] vocab_size = {vocab_size}")

    # ── Step 2: Derive max_length ──────────────────────────────────────────────
    # Primary  : read directly from the JSON config stored inside the H5 file.
    # Fallback : load clean_captions.json and compute the longest caption length.
    # Default  : 37 (known Flickr8k value — used only if both sources are absent).
    max_length = _read_max_length_from_h5(CAPTION_MODEL_PATH)

    if max_length is None:
        print("[config] Could not read max_length from H5 — "
              "computing from captions JSON...")
        captions_path = os.path.join(
            PROJECT_ROOT, "data", "processed", "clean_captions.json"
        )
        if os.path.exists(captions_path):
            with open(captions_path, "r", encoding="utf-8") as f:
                captions_dict = json.load(f)
            lengths = [
                len(cap.split())
                for caps in captions_dict.values()
                for cap in caps
            ]
            max_length = max(lengths)
        else:
            max_length = 37   # Flickr8k hard-coded default
            print(f"[config] clean_captions.json not found — "
                  f"using default max_length={max_length}")

    print(f"[config] max_length = {max_length}")

    # ── Step 3: Load caption model (rebuild architecture + weights from H5) ────
    print()
    caption_model = load_caption_model(CAPTION_MODEL_PATH, vocab_size, max_length)

    # ── Step 4: Build VGG16 feature extractor ─────────────────────────────────
    print()
    vgg_model = build_feature_extractor()

    # ── Step 5: Choose a test image ───────────────────────────────────────────
    # Option A: hard-code a custom image path
    IMAGE_PATH = None   # e.g. r"C:\path\to\your\image.jpg"

    # Option B (default): pick the first .jpg found in the Flickr8k folder
    if IMAGE_PATH is None or not os.path.exists(str(IMAGE_PATH)):
        try:
            image_dir = resolve_image_dir(IMAGE_DIR_OPTIONS)
            all_jpgs = sorted([
                f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')
            ])
            if not all_jpgs:
                raise RuntimeError("No .jpg files found in the image directory.")
            IMAGE_PATH = os.path.join(image_dir, all_jpgs[0])
            print(f"\n[demo] No custom image path set — using first dataset image:")
            print(f"       {IMAGE_PATH}")
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}")
            print("        Set IMAGE_PATH manually in main() and re-run.")
            return

    # ── Step 6: Generate caption ───────────────────────────────────────────────
    print(f"\n[predict] Extracting VGG16 features from image...")
    feature = extract_image_feature(vgg_model, IMAGE_PATH)
    print(f"[predict] Feature vector shape : {feature.shape}")   # (1, 512)

    print(f"[predict] Running greedy caption generation (max_length={max_length})...")
    caption = generate_caption(caption_model, tokenizer, feature, max_length)

    # ── Step 7: Print results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"  Image   : {os.path.basename(IMAGE_PATH)}")
    print(f"  Caption : {caption}")
    print("=" * 60)

    return caption


if __name__ == "__main__":
    main()
