"""
src/predict.py
==============
STEP 6 of the Image Captioning Pipeline:
    Load trained caption model + tokenizer → extract VGG16 fc2 features from
    any image → generate a caption using beam search (default) or greedy
    decoding → print clean output.

GREEDY vs BEAM SEARCH:
──────────────────────
Greedy: at each step pick the single most probable next word. Fast but prone
to mode collapse — the same "safe" high-probability word chain is always chosen
regardless of the image.

Beam Search (width=k): maintain k partial sequences (beams) at each step.
Expand each beam by all vocab words, score them by cumulative log-probability,
keep only the top-k. After max_length steps, return the highest-scoring complete
sequence. Beam search explores more of the probability space and produces more
image-specific, higher-quality captions at the cost of k× more forward passes.

REPETITION GUARD:
─────────────────
If the model enters a loop and repeats the same word 3+ times consecutively,
we stop early and return the caption so far. This prevents infinite (or very
long) degenerate captions like "a dog dog dog dog dog…".

L2-NORMALISATION AT INFERENCE:
────────────────────────────────
The VGG16 fc2 features saved during training were L2-normalised.
The same normalisation MUST be applied at inference to avoid a distribution
shift that would degrade caption quality.

Syllabus Alignment:
    - Unit IV : VGG16 fc2 CNN used as a real-time feature extractor
    - Unit V  : LSTM-based sequence generation at inference time
    - Unit II : Softmax → argmax (greedy) / beam search for next-word selection
"""

import os
import sys
import json
import pickle
import heapq

import h5py
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import build_model

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

MODELS_DIR         = os.path.join(PROJECT_ROOT, "models")
CAPTION_MODEL_PATH = os.path.join(MODELS_DIR, "caption_model.h5")
TOKENIZER_PATH     = os.path.join(MODELS_DIR, "tokenizer.pkl")

IMG_SIZE = (224, 224)   # VGG16 required input size

IMAGE_DIR_OPTIONS = [
    os.path.join(PROJECT_ROOT, "data", "Flickr8k_Dataset", "Flicker8k_Dataset"),
    os.path.join(PROJECT_ROOT, "data", "Flickr8k_Dataset", "Flickr8k_Dataset"),
    os.path.join(PROJECT_ROOT, "data", "images"),
]

DEFAULT_BEAM_WIDTH = 3          # default beam width for beam_search_caption
MAX_REPEAT_COUNT   = 3          # stop if same word repeated this many times in a row


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — _read_max_length_from_h5
# ──────────────────────────────────────────────────────────────────────────────

def _read_max_length_from_h5(path):
    """
    Read max_length from the H5 model config's seq_input layer.

    Returns:
        int | None: The max_length if found, else None.
    """
    try:
        with h5py.File(path, 'r') as f:
            raw = f.attrs.get('model_config', None)
            if raw is None:
                return None
            config = json.loads(raw if isinstance(raw, str) else raw.decode('utf-8'))

        layers = config.get('config', {}).get('layers', [])
        for layer in layers:
            lconf = layer.get('config', {})
            if lconf.get('name') == 'seq_input':
                shape = lconf.get('batch_input_shape', [])
                if shape and len(shape) > 1 and shape[1] is not None:
                    return int(shape[1])
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
    Load the trained captioning model by rebuilding architecture + loading weights.

    WHY NOT load_model(path)?
        Keras 2 → 3 serialisation differences cause errors like:
            'Unknown layer: NotEqual' / 'Only input tensors may be passed...'
        These are deserialisation artefacts — the weights themselves are fine.

    SOLUTION: rebuild architecture with build_model() (same function used in
    train.py), then load ONLY the weights. Keras matches by layer name.

    Args:
        path       (str): Path to caption_model.h5.
        vocab_size (int): len(tokenizer.word_index) + 1.
        max_length (int): Padded sequence length used during training.

    Returns:
        tf.keras.Model: Rebuilt model with trained weights loaded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Trained model not found at: {path}\n"
            "        Run src/train.py first to produce caption_model.h5"
        )

    print(f"[load] Rebuilding model architecture "
          f"(vocab_size={vocab_size}, max_length={max_length})...")
    model = build_model(vocab_size, max_length)

    print(f"[load] Loading weights from: {path}")
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
    Build the VGG16 fc2 feature extractor — IDENTICAL to feature_extraction.py.

    Configuration:
        weights='imagenet'  — full pretrained VGG16 including Dense layers
        include_top=True    — keep fc1 and fc2 Dense layers
        Sub-model output    — fc2 layer (4096-d)

    Input  shape: (batch, 224, 224, 3)
    Output shape: (batch, 4096)

    IMPORTANT: must match the extractor used during training exactly.
    Using the wrong extractor (e.g., pooling='avg' → 512-d) would produce
    features from a completely different distribution → garbage captions.

    Returns:
        tf.keras.Model: VGG16 fc2 extractor, output shape (None, 4096).
    """
    print("[model] Building VGG16 fc2 feature extractor (ImageNet weights)...")
    base = VGG16(
        weights='imagenet',
        include_top=True,
        input_shape=(224, 224, 3)
    )
    extractor = Model(
        inputs=base.input,
        outputs=base.get_layer('fc2').output,
        name='vgg16_fc2_extractor'
    )
    extractor.trainable = False
    print(f"[model] VGG16 fc2 extractor ready. Output shape: {extractor.output_shape}")
    return extractor


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — extract_image_feature
# ──────────────────────────────────────────────────────────────────────────────

def extract_image_feature(vgg_model, image_path):
    """
    Run a single image through VGG16 fc2 and L2-normalise the result.

    Preprocessing pipeline (must match feature_extraction.py exactly):
        1. Open with PIL             → RGB image, any size
        2. Resize to (224, 224)      → VGG16 required input size
        3. img_to_array              → (224, 224, 3) numpy float32
        4. np.expand_dims            → (1, 224, 224, 3) batch tensor
        5. preprocess_input          → subtract ImageNet per-channel mean + BGR
        6. VGG16 fc2 forward pass    → (1, 4096) raw feature
        7. L2-normalise              → (1, 4096) unit-length vector

    Args:
        vgg_model      : VGG16 fc2 extractor (from build_feature_extractor).
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: L2-normalised feature vector of shape (1, 4096).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    feature = vgg_model.predict(img_array, verbose=0)   # (1, 4096)

    # L2-normalise — MUST match training preprocessing
    norm    = np.linalg.norm(feature) + 1e-8
    feature = feature / norm

    return feature   # shape: (1, 4096)


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — generate_caption  (GREEDY DECODING)
# ──────────────────────────────────────────────────────────────────────────────

def generate_caption(caption_model, tokenizer, image_feature, max_length):
    """
    Generate a caption using greedy (argmax) decoding.

    At each step, pick the single most probable next word.
    Includes a REPETITION GUARD: if the same word repeats MAX_REPEAT_COUNT
    times in a row, stop early to avoid degenerate outputs.

    Args:
        caption_model    : Trained Keras captioning model.
        tokenizer        : Fitted Keras Tokenizer.
        image_feature    : np.ndarray of shape (1, 4096) — L2-normalised.
        max_length (int) : Maximum tokens to generate.

    Returns:
        str: Generated caption with startseq/endseq removed.
    """
    if hasattr(tokenizer, 'index_word'):
        index_to_word = tokenizer.index_word
    else:
        index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    in_text    = 'startseq'
    last_word  = None
    repeat_cnt = 0

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

        y_pred = caption_model.predict(
            {'image_input': image_feature, 'seq_input': sequence},
            verbose=0
        )
        next_word_idx = np.argmax(y_pred[0])
        next_word     = index_to_word.get(next_word_idx, None)

        if next_word is None:
            break

        # Repetition guard
        if next_word == last_word:
            repeat_cnt += 1
            if repeat_cnt >= MAX_REPEAT_COUNT:
                break
        else:
            repeat_cnt = 0
        last_word = next_word

        in_text += ' ' + next_word

        if next_word == 'endseq':
            break

    caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return caption


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 6 — beam_search_caption  (BEAM SEARCH DECODING — DEFAULT)
# ──────────────────────────────────────────────────────────────────────────────

def beam_search_caption(caption_model, tokenizer, image_feature, max_length,
                        beam_width=DEFAULT_BEAM_WIDTH):
    """
    Generate a caption using beam search decoding.

    At each step, maintain the top beam_width partial sequences ranked by
    cumulative log-probability. Expand each beam by all vocab words, score,
    and keep only the top beam_width candidates.

    When beam_width=1 this is identical to greedy decoding.

    REPETITION GUARD: if any beam's last word repeats MAX_REPEAT_COUNT times
    in a row, that beam is terminated (treated as complete).

    Args:
        caption_model    : Trained Keras captioning model.
        tokenizer        : Fitted Keras Tokenizer.
        image_feature    : np.ndarray of shape (1, 4096) — L2-normalised.
        max_length (int) : Maximum tokens to generate.
        beam_width (int) : Number of beams to maintain. Default 3.

    Returns:
        str: Best caption with startseq/endseq removed.
    """
    if hasattr(tokenizer, 'index_word'):
        index_to_word = tokenizer.index_word
    else:
        index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    # Each beam: (cumulative_log_prob, token_sequence_as_list, last_word, repeat_cnt)
    # We negate log_prob so heapq (min-heap) gives us the best (highest log_prob) beam
    beams    = [(0.0, ['startseq'], None, 0)]
    complete = []

    for _ in range(max_length):
        new_beams = []

        for log_prob, tokens, last_word, repeat_cnt in beams:
            # If already ended, carry it to complete
            if tokens[-1] == 'endseq':
                complete.append((log_prob, tokens))
                continue

            # Build current sequence
            in_text  = ' '.join(tokens)
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

            y_pred = caption_model.predict(
                {'image_input': image_feature, 'seq_input': sequence},
                verbose=0
            )[0]   # (vocab_size,)

            # Take top beam_width candidates to limit expansion
            top_indices = np.argsort(y_pred)[-beam_width:]

            for idx in top_indices:
                word = index_to_word.get(idx, None)
                if word is None:
                    continue

                word_log_prob  = np.log(y_pred[idx] + 1e-10)
                new_log_prob   = log_prob + word_log_prob
                new_tokens     = tokens + [word]

                # Repetition guard per beam
                if word == last_word:
                    new_repeat = repeat_cnt + 1
                    if new_repeat >= MAX_REPEAT_COUNT:
                        complete.append((new_log_prob, new_tokens))
                        continue
                else:
                    new_repeat = 0

                if word == 'endseq':
                    complete.append((new_log_prob, new_tokens))
                else:
                    new_beams.append((new_log_prob, new_tokens, word, new_repeat))

        if not new_beams:
            break

        # Keep only top beam_width beams (highest log_prob → least negative)
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    # Collect any still-running beams as complete
    for log_prob, tokens, _, _ in beams:
        complete.append((log_prob, tokens))

    if not complete:
        return ""

    # Normalise by length (length penalty) and pick the best
    best_log_prob, best_tokens = max(
        complete,
        key=lambda x: x[0] / max(len(x[1]), 1)
    )

    caption = ' '.join(best_tokens)
    caption = caption.replace('startseq', '').replace('endseq', '').strip()
    return caption


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 7 — predict_caption  (high-level helper, beam search by default)
# ──────────────────────────────────────────────────────────────────────────────

def predict_caption(image_path, caption_model, vgg_model, tokenizer, max_length,
                    beam_width=DEFAULT_BEAM_WIDTH):
    """
    End-to-end caption prediction for a single image.

    Chains:
        1. extract_image_feature  — VGG16 fc2 + L2-normalise → (1, 4096)
        2. beam_search_caption    — beam decode → caption string
           (set beam_width=1 for greedy)

    Args:
        image_path    (str) : Path to the image file.
        caption_model       : Trained captioning model.
        vgg_model           : VGG16 fc2 extractor.
        tokenizer           : Fitted Keras Tokenizer.
        max_length    (int) : Max tokens to generate.
        beam_width    (int) : Beam width (1 = greedy).

    Returns:
        str: Generated caption string.
    """
    feature = extract_image_feature(vgg_model, image_path)
    if beam_width == 1:
        return generate_caption(caption_model, tokenizer, feature, max_length)
    return beam_search_caption(caption_model, tokenizer, feature, max_length, beam_width)


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 8 — resolve_image_dir
# ──────────────────────────────────────────────────────────────────────────────

def resolve_image_dir(candidates):
    """
    Find the first existing image directory from a list of candidates.

    Returns:
        str: First existing directory path.
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
# VERIFICATION — image_dependence_test
# ──────────────────────────────────────────────────────────────────────────────

def image_dependence_test(caption_model, vgg_model, tokenizer, max_length,
                           image_dir, n=10, beam_width=DEFAULT_BEAM_WIDTH):
    """
    Pick n validation images and generate captions. Print them all.

    If all captions are identical → the model still ignores the image.
    Also prints the top-5 predicted words for two different images given
    the same 'startseq' input — their distributions should differ.

    Args:
        caption_model : Trained captioning model.
        vgg_model     : VGG16 fc2 extractor.
        tokenizer     : Fitted Keras Tokenizer.
        max_length    : Sequence length.
        image_dir     : Path to Flickr8k image directory.
        n (int)       : Number of test images.
        beam_width    : Beam width for caption generation.
    """
    if hasattr(tokenizer, 'index_word'):
        index_to_word = tokenizer.index_word
    else:
        index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    all_jpgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    # Sample evenly across the dataset
    step     = max(1, len(all_jpgs) // n)
    selected = [all_jpgs[i * step] for i in range(n)]

    print("\n" + "="*60)
    print(f"  IMAGE DEPENDENCE TEST  (beam_width={beam_width})")
    print("="*60)

    captions = []
    features_list = []
    for fname in selected:
        fpath   = os.path.join(image_dir, fname)
        feature = extract_image_feature(vgg_model, fpath)
        caption = beam_search_caption(caption_model, tokenizer, feature,
                                      max_length, beam_width)
        captions.append(caption)
        features_list.append(feature)
        print(f"  {fname[:40]:40s} | {caption}")

    unique = len(set(captions))
    print(f"\n  Unique captions: {unique} / {n}")
    if unique == 1:
        print("  [FAIL] All captions are identical — model still ignores the image!")
    else:
        print("  [PASS] Captions vary across images ✓")

    # Top-5 word comparison for two images given the same startseq
    print("\n  Top-5 next words after 'startseq' for image A vs image B:")
    seed_seq = tokenizer.texts_to_sequences(['startseq'])[0]
    seed_pad = pad_sequences([seed_seq], maxlen=max_length, padding='post')

    for label, feat in [('Image A', features_list[0]), ('Image B', features_list[1])]:
        y = caption_model.predict(
            {'image_input': feat, 'seq_input': seed_pad}, verbose=0
        )[0]
        top5_idx = np.argsort(y)[-5:][::-1]
        top5     = [(index_to_word.get(i, '?'), float(y[i])) for i in top5_idx]
        print(f"  {label}: {top5}")

    print("="*60)
    return captions


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Load all artefacts, run the image-dependence test on 10 images,
    and print generated captions using beam search.
    """
    print("=" * 60)
    print("  IMAGE CAPTIONING — INFERENCE DEMO")
    print("=" * 60)

    # ── Step 1: Load tokenizer ─────────────────────────────────────────────────
    tokenizer  = load_tokenizer(TOKENIZER_PATH)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"\n[config] vocab_size = {vocab_size}")

    # ── Step 2: Derive max_length ──────────────────────────────────────────────
    max_length = _read_max_length_from_h5(CAPTION_MODEL_PATH)

    if max_length is None:
        print("[config] Could not read max_length from H5 — computing from captions JSON...")
        captions_path = os.path.join(
            PROJECT_ROOT, "data", "processed", "clean_captions.json"
        )
        if os.path.exists(captions_path):
            with open(captions_path, "r", encoding="utf-8") as f:
                captions_dict = json.load(f)
            lengths    = [len(cap.split()) for caps in captions_dict.values() for cap in caps]
            max_length = max(lengths)
        else:
            max_length = 37
            print(f"[config] clean_captions.json not found — using default max_length={max_length}")

    print(f"[config] max_length = {max_length}")

    # ── Step 3: Load caption model ─────────────────────────────────────────────
    print()
    caption_model = load_caption_model(CAPTION_MODEL_PATH, vocab_size, max_length)

    # ── Step 4: Build VGG16 fc2 extractor ─────────────────────────────────────
    print()
    vgg_model = build_feature_extractor()

    # ── Step 5: Choose a test image ───────────────────────────────────────────
    IMAGE_PATH = None   # set a custom path here, or leave None to use dataset

    if IMAGE_PATH is None or not os.path.exists(str(IMAGE_PATH)):
        try:
            image_dir = resolve_image_dir(IMAGE_DIR_OPTIONS)
            all_jpgs  = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
            if not all_jpgs:
                raise RuntimeError("No .jpg files found in the image directory.")
            IMAGE_PATH = os.path.join(image_dir, all_jpgs[0])
            print(f"\n[demo] Using first dataset image: {IMAGE_PATH}")
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}")
            print("        Set IMAGE_PATH manually in main() and re-run.")
            return

    # ── Step 6: Generate caption (greedy + beam) ───────────────────────────────
    print(f"\n[predict] Extracting VGG16 fc2 features...")
    feature = extract_image_feature(vgg_model, IMAGE_PATH)
    print(f"[predict] Feature shape: {feature.shape}  |  L2-norm: {np.linalg.norm(feature):.4f}")

    print(f"\n[predict] Greedy caption:")
    greedy_cap = generate_caption(caption_model, tokenizer, feature, max_length)
    print(f"  {greedy_cap}")

    print(f"\n[predict] Beam search caption (beam_width={DEFAULT_BEAM_WIDTH}):")
    beam_cap = beam_search_caption(caption_model, tokenizer, feature, max_length, DEFAULT_BEAM_WIDTH)
    print(f"  {beam_cap}")

    # ── Step 7: Image dependence test ─────────────────────────────────────────
    try:
        image_dir = resolve_image_dir(IMAGE_DIR_OPTIONS)
        image_dependence_test(caption_model, vgg_model, tokenizer, max_length,
                              image_dir, n=10, beam_width=DEFAULT_BEAM_WIDTH)
    except FileNotFoundError:
        print("[skip] Image directory not found — skipping dependence test.")

    # ── Step 8: Final result ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"  Image   : {os.path.basename(IMAGE_PATH)}")
    print(f"  Greedy  : {greedy_cap}")
    print(f"  Beam({DEFAULT_BEAM_WIDTH}) : {beam_cap}")
    print("=" * 60)

    return beam_cap


if __name__ == "__main__":
    main()
