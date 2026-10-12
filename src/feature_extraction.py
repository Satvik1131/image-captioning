"""
src/feature_extraction.py
===========================
STEP 2 of the Image Captioning Pipeline:
    Load images → Preprocess for VGG16 → Extract 4096-dim fc2 feature vectors →
    L2-normalise every vector → Save {image_id: feature_vector} dict to
    data/processed/features.pkl

WHY fc2 INSTEAD OF pooling='avg'?
──────────────────────────────────
VGG16(include_top=False, pooling='avg') gives a 512-dim Global Average Pooling
vector. While compact, it discards the learned semantic representations from
VGG16's fully-connected layers (fc1, fc2) that were trained to distinguish
between 1000 ImageNet categories. The fc2 layer output (4096-d) carries richer,
more discriminative image semantics that give the LSTM decoder a much stronger
visual signal to condition on.

WHY L2-NORMALISE?
─────────────────
Raw fc2 activations can have very different magnitudes across images (range:
tens to thousands). L2-normalisation maps every vector to the unit hypersphere
so cosine similarity == dot product, making the feature space more uniform and
preventing large-magnitude images from dominating the loss.

VGG16 Architecture Summary:
    Input image    : (224, 224, 3)
    ... 5 conv blocks → (7, 7, 512) feature maps ...
    Flatten        : (25088,)
    fc1 (Dense 4096, relu)
    fc2 (Dense 4096, relu)   ← we extract from HERE
    predictions (Dense 1000, softmax)  ← discarded

Syllabus Alignment:
    - Unit IV : Convolutions & Pooling — VGG16's conv blocks
    - Unit III: Transfer Learning — frozen ImageNet weights reused as extractor
"""

import os
import pickle

import numpy as np
from PIL import Image
from tqdm import tqdm

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Handles Flickr8k folder-name variants across download sources
IMAGE_DIR_OPTIONS = [
    os.path.join(PROJECT_ROOT, "data", "Flickr8k_Dataset", "Flicker8k_Dataset"),
    os.path.join(PROJECT_ROOT, "data", "Flickr8k_Dataset", "Flickr8k_Dataset"),
    os.path.join(PROJECT_ROOT, "data", "images"),
]

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.pkl")

IMG_SIZE     = (224, 224)   # VGG16 required input size
LOG_EVERY_N  = 500          # console log frequency


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — resolve_image_dir
# ──────────────────────────────────────────────────────────────────────────────

def resolve_image_dir(candidates):
    """
    Find the first existing image directory from a list of candidate paths.

    Handles the Flickr8k dataset folder-name inconsistency across download
    sources (Kaggle vs official zip — "Flicker" vs "Flickr" typo).

    Args:
        candidates (list of str): Ordered list of paths to check.

    Returns:
        str: The first path that exists on disk.

    Raises:
        FileNotFoundError: If none of the candidate paths exist.
    """
    for path in candidates:
        if os.path.isdir(path):
            print(f"[resolve] Found image directory: {path}")
            return path

    raise FileNotFoundError(
        "Could not find the Flickr8k image directory!\n"
        "Checked:\n" + "\n".join(f"  {p}" for p in candidates) +
        "\nPlease unzip Flickr8k_Dataset.zip into data/Flickr8k_Dataset/ and re-run."
    )


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — load_vgg16_model
# ──────────────────────────────────────────────────────────────────────────────

def load_vgg16_model():
    """
    Build a VGG16 feature extractor that outputs the fc2 layer (4096-d).

    Configuration:
        weights='imagenet'  — full VGG16 weights including the Dense classifier
        include_top=True    — keep the Dense fc1 + fc2 layers
        (no pooling arg needed when include_top=True)

    We then build a sub-model whose output is the 'fc2' layer activation
    (before the final softmax classifier). This gives a 4096-dim vector that
    encodes rich, discriminative image semantics learned from 1.2M images.

    The sub-model weights are frozen — forward-pass only.

    Returns:
        tf.keras.Model: Extractor with output shape (None, 4096).
    """
    print("[model] Loading VGG16 (include_top=True) for fc2 feature extraction...")

    base = VGG16(
        weights='imagenet',
        include_top=True,          # keep the Dense classification head
        input_shape=(224, 224, 3)  # must be specified when include_top=True
    )

    # Build sub-model: input → fc2 output (4096-d)
    model = Model(
        inputs=base.input,
        outputs=base.get_layer('fc2').output,
        name='vgg16_fc2_extractor'
    )

    model.trainable = False   # freeze — inference only
    print(f"[model] VGG16 fc2 extractor ready. Output shape: {model.output_shape}")
    # Expected: (None, 4096)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — preprocess_image
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path):
    """
    Load a single image and prepare it for VGG16 inference.

    Preprocessing pipeline (must match predict.py):
        1. Open with PIL        → RGB image, any size
        2. Resize to (224, 224) → VGG16 required input size
        3. img_to_array         → numpy (224, 224, 3) float32
        4. expand_dims          → (1, 224, 224, 3) batch tensor
        5. preprocess_input     → subtract ImageNet per-channel mean + BGR swap

    Args:
        image_path (str): Absolute path to a .jpg image file.

    Returns:
        np.ndarray of shape (1, 224, 224, 3), or None on error.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"  [warn] Could not process {os.path.basename(image_path)}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — l2_normalize
# ──────────────────────────────────────────────────────────────────────────────

def l2_normalize(vector):
    """
    L2-normalise a feature vector to unit length.

    v_norm = v / (||v||_2 + eps)

    Adding eps=1e-8 prevents division by zero for all-zero vectors (degenerate
    case that should never occur with real images but is safe to guard against).

    After normalisation:
        ||v_norm||_2 ≈ 1.0 for every image
        cosine_similarity(a, b) == dot(a, b)  ← makes similarity well-defined

    Args:
        vector (np.ndarray): Feature vector of any shape.

    Returns:
        np.ndarray: Unit-length vector of the same shape.
    """
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — extract_features
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(model, image_dir):
    """
    Iterate over all .jpg images, run VGG16 fc2 inference, L2-normalise,
    and collect feature vectors.

    For each image:
        preprocessed tensor (1, 224, 224, 3)
            → VGG16 fc2 forward pass
            → (1, 4096) raw feature
            → squeeze → (4096,)
            → L2-normalise → (4096,) on unit sphere
            → store with filename as key

    Args:
        model     : VGG16 fc2 extractor (output shape: (None, 4096))
        image_dir : Directory containing .jpg image files

    Returns:
        dict: { "1000268201_693b08cb0e.jpg": np.array of shape (4096,) }
    """
    all_images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    total   = len(all_images)
    skipped = 0
    print(f"[extract] Found {total} image files in: {image_dir}")
    print(f"[extract] Extracting VGG16 fc2 features (4096-d) with L2 normalisation...\n")

    features = {}

    for idx, filename in enumerate(tqdm(all_images, desc="Extracting features", unit="img"), start=1):
        image_path = os.path.join(image_dir, filename)
        img_tensor = preprocess_image(image_path)

        if img_tensor is None:
            skipped += 1
            continue

        # Forward pass → (1, 4096)
        feature_vector = model.predict(img_tensor, verbose=0)
        feature_vector = feature_vector.squeeze()   # (4096,)

        # L2-normalise to unit sphere
        feature_vector = l2_normalize(feature_vector)

        features[filename] = feature_vector

        if idx % LOG_EVERY_N == 0 or idx == total:
            print(f"\n  [progress] {idx}/{total} | features={len(features)} | skipped={skipped}")

    print(f"\n[extract] Done. {len(features)} feature vectors extracted (4096-d, L2-normalised).")
    if skipped:
        print(f"[extract] Warning: {skipped} images skipped (corrupted or unreadable).")

    return features


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 6 — verify_features
# ──────────────────────────────────────────────────────────────────────────────

def verify_features(features, n_pairs=100):
    """
    Print verification statistics to confirm features are image-specific.

    Checks:
        1. Feature shape == (4096,)
        2. L2 norm ≈ 1.0
        3. Mean cosine similarity between random pairs << 0.9
           (if all captions were identical, this would approach 1.0)

    Args:
        features (dict): { image_id: np.array(4096,) }
        n_pairs (int)  : Number of random image pairs to sample for similarity.
    """
    keys = list(features.keys())
    sample_vec = features[keys[0]]

    print(f"\n[verify] Feature shape      : {sample_vec.shape}")   # expect (4096,)
    print(f"[verify] Sample L2 norm     : {np.linalg.norm(sample_vec):.6f}")  # expect ~1.0

    # Sample n_pairs random pairs and compute cosine similarity
    np.random.seed(0)
    sims = []
    for _ in range(n_pairs):
        i, j = np.random.choice(len(keys), 2, replace=False)
        a = features[keys[i]]
        b = features[keys[j]]
        sims.append(float(np.dot(a, b)))  # dot of unit vectors == cosine sim

    mean_sim = np.mean(sims)
    print(f"[verify] Mean cosine similarity over {n_pairs} random pairs: {mean_sim:.4f}")
    print(f"         (Should be clearly below 0.9 — if ~1.0, features are degenerate)")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 7 — save_features
# ──────────────────────────────────────────────────────────────────────────────

def save_features(features, path):
    """
    Save the extracted feature dictionary to disk using pickle.

    File size estimate:
        8092 images × 4096 floats × 4 bytes = ~133 MB

    Args:
        features (dict): { image_id: np.array(4096,) }
        path (str)     : Absolute path to save the .pkl file
    """
    with open(path, "wb") as f:
        pickle.dump(features, f)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[save] Features saved to: {path}  ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Full feature extraction pipeline:

        1. Create output directories
        2. Resolve the image directory path
        3. Load VGG16 with include_top=True, build fc2 (4096-d) sub-model
        4. Preprocess and extract a 4096-dim vector for every image
        5. L2-normalise each vector
        6. Verify feature statistics (shape, norm, pairwise similarity)
        7. Save the features dict to data/processed/features.pkl

    Runtime estimate:
        ~8092 images on CPU: 20–50 minutes (fc2 is deeper than GAP).
        With GPU: ~3–8 minutes.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("[init] Output directories confirmed.\n")

    image_dir = resolve_image_dir(IMAGE_DIR_OPTIONS)
    model     = load_vgg16_model()

    features  = extract_features(model, image_dir)

    # ── Verification ──────────────────────────────────────────────────────────
    verify_features(features, n_pairs=100)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_features(features, FEATURES_PATH)

    print("\n" + "="*60)
    print("  FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print(f"  Images extracted   : {len(features)}")
    sample = next(iter(features.values()))
    print(f"  Feature shape      : {sample.shape}")   # (4096,)
    print(f"  L2 norm (sample)   : {np.linalg.norm(sample):.6f}")  # ~1.0
    print(f"  Features saved     : {FEATURES_PATH}")
    print("="*60)
    print("\nNext step -> run src/caption_preprocessing.py (if not done) then src/train.py")


if __name__ == "__main__":
    main()
