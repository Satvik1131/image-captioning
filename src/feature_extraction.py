"""
src/feature_extraction.py
===========================
STEP 2 of the Image Captioning Pipeline:
    Load images → Preprocess for VGG16 → Extract 512-dim feature vectors →
    Save {image_id: feature_vector} dict to data/processed/features.pkl

Syllabus Alignment:
    - Unit IV : Convolutions & Pooling — VGG16 is a 16-layer deep CNN that uses
                repeated 3x3 convolution + 2x2 max-pooling blocks to extract
                hierarchical spatial features from raw pixel input.
    - Unit III: Transfer Learning / Dropout — we freeze VGG16 weights (trained
                on ImageNet) and use it purely as a feature extractor, avoiding
                the need to train a CNN from scratch.

VGG16 Architecture Summary (what happens to each image):
    Input image   : (224, 224, 3)   — 3-channel RGB, resized to 224x224
    After Block 1 : (112, 112, 64)  — 2x Conv2D(64) + MaxPool
    After Block 2 : (56,  56,  128) — 2x Conv2D(128) + MaxPool
    After Block 3 : (28,  28,  256) — 3x Conv2D(256) + MaxPool
    After Block 4 : (14,  14,  512) — 3x Conv2D(512) + MaxPool
    After Block 5 : (7,   7,   512) — 3x Conv2D(512) + MaxPool
    Global Avg Pool: (512,)         — average each 7x7 feature map → 1 scalar
    OUTPUT        : (512,)          — the image feature vector we save

    With include_top=False + pooling='avg' Keras does blocks 1-5 + GAP for us.

Output:
    data/processed/features.pkl  — dict { "image_filename.jpg": np.array shape (512,) }
"""

import os
import pickle

import numpy as np
from PIL import Image                               # Pillow: load .jpg files
from tqdm import tqdm                               # progress bar per image

from tensorflow.keras.applications import VGG16                       # Unit IV: CNN
from tensorflow.keras.applications.vgg16 import preprocess_input      # VGG16 pixel normalisation
from tensorflow.keras.preprocessing.image import img_to_array         # PIL → numpy array

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Absolute project root — derive all paths from here
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Image dataset folder
# NOTE: The original Flickr8k zip unpacks to "Flicker8k_Dataset" (typo in dataset name)
# We check both spellings and also data/images/ as a fallback
IMAGE_DIR_OPTIONS = [
    os.path.join(PROJECT_ROOT, "data", "Flickr8k_Dataset", "Flicker8k_Dataset"),  # primary
    os.path.join(PROJECT_ROOT, "data", "Flickr8k_Dataset", "Flickr8k_Dataset"),   # correct spelling
    os.path.join(PROJECT_ROOT, "data", "images"),                                  # fallback
]

# Output paths
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.pkl")

# VGG16 input size — the network was trained on 224x224 images; all inputs must match
IMG_SIZE = (224, 224)

# Print a progress log every N images (keeps the console readable)
LOG_EVERY_N = 500


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — resolve_image_dir
# ──────────────────────────────────────────────────────────────────────────────

def resolve_image_dir(candidates):
    """
    Find the first existing image directory from a list of candidate paths.

    This handles the common issue where the Flickr8k dataset is downloaded
    from different sources (Kaggle vs official) and unpacked with slightly
    different folder names.

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
    Load VGG16 pre-trained on ImageNet, configured as a feature extractor.

    Key parameters explained:
        weights='imagenet'    — Use weights learned on 1.2M ImageNet images.
                                These low/mid-level feature detectors (edges,
                                textures, shapes) transfer well to new images.
        include_top=False     — Discard the final 3 fully-connected layers
                                (the "classifier head"). We only want the
                                convolutional feature maps.
        pooling='avg'         — After the last conv block (7x7x512), apply
                                Global Average Pooling: average each of the
                                512 feature maps → output shape (512,).
                                This replaces the flattened 25088-dim vector
                                with a compact, fixed-size representation.

    Input  shape : (batch, 224, 224, 3)
    Output shape : (batch, 512)          ← this is our image feature vector

    Unit IV connection:
        VGG16's 13 conv layers repeatedly apply:
            Conv2D(filters, kernel=3x3, padding='same', activation='relu')
        followed by:
            MaxPooling2D(pool_size=2x2, strides=2)
        This halves the spatial dimensions at each pooling step while doubling
        the depth, building increasingly abstract feature representations.

    Returns:
        model (tf.keras.Model): VGG16 feature extractor, weights frozen.
    """
    print("[model] Loading VGG16 with ImageNet weights (feature extractor mode)...")

    model = VGG16(
        weights='imagenet',   # download/use cached ImageNet weights (~550MB first run)
        include_top=False,    # drop the Dense(1000) classification head
        pooling='avg'         # Global Average Pooling → output shape: (512,)
    )

    # Freeze all weights — we are NOT training VGG16, only using it to extract features
    # This means no gradients will be computed through VGG16 during training
    model.trainable = False

    print(f"[model] VGG16 loaded. Output shape per image: {model.output_shape}")
    # Expected: (None, 512)  — None = variable batch size, 512 = feature vector dim
    return model


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — preprocess_image
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path):
    """
    Load a single image from disk and prepare it for VGG16 inference.

    Preprocessing pipeline:
        1. Open with PIL  → RGB image of any size
        2. Resize         → exactly (224, 224) pixels  [VGG16 requirement]
        3. img_to_array   → numpy array of shape (224, 224, 3), dtype float32
        4. expand_dims    → shape (1, 224, 224, 3)     [add batch dimension]
        5. preprocess_input → normalise pixel values using VGG16's own stats:
                              subtracts the ImageNet mean per channel
                              [R: -103.939, G: -116.779, B: -123.68]
                              and converts RGB → BGR (VGG16 was trained this way)

    Args:
        image_path (str): Absolute path to a .jpg image file.

    Returns:
        np.ndarray: Preprocessed image tensor, shape (1, 224, 224, 3).
                    Ready to be passed to model.predict().

    Returns None if the image cannot be opened (corrupted / missing file).
    """
    try:
        # Step 1+2: Open and resize
        img = Image.open(image_path).convert('RGB')  # ensure 3 channels (not RGBA/grayscale)
        img = img.resize(IMG_SIZE)                   # → (224, 224) PIL Image

        # Step 3: Convert PIL Image → numpy array
        # Shape becomes: (224, 224, 3) — height × width × channels
        img_array = img_to_array(img)

        # Step 4: Add batch dimension → (1, 224, 224, 3)
        # model.predict() always expects a BATCH even if it's a single image
        img_array = np.expand_dims(img_array, axis=0)

        # Step 5: Apply VGG16-specific pixel normalisation
        # Without this, the model receives out-of-distribution pixel values
        # and the extracted features are meaningless
        img_array = preprocess_input(img_array)

        return img_array  # shape: (1, 224, 224, 3)

    except Exception as e:
        print(f"  [warn] Could not process image {os.path.basename(image_path)}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — extract_features
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(model, image_dir):
    """
    Iterate over all .jpg images, run VGG16 inference, collect feature vectors.

    For each image:
        preprocessed tensor (1, 224, 224, 3)
            → VGG16 forward pass
            → feature vector (1, 512)
            → squeeze → (512,)
            → store in dict with image filename as key

    Args:
        model     : Loaded VGG16 Keras model (output shape per image: (512,))
        image_dir : Directory containing .jpg image files

    Returns:
        dict: { "1000268201_693b08cb0e.jpg": np.array of shape (512,), ... }
    """
    # Collect all .jpg files in the image directory
    all_images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    all_images.sort()  # deterministic ordering

    total = len(all_images)
    print(f"[extract] Found {total} image files in: {image_dir}")
    print(f"[extract] Starting VGG16 feature extraction... (logs every {LOG_EVERY_N} images)\n")

    features = {}   # will hold { image_filename: feature_vector (512,) }
    skipped  = 0    # count of images we couldn't process

    # tqdm wraps the list and draws a live progress bar in the terminal
    for idx, filename in enumerate(tqdm(all_images, desc="Extracting features", unit="img"), start=1):

        image_path = os.path.join(image_dir, filename)

        # Preprocess: resize + normalise → shape (1, 224, 224, 3)
        img_tensor = preprocess_image(image_path)

        if img_tensor is None:
            skipped += 1
            continue  # skip corrupted / unreadable images

        # Run the forward pass through VGG16
        # model.predict() returns shape (1, 512) — batch of 1 image
        feature_vector = model.predict(img_tensor, verbose=0)

        # Remove the batch dimension: (1, 512) → (512,)
        # This is the compact image representation the LSTM decoder will use
        feature_vector = feature_vector.squeeze()  # shape: (512,)

        # Store using the filename as key — matches keys in clean_captions.json
        features[filename] = feature_vector

        # ── Progress log every LOG_EVERY_N images ─────────────────────────────
        if idx % LOG_EVERY_N == 0 or idx == total:
            print(f"\n  [progress] {idx}/{total} images processed "
                  f"| {len(features)} features collected "
                  f"| {skipped} skipped")

    print(f"\n[extract] Done. {len(features)} feature vectors extracted.")
    if skipped:
        print(f"[extract] Warning: {skipped} images were skipped (corrupted or unreadable).")

    return features


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — save_features
# ──────────────────────────────────────────────────────────────────────────────

def save_features(features, path):
    """
    Save the extracted feature dictionary to disk using pickle.

    Why pickle?
        The dictionary values are numpy arrays (float32, shape 512).
        Pickle serialises numpy arrays efficiently in binary format.
        Loading back with pickle.load() restores the exact numpy arrays.

    File size estimate:
        8092 images × 512 floats × 4 bytes = ~16.6 MB

    This file is loaded by:
        - data_preprocessing.py (merge features + captions into training pairs)
        - predict.py            (extract feature for a new query image at inference)
        - app/app.py            (real-time feature extraction in the Streamlit UI)

    Args:
        features (dict) : { image_id: np.array(512,) }
        path (str)      : Absolute path to save the .pkl file
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
        3. Load VGG16 as a feature extractor (include_top=False, pooling='avg')
        4. Preprocess and extract a 512-dim vector for every image
        5. Save the features dict to data/processed/features.pkl

    Runtime estimate:
        ~8092 images, CPU-only: 15–40 minutes depending on hardware.
        With GPU: ~2–5 minutes.
        The tqdm bar + LOG_EVERY_N logs keep you informed of progress.
    """

    # -- Step 0: Ensure output directory exists --------------------------------
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("[init] Output directories confirmed.\n")

    # -- Step 1: Locate the image directory ------------------------------------
    image_dir = resolve_image_dir(IMAGE_DIR_OPTIONS)

    # -- Step 2: Load VGG16 model ---------------------------------------------
    model = load_vgg16_model()

    # Quick sanity check — print first layer and last layer names
    print(f"[model] First layer : {model.layers[0].name}")
    print(f"[model] Last layer  : {model.layers[-1].name}\n")

    # -- Step 3: Extract features for all images --------------------------------
    features = extract_features(model, image_dir)

    # -- Step 4: Save to disk --------------------------------------------------
    save_features(features, FEATURES_PATH)

    # -- Step 5: Summary -------------------------------------------------------
    print("\n" + "="*60)
    print("  FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print(f"  Images extracted   : {len(features)}")
    print(f"  Feature shape      : {next(iter(features.values())).shape}")
    print(f"  Features saved     : {FEATURES_PATH}")
    print("="*60)
    print("\nNext step -> run src/data_preprocessing.py")


if __name__ == "__main__":
    main()
