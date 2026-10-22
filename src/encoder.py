"""
src/encoder.py
==============
SINGLE SOURCE OF TRUTH for the image encoder.

feature_extraction.py (training features), predict.py (inference) and
model.py (input size) all import from here, so they can never disagree.

Encoder: InceptionV3 (ImageNet), global-average-pooled -> 2048-d vector.
    Input  : (299, 299, 3), pixels scaled to [-1, 1] by inception_v3.preprocess_input
    Output : (2048,)

To try a different encoder later (ResNet50, EfficientNet...), change ONLY this file
and re-run feature_extraction.py + train.py.
"""

import numpy as np

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as _inception_preprocess

ENCODER_NAME = "inceptionv3"
IMG_SIZE     = (299, 299)
FEATURE_DIM  = 2048


def build_encoder():
    """Return a frozen InceptionV3 model with output shape (None, 2048)."""
    model = InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    model.trainable = False
    return model


def preprocess(batch):
    """batch: float array (N, 299, 299, 3) in 0..255 -> scaled to [-1, 1]."""
    return _inception_preprocess(batch)


def l2_normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-8)
