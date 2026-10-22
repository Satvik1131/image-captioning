"""
coco_config.py - shared settings for the COCO pipeline.

Everything that must survive a Colab disconnect lives under BASE (Google Drive).
Every step is RESUMABLE: re-running it skips whatever is already finished.
"""
import os

# Folder on Google Drive that holds ALL results (override with env var COCO_BASE).
BASE = os.environ.get("COCO_BASE", "/content/drive/MyDrive/coco_captioning")

ANN_DIR   = os.path.join(BASE, "annotations")   # raw COCO caption json files
PROC_DIR  = os.path.join(BASE, "processed")     # cleaned captions, tokenizer, manifests
FEAT_DIR  = os.path.join(BASE, "features")      # image features, saved in small chunks
CKPT_DIR  = os.path.join(BASE, "checkpoints")   # training state (resume from here)
FINAL_DIR = os.path.join(BASE, "final")         # files you download at the end

# ---- text settings (same rules as the Flickr8k pipeline) ----
START_TOKEN = "startseq"
END_TOKEN   = "endseq"
UNK_TOKEN   = "unk"
MIN_WORD_COUNT       = 5     # words rarer than this become 'unk'
MAX_CAPTION_WORDS    = 30    # longer captions are dropped (tiny fraction of COCO)
MAX_CAPTIONS_PER_IMG = 5     # COCO has 5-7 per image; keep at most 5

# ---- training ----
DEFAULT_MAX_EPOCHS = 40      # mini-epochs of 2000 steps (about 6 passes over COCO)

# ---- feature extraction ----
CHUNK_SIZE = 4000            # images per saved chunk (a disconnect loses <= 1 chunk)

ANN_URLS = [
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
]
ANN_MEMBERS = {
    "train": "annotations/captions_train2017.json",
    "val":   "annotations/captions_val2017.json",
}

def p(*parts):
    return os.path.join(*parts)
