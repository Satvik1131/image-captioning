"""
rebuild_tokenizer.py  -  LOCAL BACKUP (run on your own PC, only if tokenizer.pkl will not load)

Rebuilds models/tokenizer.pkl from models/word_index.json, using your own TensorFlow version.
Usage (from the project folder):   python rebuild_tokenizer.py
"""
import json, os, pickle
from tensorflow.keras.preprocessing.text import Tokenizer

models = "models"
wi = json.load(open(os.path.join(models, "word_index.json"), encoding="utf-8"))
tok = Tokenizer(oov_token=None, filters='')
tok.word_index = {k: int(v) for k, v in wi.items()}
tok.index_word = {int(v): k for k, v in wi.items()}
with open(os.path.join(models, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tok, f)
print("rebuilt models/tokenizer.pkl with", len(wi), "words")
