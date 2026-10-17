"""src/diagnose.py - checks whether the model actually uses the image."""
import os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import (load_clean_captions, load_features,
                                load_tokenizer, get_max_length,
                                CLEAN_CAPTIONS_PATH, FEATURES_PATH, TOKENIZER_PATH)
from predict import (load_caption_model, beam_search_caption,
                     CAPTION_MODEL_PATH)
from tensorflow.keras.preprocessing.sequence import pad_sequences

caps  = load_clean_captions(CLEAN_CAPTIONS_PATH)
feats = load_features(FEATURES_PATH)
tok   = load_tokenizer(TOKENIZER_PATH)
vocab = len(tok.word_index) + 1
maxlen = get_max_length(caps)
model = load_caption_model(CAPTION_MODEL_PATH, vocab, maxlen)
idx2w = tok.index_word

ids = [i for i in caps if i in feats]
random.seed(1); sample = random.sample(ids, 8)

# TEST 1: are the image features different from each other?
sims = []
for _ in range(200):
    a, b = random.sample(ids, 2)
    fa, fb = feats[a].ravel(), feats[b].ravel()
    sims.append(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb)))
print(f"\n[TEST 1] mean cosine similarity between different images: {np.mean(sims):.3f}")
print("         (below ~0.9 is OK; near 1.0 means features are not distinguishing images)")

# TEST 2: does the first predicted word change with the image?
print("\n[TEST 2] top-3 first words for 4 different images:")
seq = pad_sequences([tok.texts_to_sequences(["startseq"])[0]], maxlen=maxlen, padding='post')
for i in sample[:4]:
    p = model.predict({'image_input': feats[i].reshape(1, -1), 'seq_input': seq}, verbose=0)[0]
    top = np.argsort(p)[-3:][::-1]
    print(f"   {i}: " + ", ".join(f"{idx2w.get(int(t))}({p[t]:.2f})" for t in top))

# TEST 3: generated vs real captions
print("\n[TEST 3] generated vs real:")
for i in sample:
    gen = beam_search_caption(model, tok, feats[i].reshape(1, -1), maxlen, beam_width=3)
    real = caps[i][0].replace('startseq', '').replace('endseq', '').strip()
    print(f"   {i}\n      model: {gen}\n      real : {real}")