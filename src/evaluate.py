import os, sys, random, time
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import (load_clean_captions, load_features, load_tokenizer,
                                get_max_length, get_train_val_split,
                                CLEAN_CAPTIONS_PATH, FEATURES_PATH, TOKENIZER_PATH)
from predict import load_caption_model, CAPTION_MODEL_PATH

N_IMAGES = 500     
BEAM     = 3

def fast_beam(model, tok, feat, maxlen, beam=BEAM):
    """Beam search that scores all beams in ONE model call per step."""
    unk   = tok.word_index.get('unk')
    start = tok.word_index['startseq']
    end   = tok.word_index['endseq']
    beams, done = [(0.0, [start])], []
    for _ in range(maxlen):
        seqs  = pad_sequences([b[1] for b in beams], maxlen=maxlen, padding='post')
        imgs  = np.repeat(feat, len(beams), axis=0)
        probs = model([imgs, seqs], training=False).numpy()
        cand = []
        for (lp, toks), p in zip(beams, probs):
            if unk is not None:
                p[unk] = 0.0
            for idx in np.argsort(p)[-beam:]:
                cand.append((lp + np.log(p[idx] + 1e-10), toks + [int(idx)]))
        beams = []
        for lp, toks in sorted(cand, key=lambda x: x[0], reverse=True)[:beam]:
            (done if toks[-1] == end else beams).append((lp, toks))
        if not beams:
            break
    done += beams
    best = max(done, key=lambda x: x[0] / len(x[1]))
    return [tok.index_word[i] for i in best[1] if i not in (start, end)]

caps  = load_clean_captions(CLEAN_CAPTIONS_PATH)
feats = load_features(FEATURES_PATH)
tok   = load_tokenizer(TOKENIZER_PATH)
maxlen = get_max_length(caps)
model = load_caption_model(CAPTION_MODEL_PATH, len(tok.word_index) + 1, maxlen)

_, val_caps, _, val_feats = get_train_val_split(caps, feats, 0.1)
ids = list(val_caps.keys())
random.seed(0)
ids = random.sample(ids, min(N_IMAGES, len(ids)))

clean = lambda s: s.replace('startseq', '').replace('endseq', '').split()
refs, hyps = [], []
t0 = time.time()
for n, i in enumerate(ids, 1):
    hyps.append(fast_beam(model, tok, val_feats[i].reshape(1, -1), maxlen))
    refs.append([clean(c) for c in val_caps[i]])
    if n % 10 == 0:
        print(f"  {n}/{len(ids)} done  ({time.time()-t0:.0f}s elapsed)", flush=True)

sm = SmoothingFunction().method1
for k, w in enumerate([(1,0,0,0), (.5,.5,0,0), (1/3,1/3,1/3,0), (.25,.25,.25,.25)], 1):
    print(f"BLEU-{k}: {corpus_bleu(refs, hyps, weights=w, smoothing_function=sm):.3f}")