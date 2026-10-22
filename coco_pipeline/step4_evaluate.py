"""
step4_evaluate.py  -  STEP 4 of 5  (GPU, about 5-15 minutes)

Measures BLEU-1..4 on COCO val2017 images the model NEVER trained on, using beam
search, and prints example captions next to the human captions.

Usage
  python step4_evaluate.py              (1000 random validation images, beam 3)
  python step4_evaluate.py --n 2000 --beam 5
"""
import argparse, os, random, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_config import *
from coco_common import *


def beam_caption(model, word_index, index_word, feat, L, beam=3):
    """Beam search that scores all beams in ONE model call per step. 'unk' is never allowed."""
    from keras.utils import pad_sequences
    unk = word_index.get(UNK_TOKEN)
    start, end = word_index[START_TOKEN], word_index[END_TOKEN]
    beams, done = [(0.0, [start])], []
    for _ in range(L):
        seqs = pad_sequences([b[1] for b in beams], maxlen=L, padding="post")
        imgs = np.repeat(feat, len(beams), axis=0)
        probs = model({"image_input": imgs, "seq_input": seqs}, training=False)
        probs = np.array(probs)
        cand = []
        for (lp, toks), pr in zip(beams, probs):
            if unk is not None:
                pr[unk] = 0.0
            for idx in np.argsort(pr)[-beam:]:
                cand.append((lp + float(np.log(pr[idx] + 1e-10)), toks + [int(idx)]))
        beams = []
        for lp, toks in sorted(cand, key=lambda x: x[0], reverse=True)[:beam]:
            (done if toks[-1] == end else beams).append((lp, toks))
        if not beams:
            break
    done += beams
    best = max(done, key=lambda x: x[0] / len(x[1]))              # length-normalised score
    return [index_word[i] for i in best[1] if i not in (start, end)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of validation images")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--model", default=p(CKPT_DIR, "best.keras"))
    args = ap.parse_args()

    import keras
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    meta = read_json(p(PROC_DIR, "meta.json"))
    L = meta["max_length"]
    word_index = read_json(p(PROC_DIR, "word_index.json"))
    index_word = {int(v): k for k, v in word_index.items()}
    refs_all = read_json(p(PROC_DIR, "val_captions_clean.json"))
    feats = np.load(p(FEAT_DIR, "features_val.npy"))
    ids = read_json(p(FEAT_DIR, "ids_val.json"))

    if not os.path.exists(args.model):
        raise SystemExit(f"[ERROR] {args.model} not found. Run step 3 first.")
    model = keras.models.load_model(args.model, compile=False)
    log(f"[step4] loaded {args.model}")

    rows = [r for r, i in enumerate(ids) if str(i) in refs_all]
    random.seed(0)
    rows = random.sample(rows, min(args.n, len(rows)))
    log(f"[step4] decoding {len(rows)} validation images with beam {args.beam} ...")

    refs, hyps, examples = [], [], []
    t0 = time.time()
    for k, r in enumerate(rows, 1):
        hyp = beam_caption(model, word_index, index_word, feats[r:r + 1].astype(np.float32), L, args.beam)
        ref = [c.split() for c in refs_all[str(ids[r])]]
        hyps.append(hyp); refs.append(ref)
        if len(examples) < 12:
            examples.append((ids[r], " ".join(hyp), " ".join(ref[0])))
        if k % 100 == 0:
            log(f"   {k}/{len(rows)} done ({time.time() - t0:.0f}s)")

    sm = SmoothingFunction().method1
    scores = {}
    for k, w in enumerate([(1, 0, 0, 0), (.5, .5, 0, 0), (1/3, 1/3, 1/3, 0), (.25, .25, .25, .25)], 1):
        scores[f"BLEU-{k}"] = round(corpus_bleu(refs, hyps, weights=w, smoothing_function=sm), 3)

    log("\nExamples (model vs one human caption):")
    for i, h, r in examples:
        log(f"  image {i}\n     model: {h}\n     human: {r}")
    log("\nRESULTS (COCO val2017, %d images, beam %d):" % (len(rows), args.beam))
    for k, v in scores.items():
        log(f"  {k}: {v:.3f}")
    ensure_dirs(FINAL_DIR)
    write_json_atomic({"n_images": len(rows), "beam": args.beam, **scores}, p(FINAL_DIR, "bleu_results.json"))
    log(f"\n[step4] saved {p(FINAL_DIR, 'bleu_results.json')}")


if __name__ == "__main__":
    main()
