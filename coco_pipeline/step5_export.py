"""
step5_export.py  -  STEP 5 of 5  (a minute or two)

Packs everything the local app needs into  <Drive>/coco_captioning/final/ :
    caption_model.h5      the best trained model, in the same format your app already loads
    tokenizer.pkl         the vocabulary
    word_index.json       the same vocabulary as plain JSON (backup, see rebuild_tokenizer.py)
    meta.json             vocab_size and max_length (the app reads max_length from here)
    training_curves.png   loss / accuracy graph for your report
    bleu_results.json     from step 4 (if you ran it)

It also CHECKS the exported .h5 the way your app loads it (rebuild the model, load the
weights by layer name) and refuses to finish if the predictions do not match.
"""
import os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_config import *
from coco_common import *


def main():
    import keras
    best = p(CKPT_DIR, "best.keras")
    if not os.path.exists(best):
        raise SystemExit("[ERROR] checkpoints/best.keras not found. Run step 3 first.")
    ensure_dirs(FINAL_DIR)
    meta = read_json(p(PROC_DIR, "meta.json"))
    vocab_size, L = meta["vocab_size"], meta["max_length"]

    # ---- 1. model -> .h5 (done HERE, after training, because writing .h5 during training breaks resume)
    model = keras.models.load_model(best, compile=False)
    h5 = p(FINAL_DIR, "caption_model.h5")
    tmp = h5 + ".tmp.h5"
    model.save(tmp)
    os.replace(tmp, h5)
    log(f"[step5] wrote {h5}  ({os.path.getsize(h5) / 1e6:.0f} MB)")

    # ---- 2. verify it loads exactly like predict.py does (rebuild + load_weights by name)
    from model import build_model
    rebuilt = build_model(vocab_size, L)
    rebuilt.load_weights(h5, by_name=True)
    rng = np.random.default_rng(0)
    imgs = rng.random((8, 2048), dtype=np.float32)
    imgs /= np.linalg.norm(imgs, axis=1, keepdims=True)
    seqs = rng.integers(1, vocab_size, size=(8, L)).astype("int32")
    a = np.array(model({"image_input": imgs, "seq_input": seqs}, training=False))
    b = np.array(rebuilt({"image_input": imgs, "seq_input": seqs}, training=False))
    if not np.allclose(a, b, atol=1e-5):
        raise SystemExit("[ERROR] exported .h5 does NOT reproduce the trained model. Do not use it. "
                         "Send me this message.")
    log("[step5] check passed: the .h5 file loads like the app loads it and gives identical predictions")

    # ---- 3. the other files
    for name in ["tokenizer.pkl", "word_index.json", "meta.json"]:
        shutil.copy(p(PROC_DIR, name), p(FINAL_DIR, name))
        log(f"[step5] copied {name}")

    hist_path = p(CKPT_DIR, "history.json")
    if os.path.exists(hist_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = read_json(hist_path)
        ep = [r["epoch"] for r in h]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ep, [r["loss"] for r in h], label="train"); ax[0].plot(ep, [r["val_loss"] for r in h], label="validation")
        ax[0].set_title("Loss"); ax[0].set_xlabel("mini-epoch (2000 steps each)"); ax[0].legend()
        ax[1].plot(ep, [r["accuracy"] for r in h], label="train"); ax[1].plot(ep, [r["val_accuracy"] for r in h], label="validation")
        ax[1].set_title("Next-word accuracy"); ax[1].set_xlabel("mini-epoch (2000 steps each)"); ax[1].legend()
        fig.tight_layout(); fig.savefig(p(FINAL_DIR, "training_curves.png"), dpi=120); plt.close(fig)
        log("[step5] wrote training_curves.png")

    log("\n[step5] DONE. Download these from Drive (MyDrive/coco_captioning/final/):")
    for f in sorted(os.listdir(FINAL_DIR)):
        if not f.endswith(".tmp.h5"):
            log(f"          {f}")


if __name__ == "__main__":
    main()
