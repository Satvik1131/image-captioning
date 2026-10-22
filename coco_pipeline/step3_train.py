"""
step3_train.py  -  STEP 3 of 5  (GPU, the longest step)

What it does
  Trains the caption model on all COCO train images (features from step 2).

Built so a disconnect costs you almost nothing
  Training is cut into short "mini-epochs" (2000 steps, a few minutes each).
  After EVERY mini-epoch it saves, to Drive:
      checkpoints/last.keras   full model + optimizer (used to resume)
      checkpoints/best.keras   best model so far (step 5 turns it into the file the app uses)
      checkpoints/state.json   epoch number, best loss, learning rate, patience counters
  Re-running this script continues exactly from the last saved mini-epoch.

Stops by itself when validation loss has not improved for 10 mini-epochs
(or after --max-epochs). Learning rate is halved after 4 mini-epochs without gain.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_config import *
from coco_common import *

LR_PATIENCE   = 4      # mini-epochs without improvement before halving the learning rate
STOP_PATIENCE = 10     # mini-epochs without improvement before stopping
MIN_LR        = 1e-5
MIN_DELTA     = 1e-4
VAL_SAMPLES   = 30000


# ---------------------------------------------------------------- data
def encode_caption(cap, word_index, unk_id):
    return [word_index.get(w, unk_id) for w in cap.split()]


def build_samples(caps, id_to_row, word_index, unk_id, L):
    """One caption of n tokens -> n-1 samples: (words so far) -> next word.
    Returns X (N,L) int16 padded AFTER, y (N,) int16, R (N,) int32 feature-row index."""
    seqs, rows = [], []
    for sid, cs in caps.items():
        row = id_to_row.get(int(sid))
        if row is None:
            continue
        for c in cs:
            s = encode_caption(c, word_index, unk_id)
            if 2 <= len(s) <= L:
                seqs.append(np.asarray(s, dtype=np.int16)); rows.append(row)
    total = sum(len(s) - 1 for s in seqs)
    X = np.zeros((total, L), dtype=np.int16)
    y = np.zeros(total, dtype=np.int16)
    R = np.zeros(total, dtype=np.int32)
    tri = np.tril(np.ones((L, L), dtype=bool))
    pos = 0
    for s, row in zip(seqs, rows):
        n = len(s) - 1
        X[pos:pos + n, :n] = np.where(tri[:n, :n], s[:n][None, :], 0)
        y[pos:pos + n] = s[1:]
        R[pos:pos + n] = row
        pos += n
    return X, y, R


def load_split(split, word_index, unk_id, L):
    feats = np.load(p(FEAT_DIR, f"features_{split}.npy"))
    ids = read_json(p(FEAT_DIR, f"ids_{split}.json"))
    id_to_row = {int(i): r for r, i in enumerate(ids)}
    caps = read_json(p(PROC_DIR, f"{split}_captions.json"))
    X, y, R = build_samples(caps, id_to_row, word_index, unk_id, L)
    log(f"[data] {split}: {len(ids):,} images with features, {len(y):,} training samples")
    return feats, X, y, R


def make_train_dataset(feats, X, y, R, batch):
    import tensorflow as tf
    N = len(y)

    def gen():
        rng = np.random.default_rng()
        while True:                                   # endless; every pass is freshly shuffled
            perm = rng.permutation(N)
            for i in range(0, N - batch + 1, batch):
                idx = np.sort(perm[i:i + batch])
                yield ({"image_input": feats[R[idx]].astype(np.float32),
                        "seq_input": X[idx].astype(np.int32)},
                       y[idx].astype(np.int32))

    sig = ({"image_input": tf.TensorSpec((batch, feats.shape[1]), tf.float32),
            "seq_input": tf.TensorSpec((batch, X.shape[1]), tf.int32)},
           tf.TensorSpec((batch,), tf.int32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(8)


# ---------------------------------------------------------------- resumable callback
class ResumableCheckpoint:
    """Built lazily so keras is imported only when training starts."""
    @staticmethod
    def create(paths, state, history):
        import keras

        class CB(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val = float(logs.get("val_loss", np.inf))
                improved = val < state["best_val"] - MIN_DELTA
                if improved:
                    state["best_val"] = val
                    state["wait"] = 0
                    state["best_epoch"] = epoch + 1
                    # NOTE: only .keras files are written during training. Saving a legacy .h5
                    # here would silently delete the optimizer settings from later .keras saves
                    # (Keras 3 side effect), breaking resume. Step 5 converts best.keras -> .h5.
                    tmp = paths["best"].replace(".keras", ".tmp.keras")
                    self.model.save(tmp)
                    os.replace(tmp, paths["best"])
                else:
                    state["wait"] += 1

                lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
                if (not improved) and state["wait"] > 0 and state["wait"] % LR_PATIENCE == 0 and lr > MIN_LR:
                    lr = max(lr * 0.5, MIN_LR)
                    self.model.optimizer.learning_rate.assign(lr)
                    log(f"         learning rate lowered to {lr:.2e}")
                state["lr"] = lr
                state["epoch"] = epoch + 1                       # number of mini-epochs completed

                history.append({"epoch": epoch + 1, "lr": lr, "improved": bool(improved),
                                **{k: float(v) for k, v in logs.items()}})

                # save resume files LAST; atomic replace so a disconnect cannot corrupt them
                tmp = paths["last"] + ".tmp.keras"
                self.model.save(tmp)
                os.replace(tmp, paths["last"])
                write_json_atomic(history, paths["history"])

                if state["wait"] >= STOP_PATIENCE:
                    state["done"] = True
                    self.model.stop_training = True
                    log(f"[step3] No improvement for {STOP_PATIENCE} mini-epochs - stopping.")
                write_json_atomic(state, paths["state"])
                tag = "  <- new best, saved" if improved else f"  (no gain {state['wait']}/{STOP_PATIENCE})"
                log(f"[step3] mini-epoch {epoch + 1} saved to Drive | val_loss {val:.4f}{tag}")
        return CB()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="40 x 2000 steps ~ 6 passes over COCO")
    ap.add_argument("--val-samples", type=int, default=VAL_SAMPLES)
    args = ap.parse_args()

    ensure_dirs(CKPT_DIR, FINAL_DIR)
    meta = read_json(p(PROC_DIR, "meta.json"))
    vocab_size, L = meta["vocab_size"], meta["max_length"]
    word_index = read_json(p(PROC_DIR, "word_index.json"))
    unk_id = word_index[UNK_TOKEN]

    paths = {k: p(CKPT_DIR, v) for k, v in
             {"last": "last.keras", "best": "best.keras", "state": "state.json", "history": "history.json"}.items()}
    state = {"epoch": 0, "best_val": float("inf"), "best_epoch": 0, "wait": 0, "lr": 1e-3, "done": False}
    history = []
    resuming = os.path.exists(paths["state"]) and os.path.exists(paths["last"])
    if resuming:
        state = read_json(paths["state"])
        history = read_json(paths["history"]) if os.path.exists(paths["history"]) else []
        if state.get("done") or state["epoch"] >= args.max_epochs:
            log(f"[step3] Training already finished (best val_loss {state['best_val']:.4f} at mini-epoch "
                f"{state['best_epoch']}, {state['epoch']} mini-epochs done). Go to step 4.\n"
                f"        (To train longer, run again with a larger --max-epochs.)")
            return
        log(f"[step3] RESUMING from mini-epoch {state['epoch']} "
            f"(best val_loss {state['best_val']:.4f}, lr {state['lr']:.2e})")
    elif os.path.exists(paths["state"]):
        log("[step3] state.json found but last.keras missing - starting from scratch.")

    import tensorflow as tf, keras
    if not tf.config.list_physical_devices("GPU"):
        log("[WARNING] No GPU found! Training will be extremely slow. Runtime > Change runtime type > T4 GPU.")

    t0 = time.time()
    log("[step3] Loading features and building training samples (1-3 minutes)...")
    feats, X, y, R = load_split("train", word_index, unk_id, L)
    vfeats, vX, vy, vR = load_split("val", word_index, unk_id, L)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(vy), size=min(args.val_samples, len(vy)), replace=False)
    val_data = ({"image_input": vfeats[vR[sel]].astype(np.float32), "seq_input": vX[sel].astype(np.int32)},
                vy[sel].astype(np.int32))
    del vX, vy, vR
    log(f"[data] ready in {fmt_time(time.time() - t0)}. steps per pass over the data: {len(y) // args.batch:,}")

    if resuming:
        model = keras.models.load_model(paths["last"])
        log(f"[step3] model + optimizer state loaded from {paths['last']}")
    else:
        from model import build_model
        model = build_model(vocab_size, L)
    log(f"[step3] model has {model.count_params():,} parameters")

    train_ds = make_train_dataset(feats, X, y, R, args.batch)
    cb = ResumableCheckpoint.create(paths, state, history)
    model.fit(train_ds, steps_per_epoch=args.steps_per_epoch,
              epochs=args.max_epochs, initial_epoch=state["epoch"],
              validation_data=val_data, validation_batch_size=1024,
              callbacks=[cb], verbose=2)

    if state.get("done") or state["epoch"] >= args.max_epochs:
        log(f"[step3] DONE. Best val_loss {state['best_val']:.4f} at mini-epoch {state['best_epoch']}. "
            "Go to step 4.")
    else:
        log("[step3] Stopped before finishing. Run this step again to continue.")


if __name__ == "__main__":
    main()
