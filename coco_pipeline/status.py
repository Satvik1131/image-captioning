"""status.py - shows what is finished and what is not. Safe to run any time (reads files only)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_config import *

def exists(*a): return os.path.exists(p(*a))
def rj(*a): return json.load(open(p(*a)))
def count(d, pat):
    return len([f for f in os.listdir(d) if f.startswith(pat) and f.endswith(".npz") and ".tmp" not in f]) if os.path.isdir(d) else 0

print(f"Results folder: {BASE}\n")
print("STEP 1  prepare captions   :", "DONE" if exists(PROC_DIR, "meta.json") else "not done")
if exists(PROC_DIR, "meta.json"):
    m = rj(PROC_DIR, "meta.json")
    print(f"          vocab {m['vocab_size']:,} words | max length {m['max_length']} | "
          f"{m['n_train_images']:,} train images | {m['n_val_images']:,} val images")
    for split, n in (("val", m["n_val_images"]), ("train", m["n_train_images"])):
        total = -(-n // CHUNK_SIZE)
        got = count(p(FEAT_DIR, split), "chunk_")
        done = exists(FEAT_DIR, f"features_{split}.npy")
        print(f"STEP 2  features ({split:5s})    : {'DONE' if done else 'in progress/not done'}  ({got}/{total} chunks saved)")
print("STEP 3  training           :", end=" ")
if exists(CKPT_DIR, "state.json"):
    s = rj(CKPT_DIR, "state.json")
    print(("DONE" if (s.get("done") or s["epoch"] >= DEFAULT_MAX_EPOCHS) else "in progress") +
          f"  | mini-epoch {s['epoch']} | best val_loss {s['best_val']:.4f} (mini-epoch {s['best_epoch']}) | lr {s['lr']:.1e}")
else:
    print("not started")
print("STEP 4  evaluation         :", "DONE  " + json.dumps(rj(FINAL_DIR, "bleu_results.json")) if exists(FINAL_DIR, "bleu_results.json") else "not done")
print("STEP 5  export             :", "DONE" if exists(FINAL_DIR, "caption_model.h5") else "not done")
