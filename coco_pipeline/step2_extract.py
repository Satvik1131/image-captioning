"""
step2_extract.py  -  STEP 2 of 5  (needs the GPU, run once for val and once for train)

What it does
  Downloads COCO images in small groups ("chunks" of 4000), turns each image into
  2048 numbers with InceptionV3, and saves every finished chunk to your Drive.
  The images themselves are NOT kept (no 18 GB download, no big disk needed).

If Colab disconnects
  Reconnect, run the same cell again. Finished chunks are skipped, so you lose
  at most the one chunk that was being processed (a few minutes).

Usage
  python step2_extract.py --split val       (5,000 images - do this first, it is a speed test)
  python step2_extract.py --split train     (118,000 images - the long one)
"""
import argparse, concurrent.futures as cf, io, os, sys, time, urllib.request
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_config import *
from coco_common import *
from encoder import build_encoder, preprocess, IMG_SIZE, FEATURE_DIM

RETRIES = 3
HEADERS = {"User-Agent": "Mozilla/5.0 (coco-captioning-student-project)"}


def fetch_and_resize(item):
    """Download one image and resize it EXACTLY like predict.py does (PIL, RGB, 299x299).
    Returns a uint8 array, or None if all retries failed."""
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(item["url"], headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            img = Image.open(io.BytesIO(data)).convert("RGB").resize(IMG_SIZE)
            return np.asarray(img, dtype=np.uint8)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def chunk_is_valid(path):
    """A chunk counts as finished only if it opens cleanly and shapes agree."""
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as z:
            ids, feats = z["ids"], z["feats"]
            return feats.ndim == 2 and feats.shape[0] == ids.shape[0] and feats.shape[1] == FEATURE_DIM
    except Exception:
        return False


def encode_batch(model, arrs):
    x = preprocess(np.stack(arrs).astype("float32"))
    f = model.predict(x, batch_size=len(arrs), verbose=0)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)    # L2-normalise, same as before
    return f.astype("float16")


def process_chunk(model, items, workers, batch):
    ids_ok, feats, failed = [], [], []
    ex = cf.ThreadPoolExecutor(max_workers=workers)
    try:
        results = ex.map(fetch_and_resize, items)        # downloads run ahead of the GPU
        buf, buf_ids = [], []
        for item, arr in zip(items, results):
            if arr is None:
                failed.append(item["id"])
                continue
            buf.append(arr); buf_ids.append(item["id"])
            if len(buf) == batch:
                feats.append(encode_batch(model, buf)); ids_ok += buf_ids
                buf, buf_ids = [], []
        if buf:
            feats.append(encode_batch(model, buf)); ids_ok += buf_ids
    finally:
        ex.shutdown(wait=False, cancel_futures=True)     # stop quickly on Ctrl-C / disconnect
    feats = np.concatenate(feats) if feats else np.zeros((0, FEATURE_DIM), "float16")
    return np.array(ids_ok, dtype=np.int64), feats, np.array(failed, dtype=np.int64)


def merge(split, chunk_paths):
    out_npy = p(FEAT_DIR, f"features_{split}.npy")
    out_ids = p(FEAT_DIR, f"ids_{split}.json")
    ids_all, feats_all = [], []
    for cp in chunk_paths:
        with np.load(cp) as z:
            ids_all.append(z["ids"]); feats_all.append(z["feats"])
    ids = np.concatenate(ids_all); feats = np.concatenate(feats_all)
    tmp = out_npy + ".tmp.npy"
    np.save(tmp, feats); os.replace(tmp, out_npy)
    write_json_atomic([int(i) for i in ids], out_ids)
    log(f"[merge] wrote {out_npy}  shape={feats.shape}  ({feats.nbytes / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--workers", type=int, default=32, help="parallel downloads")
    ap.add_argument("--batch", type=int, default=128, help="GPU batch size")
    ap.add_argument("--limit", type=int, default=0, help="only first N images (testing)")
    args = ap.parse_args()

    manifest_path = p(PROC_DIR, f"manifest_{args.split}.json")
    if not os.path.exists(manifest_path):
        raise SystemExit("[ERROR] Run step 1 first (manifest not found).")
    manifest = read_json(manifest_path)
    if args.limit:
        manifest = manifest[:args.limit]

    out_dir = p(FEAT_DIR, args.split)
    ensure_dirs(out_dir)
    chunks = [manifest[i:i + args.chunk_size] for i in range(0, len(manifest), args.chunk_size)]
    paths = [p(out_dir, f"chunk_{i:04d}.npz") for i in range(len(chunks))]
    done = [chunk_is_valid(cp) for cp in paths]
    log(f"[step2] split={args.split}: {len(manifest):,} images in {len(chunks)} chunks; "
        f"{sum(done)} already finished, {len(chunks) - sum(done)} to do.")

    if all(done):
        if not os.path.exists(p(FEAT_DIR, f"features_{args.split}.npy")):
            merge(args.split, paths)
        log("[step2] Everything already extracted. Nothing to do.")
        return

    import tensorflow as tf
    if not tf.config.list_physical_devices("GPU"):
        log("[WARNING] No GPU found! Runtime > Change runtime type > T4 GPU. "
            "Extraction will be very slow on CPU.")
    log("[step2] Loading InceptionV3 (first time downloads ~90 MB of ImageNet weights)...")
    model = build_encoder()

    t_start, n_new, imgs_new = time.time(), 0, 0
    for ci, (items, cp) in enumerate(zip(chunks, paths)):
        if done[ci]:
            continue
        t0 = time.time()
        ids, feats, failed = process_chunk(model, items, args.workers, args.batch)
        tmp = cp[:-4] + ".tmp.npz"
        np.savez(tmp, ids=ids, feats=feats, failed=failed)
        os.replace(tmp, cp)                                 # chunk exists only if fully written
        n_new += 1; imgs_new += len(items)
        el = time.time() - t0
        remaining = len(chunks) - sum(done) - n_new
        eta = (time.time() - t_start) / n_new * remaining
        log(f"[step2] chunk {ci + 1}/{len(chunks)} saved: {len(ids)} ok, {len(failed)} failed | "
            f"{len(items) / el:.1f} img/s | chunks left {remaining} | ETA {fmt_time(eta)}")

    merge(args.split, paths)
    total_failed = 0
    for cp in paths:
        with np.load(cp) as z:
            total_failed += len(z["failed"])
    log(f"[step2] DONE ({args.split}). Failed images in total: {total_failed} (they are simply skipped).")
    if total_failed > 0.01 * len(manifest):
        log("[WARNING] More than 1% of the downloads failed. The server may be rate-limiting you.\n"
            "          The skipped images are just not used for training, which is fine for a few hundred.\n"
            "          If you want fewer failures next time, run with fewer parallel downloads: --workers 12")


if __name__ == "__main__":
    main()
