"""
step1_prepare.py  -  STEP 1 of 5  (CPU only, about 5-10 minutes)

What it does
  1. Downloads the COCO 2017 caption annotations (~250 MB) into your Drive.
  2. Cleans every caption with the SAME rules as the Flickr8k project.
  3. Replaces rare words (seen < 5 times) by 'unk', fits the tokenizer.
  4. Writes the list of images to download in step 2 (manifests).
  5. Prints how often key words (tiger, chess, ...) appear, so you can see
     what COCO really contains BEFORE spending hours on training.

Safe to re-run: if everything is already done it just says so.
Use --force to redo it.
"""
import argparse, collections, io, os, pickle, sys, time, urllib.request, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_config import *
from coco_common import *

REPORT_WORDS = ["tiger", "lion", "chess", "laptop", "cat", "dog", "giraffe", "zebra",
                "pizza", "kitchen", "soccer", "selfie", "bicycle", "bus", "man"]
FLICKR_COUNTS = {"tiger": 21, "chess": 5, "lion": 5, "laptop": 12, "cat": 65, "dog": 8136, "man": 7266}


# ---------------------------------------------------------------- download
def download_annotations():
    ensure_dirs(ANN_DIR)
    targets = {s: os.path.join(ANN_DIR, os.path.basename(m)) for s, m in ANN_MEMBERS.items()}
    if all(os.path.exists(t) and os.path.getsize(t) > 0 for t in targets.values()):
        log("[download] annotation files already in Drive - skipping download.")
        return targets

    tmp_zip = "/tmp/annotations_trainval2017.zip"
    last_err = None
    for url in ANN_URLS:
        for attempt in range(3):
            try:
                log(f"[download] {url} (attempt {attempt + 1}/3)")
                with urllib.request.urlopen(url, timeout=60) as r, open(tmp_zip, "wb") as out:
                    total = int(r.headers.get("Content-Length", 0))
                    done, next_mark = 0, 10
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        out.write(buf)
                        done += len(buf)
                        if total and done * 100 // total >= next_mark:
                            log(f"           {done * 100 // total}%  ({done >> 20} MB)")
                            next_mark += 10
                if not zipfile.is_zipfile(tmp_zip):
                    raise IOError("downloaded file is not a valid zip")
                last_err = None
                break
            except Exception as e:
                last_err = e
                log(f"[download] failed: {e}")
                time.sleep(3 * (attempt + 1))
        if last_err is None:
            break
    if last_err is not None:
        raise SystemExit(f"[ERROR] Could not download annotations: {last_err}\n"
                         "        Download annotations_trainval2017.zip manually from cocodataset.org,\n"
                         f"        and put captions_train2017.json and captions_val2017.json in {ANN_DIR}")

    with zipfile.ZipFile(tmp_zip) as z:
        for split, member in ANN_MEMBERS.items():
            data = z.read(member)
            tmp = targets[split] + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, targets[split])
            log(f"[download] saved {targets[split]}")
    os.remove(tmp_zip)
    return targets


# ---------------------------------------------------------------- parsing
def load_split(path):
    """Return (manifest, clean_captions) for one split. clean_captions: {id: [str,...]}."""
    d = read_json(path)
    images = {im["id"]: im for im in d["images"]}
    raw = collections.defaultdict(list)
    for a in d["annotations"]:
        raw[a["image_id"]].append(a["caption"])

    manifest, caps = [], {}
    dropped_long = dropped_empty = 0
    for img_id in sorted(raw):
        if img_id not in images:
            continue
        kept = []
        for c in raw[img_id]:
            cc = clean_caption(c)
            n = len(cc.split())
            if n == 0:
                dropped_empty += 1
            elif n > MAX_CAPTION_WORDS:
                dropped_long += 1
            else:
                kept.append(cc)
        kept = kept[:MAX_CAPTIONS_PER_IMG]
        if not kept:
            continue
        im = images[img_id]
        url = im.get("coco_url") or f"http://images.cocodataset.org/{im['file_name']}"
        manifest.append({"id": img_id, "url": url, "file": im["file_name"]})
        caps[img_id] = kept
    log(f"[parse] {os.path.basename(path)}: {len(manifest):,} images, "
        f"{sum(len(v) for v in caps.values()):,} captions "
        f"(dropped {dropped_long} too long, {dropped_empty} empty)")
    return manifest, caps


def wrap(cap, vocab):
    toks = [w if w in vocab else UNK_TOKEN for w in cap.split()]
    return f"{START_TOKEN} {' '.join(toks)} {END_TOKEN}"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="redo even if outputs exist")
    args = ap.parse_args()

    ensure_dirs(BASE, ANN_DIR, PROC_DIR, FEAT_DIR, CKPT_DIR, FINAL_DIR)
    meta_path = p(PROC_DIR, "meta.json")
    if os.path.exists(meta_path) and not args.force:
        meta = read_json(meta_path)
        log("[step1] Already done. Nothing to do.")
        log(f"         vocab_size={meta['vocab_size']}  max_length={meta['max_length']}  "
            f"train images={meta['n_train_images']:,}  val images={meta['n_val_images']:,}")
        log("         (use --force to redo)")
        return

    t0 = time.time()
    paths = download_annotations()

    train_manifest, train_caps = load_split(paths["train"])
    val_manifest,   val_caps   = load_split(paths["val"])

    # ---- vocabulary from TRAIN captions only ----
    freq = collections.Counter(w for caps in train_caps.values() for c in caps for w in c.split())
    vocab = {w for w, n in freq.items() if n >= MIN_WORD_COUNT}
    log(f"[vocab] {len(freq):,} unique words in train -> keeping {len(vocab):,} "
        f"(seen >= {MIN_WORD_COUNT} times); the rest become '{UNK_TOKEN}'")

    train_wrapped = {str(i): [wrap(c, vocab) for c in cs] for i, cs in train_caps.items()}
    val_wrapped   = {str(i): [wrap(c, vocab) for c in cs] for i, cs in val_caps.items()}
    val_clean     = {str(i): cs for i, cs in val_caps.items()}     # untouched references for BLEU

    # ---- tokenizer (same settings as Flickr8k version) ----
    from tensorflow.keras.preprocessing.text import Tokenizer
    all_caps = [c for cs in train_wrapped.values() for c in cs]
    tok = Tokenizer(oov_token=None, filters='')
    tok.fit_on_texts(all_caps + [UNK_TOKEN])      # guarantees 'unk' has an index, even if no word was rare
    vocab_size = len(tok.word_index) + 1
    max_length = max(len(c.split()) for c in all_caps)
    if vocab_size >= 32000:
        raise SystemExit("vocab too large for int16 storage - raise MIN_WORD_COUNT")
    log(f"[tokenizer] vocab_size={vocab_size}  max_length={max_length}")

    with open(p(PROC_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tok, f)
    write_json_atomic(tok.word_index, p(PROC_DIR, "word_index.json"))
    write_json_atomic(train_wrapped, p(PROC_DIR, "train_captions.json"))
    write_json_atomic(val_wrapped,   p(PROC_DIR, "val_captions.json"))
    write_json_atomic(val_clean,     p(PROC_DIR, "val_captions_clean.json"))
    write_json_atomic(train_manifest, p(PROC_DIR, "manifest_train.json"))
    write_json_atomic(val_manifest,   p(PROC_DIR, "manifest_val.json"))

    # ---- report ----
    lines = ["Word counts in COCO train captions (Flickr8k counts in brackets):", ""]
    for w in REPORT_WORDS:
        fl = f"   [Flickr8k: {FLICKR_COUNTS[w]}]" if w in FLICKR_COUNTS else ""
        lines.append(f"  {w:10s} {freq.get(w, 0):>8,}   in vocab: {str(w in vocab):5s}{fl}")
    report = "\n".join(lines)
    with open(p(PROC_DIR, "vocab_report.txt"), "w") as f:
        f.write(report + "\n")
    log("\n" + report + "\n")

    # meta.json is written LAST: its presence means step 1 finished completely
    write_json_atomic({
        "vocab_size": vocab_size, "max_length": max_length,
        "n_train_images": len(train_manifest), "n_val_images": len(val_manifest),
        "n_train_captions": len(all_caps),
        "min_word_count": MIN_WORD_COUNT, "max_caption_words": MAX_CAPTION_WORDS,
    }, meta_path)
    log(f"[step1] DONE in {fmt_time(time.time() - t0)}. Files are in {PROC_DIR}")


if __name__ == "__main__":
    main()
