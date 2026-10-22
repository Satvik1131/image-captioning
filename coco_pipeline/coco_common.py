"""coco_common.py - small helpers shared by all steps."""
import json, os, re, string, time

def log(msg):
    print(msg, flush=True)

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def clean_caption(caption):
    """IDENTICAL rules to src/caption_preprocessing.py (so COCO and Flickr8k match)."""
    caption = caption.lower()
    caption = re.sub(r'\d+', '', caption)
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = re.sub(r'\s+', ' ', caption).strip()
    tokens = [t for t in caption.split() if len(t) > 1 or t in ('a',)]
    return ' '.join(tokens)

def write_json_atomic(obj, path):
    """Write to a temp file then rename, so a disconnect can never leave a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fmt_time(sec):
    sec = int(sec)
    return f"{sec//3600}h{(sec%3600)//60:02d}m{sec%60:02d}s"
