# COCO Pipeline (resumable, runs on Google Colab)

Trains the caption model on the full MS-COCO 2017 training set (118,287 images, 591,022 captions).
Built so that **a Colab disconnect never loses more than a few minutes of work**: every step saves its
results to Google Drive, and re-running a step continues where it stopped.

## Files

| File | Purpose |
|---|---|
| `coco_colab.ipynb` | The notebook you run in Colab, one cell per step (kept next to this folder) |
| `coco_config.py` | All paths and settings in one place |
| `coco_common.py` | Shared helpers (caption cleaning, safe file writing) |
| `step1_prepare.py` | Download annotations, clean captions, build the vocabulary |
| `step2_extract.py` | Download images in chunks and extract InceptionV3 features |
| `step3_train.py` | Train the model, saving after every 2,000 steps |
| `step4_evaluate.py` | BLEU-1 to BLEU-4 on COCO val2017 images |
| `step5_export.py` | Pack the final files and check they load like the app loads them |
| `status.py` | Show what is finished (reads files only, safe to run any time) |
| `model.py`, `encoder.py` | Copies of `src/model.py` and `src/encoder.py` used by the steps |
| `rebuild_tokenizer.py` | Local backup: rebuild `tokenizer.pkl` from `word_index.json` if it will not load |
| `local_patch/predict.py` | Version of `src/predict.py` that reads `max_length` from `meta.json` |

## Folder layout on Google Drive

```
My Drive/coco_captioning/
├── coco_pipeline.zip    the only thing you create by hand (upload, do not unzip)
├── annotations/         step 1: raw COCO caption files
├── processed/           step 1: cleaned captions, tokenizer, image lists, vocab_report.txt
├── features/            step 2: image features, saved in chunks
├── checkpoints/         step 3: last.keras, best.keras, state.json, history.json
└── final/               step 5: the files you download
```

## How to run

1. Upload `coco_pipeline.zip` to `My Drive/coco_captioning/`.
2. Open `coco_colab.ipynb` in Colab and choose **Runtime > Change runtime type > T4 GPU**.
3. Run the cells **one at a time, top to bottom** (not "Run all"). Move on when the previous cell prints `DONE`.

| Cell | What it does |
|---|---|
| Cell 0 | Connects Drive and unpacks the code. Run it first in every session. |
| STEP 1 | Captions: clean, vocabulary, word-count report (CPU) |
| STEP 2a | Features for the 5,000 validation images (GPU, also a speed test) |
| STEP 2b | Features for the 118,000 training images (GPU, long) |
| STEP 3 | Training (GPU, long) |
| STEP 4 | BLEU evaluation |
| STEP 5 | Export the final files |

**If Colab disconnects:** reconnect, run Cell 0, then run the same step cell again.
Use the STATUS cell at any time to see progress.

To train longer after Step 3 has finished, run
`!python step3_train.py --max-epochs 80`. It continues from mini-epoch 40.

## After Step 5: use the model locally

1. Download from Drive (`coco_captioning/final/`) into the project's `models/` folder:
   `caption_model.h5`, `tokenizer.pkl`, `word_index.json`, `meta.json`
   (plus `training_curves.png` and `bleu_results.json` for your report).
2. Copy `local_patch/predict.py` over `src/predict.py`, so the app reads `max_length` from `meta.json`.
3. If `tokenizer.pkl` will not load (a version mismatch between Colab and your PC), run
   `python rebuild_tokenizer.py` from the project root.
4. `streamlit run app/app.py`

## Design decisions

- **Images are never stored.** Each chunk of 4,000 images is downloaded, converted to 2,048 numbers, and
  discarded. This avoids the 18 GB image archive and keeps the disk small.
- **Atomic file writes.** Chunks, JSON files and checkpoints are written under a temporary name and renamed when
  complete, so a disconnect cannot leave a half-written file counted as finished.
- **Resume state is explicit.** Step 3 saves the full model, the optimizer state, the learning rate, the best
  validation loss and the patience counters after every mini-epoch.
- **Only `.keras` files are written during training.** Writing a legacy `.h5` silently deletes the optimizer
  settings from later `.keras` saves in the Keras version used, which would break resume. Step 5 converts the
  best model to `.h5` once, at the end, and checks that the converted file reproduces the same predictions.
- **Same text and image rules as the Flickr8k pipeline:** identical caption cleaning, rare-word threshold of 5,
  InceptionV3 at 299 x 299, and the same resize call as `src/predict.py`.

## Settings (all in `coco_config.py` and the step scripts)

| Setting | Value |
|---|---|
| Rare-word threshold | words seen fewer than 5 times become `unk` |
| Caption length limit | captions over 30 words dropped, at most 5 captions per image |
| Chunk size | 4,000 images |
| Batch size / steps per mini-epoch | 512 / 2,000 |
| Maximum training | 40 mini-epochs |
| Learning rate | 1e-3, halved after 4 mini-epochs without improvement |
| Early stop | 10 mini-epochs without improvement |
| Validation during training | a fixed random sample of 30,000 validation examples |

## Testing

All five steps were tested on a small fake COCO-style dataset, including: a re-run that skips finished work,
a half-written feature chunk, images that fail to download, a hard kill during training followed by a resume,
learning-rate halving and the automatic stop, and loading the exported model through the app's own code.
The real COCO run was then executed in Colab.
