# Project History

This document records what was built, what went wrong, how each problem was found, and what was decided.
It is written in the order things happened.

## Summary table

| Version | Encoder | Data | Key settings | Result |
|---|---|---|---|---|
| **v1** | VGG16, average pooling (512 numbers) | Flickr8k | Dropout 0.5 on the raw image vector, vocabulary 8,769, fixed image order in batches, greedy decoding | About 38% next-word accuracy, **same caption for every image** |
| **v2** | VGG16 `fc2` layer (4,096 numbers, L2-normalised) | Flickr8k | Shuffled mixed-image batches, dropout 0.3 after the Dense layer, vocabulary 2,986, sparse cross-entropy, beam search | BLEU-1 0.536, BLEU-4 0.166 (100 validation images) |
| **v3** | InceptionV3 (2,048 numbers) | Flickr8k | Rescaling + LayerNormalization on both branches, `unk` blocked, beam width 5 in the app | BLEU-1 0.562, BLEU-4 0.162 (500 validation images) |
| **COCO** | InceptionV3 (2,048 numbers) | MS-COCO 2017, 118,287 images | Same decoder, vocabulary 10,183, batch 512, resumable training | BLEU-1 0.658, BLEU-4 0.254 (1,000 validation images) |

---

## Stage 1: first model and the "same caption for every image" problem

**What was built.** VGG16 (ImageNet weights, no classification head, global average pooling) turned each Flickr8k
photo into 512 numbers. An LSTM decoder produced captions word by word.

**What went wrong.** After training, per-word accuracy was about 38% and the model produced the **same caption for
every image**.

**How it was diagnosed.** 38% is normal for Flickr8k, so the accuracy was not the bug. A model that ignores the
image and only learns English caption patterns reaches about that level. Reading the code showed several causes:

1. The data generator went through the images in a fixed order. One image yields about 50 training samples, so
   each batch of 32 contained samples from only one or two images. Training was dominated by the language pattern.
2. The 512-number average-pooled features carried little detail and were not normalised.
3. Dropout 0.5 was applied directly on the raw image vector, throwing away half of an already weak signal.
4. The vocabulary was unfiltered (8,769 words, many seen only once or twice).
5. Decoding was greedy, which pushes every image toward the same "safe" sentence.

## Stage 2: first round of fixes (v2)

| Change | Reason |
|---|---|
| VGG16 `fc2` features (4,096 numbers), L2-normalised | Much richer image description |
| Samples from about 100 images mixed before each batch, reshuffled every pass | Every batch now sees many images |
| Dropout 0.3 placed after the Dense layer | Keeps the image signal |
| Words seen fewer than 5 times replaced by `unk` (vocabulary 8,769 down to 2,986) | Less noise, smaller output layer |
| Sparse categorical cross-entropy with integer targets | No giant one-hot arrays, larger batches fit |
| Beam search and a repetition guard | Less generic captions |

**Result.** Captions now differed between images. A diagnostic script (`src/diagnose.py`) confirmed that
features differed between images (average cosine similarity 0.238) and that captions described the right
subject in most validation examples. BLEU-1 0.536 and BLEU-4 0.166 on 100 validation images.

**Remaining failure.** A photo of a child kicking a football was captioned *"a group of children are playing in
the grass with a ball in its mouth"*. The training data contains hundreds of "dog with a ball in its mouth"
captions, so the decoder completed the phrase out of habit.

## Stage 3: better encoder (v3)

| Change | Reason |
|---|---|
| InceptionV3 (2,048 numbers) instead of VGG16 | Stronger, lighter image features |
| `Rescaling(sqrt(2048))` and `LayerNormalization` on both branches | L2-normalised features were tiny (about 0.02 each), so the image branch was far weaker than the LSTM branch |
| Block the `unk` token during decoding | `unk` should never appear in a caption |
| One shared `src/encoder.py` | Training and prediction cannot disagree about image size or preprocessing |

**Result.** BLEU on the same 100 validation images was BLEU-1 0.566, BLEU-4 0.125. Re-measured on 500 images it
was BLEU-1 0.562, BLEU-2 0.375, BLEU-3 0.248, BLEU-4 0.162. The BLEU-4 of 0.125 on 100 images was a small-sample
effect: differences of a few hundredths on 100 images are noise, which is why the 500-image run was used.

**Remaining failure.** Photos unlike Flickr8k still failed: a tiger was captioned as a dog and a chess game as
"a man and a woman smile for the camera".

## Stage 4: why did odd photos fail?

The first hypothesis was that the words "tiger" and "chess" were missing from the vocabulary. **That was wrong.**
Checking the tokenizer showed both words were in the vocabulary. Counting how often they appeared explained it:

| Word | Times in Flickr8k training captions |
|---|---|
| dog | 8,136 |
| man | 7,266 |
| soccer | 580 |
| cat | 65 |
| tiger | 21 |
| laptop | 12 |
| chess | 5 |
| lion | 5 |

Reading the actual captions containing "tiger" showed that the 21 mentions appeared in 20 captions from 10 images,
and at most 2 of those images showed a real tiger. The rest were tiger-striped dogs and toy or inflatable tigers. All 5 "chess" captions
came from a single photo (a crowd watching a game with giant pieces in a park).

**Conclusion.** The model cannot learn a concept from one or two photos. This is the **long-tail problem**: a word
being in the vocabulary only means it appeared at least 5 times, and it still gets almost no weight against
words that appear thousands of times. No change to the model design fixes this. Only more data, or knowledge
from outside, can.

## Stage 5: MS-COCO

**Decision.** Train on the full MS-COCO 2017 training set: 118,287 images and 591,022 captions, about 14 times
more images than Flickr8k and far more varied (indoor scenes, food, vehicles, many animals).

**Design for safety.** Training ran on Google Colab, which can disconnect at any time, so every step was built
to resume:

- Features were extracted in chunks of 4,000 images. Each finished chunk is saved to Google Drive; a re-run skips
  finished chunks. Images are downloaded one chunk at a time and never stored (no 18 GB download).
- A chunk file is written under a temporary name and renamed when complete, so a disconnect cannot leave a
  half-written file counted as finished.
- Training is cut into "mini-epochs" of 2,000 steps. After each one the full model, optimizer state and
  progress counters are saved to Drive, and a re-run continues from there.
- Text cleaning rules are identical to the Flickr8k pipeline. Words seen fewer than 5 times became `unk`
  (vocabulary 10,183). Captions longer than 30 words were dropped.

**A real bug found by testing.** Before the real run, the pipeline was tested on a small fake COCO-style dataset,
including killing the training process mid-run. This revealed that in the Keras version used, saving a legacy
`.h5` file silently deletes the optimizer settings from every later `.keras` save. After any resume, training
would have silently switched from Adam to RMSprop and discarded Adam's state. The fix: write only `.keras`
files during training, and convert the best model to `.h5` once at the end (Step 5).

**What was tested and what was not.** The five steps were tested on fake data, including a simulated disconnect,
a half-written chunk, failed image downloads, a hard kill of training, and loading the exported model through
the app's own code. The real COCO download and the real ImageNet weights could only be exercised in the actual
Colab run.

**Result.** The run completed without errors.

| Metric | Value |
|---|---|
| BLEU-1 / 2 / 3 / 4 (1,000 val2017 images, beam 3) | 0.658 / 0.482 / 0.349 / 0.254 |
| Final validation loss / accuracy | about 2.38 / about 50% |
| Training length | 40 mini-epochs x 2,000 steps, batch 512 |

Training and validation curves stayed close together (no overfitting), and validation loss was still falling
slowly when the 40 mini-epoch limit was reached.

## Stage 6: testing the COCO model in the app

| Photo | Caption | Verdict |
|---|---|---|
| A child kicking a football | "a soccer player kicking a soccer ball on a field" | Correct (the Flickr8k-era model said "ball in its mouth") |
| Older men playing chess | "a group of people sitting around a table with a cake" | People at a table is right, "cake" is false |
| A single tiger | "a couple of zebra standing next to each other on a lush green field" | False: wrong animal and number |
| A close-up photo of a face | "a man holding a doughnut up to his face" | False |

**Reading.** The COCO model is clearly better on everyday scenes, but it still states false objects with
confidence on photos unlike its training data. To my knowledge COCO has no tiger category, which would explain
"zebra". The doughnut caption is likely (untested) a result of close-up faces in COCO captions often involving
food.

## Decision: finish here

Options that were considered and not done:

| Option | Why not done |
|---|---|
| Train longer | Probably a small gain only; would not fix tigers or face close-ups |
| Compare against a pre-trained captioner (BLIP) | Different model, not part of the original scope; listed as future work |
| Confidence warning in the app | Cheap to add, but a wrong caption can also be confident; listed as future work |

## Lessons learned

1. **Accuracy per word is not caption quality.** 38% looked like a failure but was normal. The real signal was
   that every caption was identical.
2. **Test with deliberately unlike photos.** BLEU on in-distribution images looked fine while tigers, chess and
   face close-ups failed.
3. **Check the data before changing the model.** The tiger failure was fully explained by reading 21 captions.
4. **State the cause only after checking it.** The first explanation (missing vocabulary) was wrong and was
   corrected by one command.
5. **Long jobs need resumable design and tests that simulate failure.** The optimizer bug was found only because
   the test killed the process mid-training.
6. **Small test sets are noisy.** A BLEU-4 of 0.125 on 100 images became 0.162 on 500.
