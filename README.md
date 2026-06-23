# ASL Sign Recognition

Landmark-based recognition of isolated **ASL signs**, built on **MediaPipe Holistic** (feature extraction) and **PyTorch** (classification).

This repo implements **Phase 1** of the project roadmap. The MVP domain is **everyday conversation signs** (greetings, question words, common verbs, feelings) — high-frequency vocabulary that is well represented in [WLASL](https://github.com/dxli94/WLASL), so pretraining transfers cleanly. The medical vocabulary is **paused** for a later phase (see `configs/medical_glosses.txt`). The demo identifies the chosen ~50–100 signs from short video clips and is designed to be fine-tuned on signers you recruit yourself after pre-training on WLASL.

> Approach (per roadmap): extract hand + face + pose landmarks with MediaPipe (compact, fast, robust to lighting/skin tone) → train an LSTM or Transformer sequence classifier. No raw-video CNN, no avatar, no audio pipeline at this stage.

## Pipeline

```
video clips ──► MediaPipe Holistic ──► cached landmark .npy ──► PyTorch Dataset ──► LSTM / Transformer ──► sign label + confidence
   (mp4)          (extract.py)          (data/landmarks/)        (dataset.py)        (models/)            (softmax, gated < 0.8)
```

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Pull training clips for your vocab from WLASL.
#    Get WLASL_v0.3.json from github.com/dxli94/WLASL (start_kit/).
#    Preview matches first, then download into data/raw/<gloss>/.
python scripts/fetch_wlasl.py --metadata WLASL_v0.3.json \
       --vocab configs/daily_glosses.txt --dry-run
python scripts/fetch_wlasl.py --metadata WLASL_v0.3.json \
       --vocab configs/daily_glosses.txt --out data/raw --max-per-gloss 20

# 3. Extract landmarks from raw videos
#    Expects data/raw/<gloss>/<clip>.mp4
python -m src.extract --input data/raw --output data/landmarks

# 4. Train
python -m src.train --config configs/default.yaml

# 5. Run inference on a clip or webcam
python -m src.infer --checkpoint checkpoints/best.pt --source webcam
python -m src.infer --checkpoint checkpoints/best.pt --source path/to/clip.mp4
```

## Data layout

```
data/
  raw/                 # input videos, one folder per gloss (sign label)
    HELLO/clip001.mp4
    THANK-YOU/clip002.mp4
  landmarks/           # cached (T, 1629) landmark arrays (.npy), mirrors raw/
  splits/              # train/val/test gloss lists (signer-disjoint!)
```

**Important:** build your val/test split from signers the model was **not** trained on. Real WER on unseen signers is the number that matters.

## Datasets

Three helper scripts populate the data layout from public datasets. Each supports `--dry-run` to preview vocab overlap before downloading or writing anything.

| Script | Dataset | What you get |
| --- | --- | --- |
| `scripts/fetch_wlasl.py` | [WLASL](https://github.com/dxli94/WLASL) | Downloads + trims clips into `data/raw/`. Large vocab, but many dead source links. |
| `scripts/organize_asl_citizen.py` | [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) | Files clips into `data/raw/` and emits a **signer-disjoint** `data/splits/`. Clean, consented, no dead links — recommended primary source. |
| `scripts/adapt_google_islr.py` | [Google/PopSign ISLR](https://www.kaggle.com/competitions/asl-signs) | Converts pre-extracted MediaPipe landmarks straight to `data/landmarks/` — **no video, no extraction step**. Fast vocab boost. |

```bash
# Recommended: organize ASL Citizen (gives you a signer-disjoint split for free)
python scripts/organize_asl_citizen.py --src /path/to/ASL_Citizen \
       --vocab configs/daily_glosses.txt --out data/raw --splits-out data/splits

# Bonus: add Google/PopSign landmarks directly (skips src.extract for these)
python scripts/adapt_google_islr.py --data-dir /path/to/asl-signs \
       --vocab configs/daily_glosses.txt --out data/landmarks
```

## Project structure

```
configs/default.yaml      # all hyperparameters & paths
src/
  extract.py              # MediaPipe Holistic -> cached landmarks
  landmarks.py            # landmark spec, flattening, normalization
  dataset.py              # PyTorch Dataset + collate (padding)
  models/
    gru.py                # BiGRU classifier (default — best at small data)
    lstm.py               # BiLSTM classifier
    transformer.py        # Transformer-encoder classifier (scales with data)
    __init__.py           # build_model() factory
  train.py                # training loop, checkpointing, metrics
  infer.py                # clip + webcam inference with confidence gate
  utils.py                # seeding, config loading, logging
scripts/
  fetch_wlasl.py          # download + trim WLASL clips
  organize_asl_citizen.py # file ASL Citizen into data/raw + signer splits
  adapt_google_islr.py    # Google/PopSign parquet landmarks -> data/landmarks
tests/                    # smoke tests
```

## Roadmap context

This is Phase 1 of a longer plan (confidence gating + interpreter-in-the-loop → continuous signing → text→avatar). See the project roadmap doc. Confidence scores are emitted on every prediction so the later gating/clarification layer can be built on top without rework.

## Licensing & ethics

- WLASL / research datasets: fine for training models used in a pilot, but **do not redistribute raw videos**.
- Keep any clinic footage on encrypted local drives; delete raw video after landmark extraction unless the subject opts in to longer storage.
