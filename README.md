# ChaosNarrator

ChaosNarrator is a small multimodal transformer for image-conditioned story generation. It uses CLIP image features and a GPT-2 tokenizer to train a GPT-style decoder that generates short stories conditioned on images and optional keywords.

Quick start
-----------

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare datasets (Hugging Face datasets):
- For image-conditioned training we use Flickr30k via the Hugging Face Hub. Run `download_flickr.py` to load `nlphuji/flickr30k` and extract CLIP features into `data/clip_features/train_features.pt` (this can be large).
- For text-only training we use the WritingPrompts dataset from the Hugging Face Hub. Run `download_wp.py` to load `euclaise/writingprompts` and save `data/larger_model/writingprompts/wp_train.pt`.

4. Split features (if needed):

```bash
python3 split_data.py
```

5. Train stage 1 (image-conditioned):

```bash
python3 train.py
```

6. Train stage 2 (text-only fine-tuning):

```bash
python3 train2.py
```

Files & folders
---------------

- `transformer.py` — model implementation (multimodal embeddings, transformer blocks, generation).
- `dataset.py` / `dataset_wp.py` — PyTorch Datasets for VIST and WritingPrompts.
- `download_flickr.py`, `download_wp.py` — scripts that download and preprocess datasets.
- `split_data.py` — splits CLIP features into train/val/test.
- `train.py`, `train2.py` — training scripts for stage 1 and stage 2.
- `data/` — default dataset location used by scripts.
- `requirements.txt` — detected project dependencies.

Notes
----------------
- If dataset files are missing, training scripts may create minimal placeholder `.pt` files to avoid crashing, but training meaningful models requires real datasets.
- `transformers` and `torch` versions can affect behavior; pin versions in `requirements.txt` if reproducibility is required.
