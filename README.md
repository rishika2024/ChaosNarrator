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
- `genrate_story.py` - generate the stories from the image and keywords
- `save_test_img.py` - save 10 test images
- `requirements.txt` — detected project dependencies.


-Architecture
-------------

High-level data and model flow (arrow diagram):

Image files  →  CLIP Vision Encoder  →  CLIP feature vector (per image)
	↓
	(stack/collect multiple images)
	↓
CLIP feature vectors  →  Linear projection  →  Image tokens
	↓
Image tokens  ⟶  Concatenate  ⟶  [Image tokens + Text token embeddings]
	↓
Add positional embeddings
	↓
Transformer (stack of pre-norm blocks):
	LayerNorm → Causal Multi-Head Self-Attention → Residual
	LayerNorm → Feed-Forward (4× expand, GELU) → Residual
	↓
LM head (linear projection to vocab)  →  Softmax / Sampling

Stage 1 — Image-conditioned pretraining
--------------------------------------

- Purpose: teach the model to condition generation on visual context extracted by CLIP.
- Input: CLIP features saved in `data/clip_features/*.pt` (produced by `download_flickr.py`).
- Script: `train.py`.
- What trains: image projection layer + transformer + LM head (all or configurable).
- Loss: cross-entropy on text-token positions only; image-token positions are ignored.
- Checkpoints: saved to `larger_model_checkpoints/` (e.g. `best_model.pt`).

Stage 2 — Text-only fine-tuning (scale language behavior)
-------------------------------------------------------

- Purpose: improve narrative fluency and diversity using large text while retaining visual grounding from Stage 1.
- Input: WritingPrompts data saved to `data/larger_model/writingprompts/wp_train.pt` (produced by `download_wp.py`).
- Script: `train2.py`.
- Behavior: loads Stage 1 checkpoint (`larger_model_checkpoints/best_model.pt`) and continues training on text-only data. You can optionally freeze the image projection layer to preserve image grounding (there's commented code in `train2.py` to do this).
- Checkpoints: saved to `larger_model_checkpoints_stage2/`.

---------------------------------------------------------------

Outputs: 

Stage-1 (train.py)
- set 5 Stories with keyword input and Generated text

Stage-2 (train2.py)
- set 5 Creative Stories with keyword input and Generated text

If you want to view the corresponding image:
run `save_test_img.py`
test_image0 -> corresponds to story1
test_image1 -> corresponds to story2
and so on
---------------------------------------------------------------------

### To generate output stories

- Download the clip features, test stories and the best model from:
https://drive.google.com/drive/folders/1Fst4WQp71tv8zd4fgPAm3iZWdQ8ZPKuk?usp=drive_link

- run `generate_story.py`






