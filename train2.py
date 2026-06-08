import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_wp import WritingPromptsDataset
from transformer import ChaosNarrator
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os

# Settings (must match Stage 1)
vocab_size = 50257
embed_dim = 256
num_heads = 4
num_layers = 4
max_len = 200
clip_embed_dim = 768
batch_size = 16
learning_rate = 1e-4  # lower than Stage 1 to preserve image grounding
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def fix_story_size(batch):
    clip_vectors = []
    input_tokens = []
    target_tokens = []

    for img, inp, tgt in batch:
        clip_vectors.append(img)
        input_tokens.append(inp)
        target_tokens.append(tgt)

    clip_vectors = torch.stack(clip_vectors)

    max_seq_len = 0
    for t in input_tokens:
        if t.size(0) > max_seq_len:
            max_seq_len = t.size(0)

    padded_inputs = []
    padded_targets = []
    eos_token = 50256
    ignore_label = -1

    for i in range(len(input_tokens)):
        inp = input_tokens[i]
        tgt = target_tokens[i]
        pad_len = max_seq_len - inp.size(0)
        padded_inp = F.pad(inp, (0, pad_len), value=eos_token)
        padded_inputs.append(padded_inp)
        padded_tgt = F.pad(tgt, (0, pad_len), value=ignore_label)
        padded_targets.append(padded_tgt)

    padded_inputs = torch.stack(padded_inputs)
    padded_targets = torch.stack(padded_targets)

    return clip_vectors, padded_inputs, padded_targets

