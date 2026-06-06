import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import VISTDataset
from transformer import ChaosNarrator
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os

# Settings
vocab_size = 50257
embed_dim = 256
num_heads = 4
num_layers = 4
max_len = 150
clip_embed_dim = 768
batch_size = 16
learning_rate = 3e-4
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# A helper function to fix story sizes in a batch by padding them to the same length
def fix_story_size(batch):
    # Unpacking the batch into separate lists
    image_feats = []
    input_tokens = []
    target_tokens = []

    for img, inp, tgt in batch:
        image_feats.append(img)
        input_tokens.append(inp)
        target_tokens.append(tgt)

    # Stacking the image since they are already the same size (CLIP features)
    image_feats = torch.stack(image_feats)

    # Finding the longest story in this batch
    max_seq_len = 0
    for t in input_tokens:
        if t.size(0) > max_seq_len:
            max_seq_len = t.size(0)

    # Pad every story to match the longest one
    padded_inputs = []
    padded_targets = []

    eos_token = 50256  # using GPT-2's eos token for padding the extra input positions
    ignore_label = -1  # loss will ignore these positions in targets

    for i in range(len(input_tokens)):
        inp = input_tokens[i]
        tgt = target_tokens[i]
    
        pad_len = max_seq_len - inp.size(0)
    
        padded_inp = F.pad(inp, (0, pad_len), value=eos_token)
        padded_inputs.append(padded_inp)
    
        padded_tgt = F.pad(tgt, (0, pad_len), value=ignore_label)
        padded_targets.append(padded_tgt)

