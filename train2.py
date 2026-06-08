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
    # A helper function to fix story sizes in a batch by padding them to the same length
    clip_vectors = [] # raw CLIP features for each story in the batch (will be dummy zeros for Stage 2)
    input_tokens = [] # tokenized story inputs (without the last token)
    target_tokens = [] # tokenized story targets (without the first token)

    for img, inp, tgt in batch:
        clip_vectors.append(img)
        input_tokens.append(inp)
        target_tokens.append(tgt)
    
    # Stack the clip vectors since they are already the same size (dummy zeros)
    clip_vectors = torch.stack(clip_vectors)

    # Find the longest story in this batch
    max_seq_len = 0
    for t in input_tokens:
        if t.size(0) > max_seq_len:
            max_seq_len = t.size(0)

    # Padding every story to match the longest one
    padded_inputs = []
    padded_targets = []
    eos_token = 50256 # using GPT-2's eos token for padding the extra input positions
    ignore_label = -1 # loss will ignore these positions in targets

    for i in range(len(input_tokens)):
        inp = input_tokens[i]
        tgt = target_tokens[i]

        # Calculate how much padding is needed
        # pad_len = max_seq_len - size of the row (current story length)
        pad_len = max_seq_len - inp.size(0)

        # F.pad adds extra values to the edges of a tensor
        # Padding EOF tokens to the input
        # 0-> 0 padding on the left, pad_len -> padding on the right, value is the token to pad with
        padded_inp = F.pad(inp, (0, pad_len), value=eos_token)
        padded_inputs.append(padded_inp)

        # Padding ignore labels to the target  
        padded_tgt = F.pad(tgt, (0, pad_len), value=ignore_label)
        padded_targets.append(padded_tgt)

     # a list of individual story tensors of the same length.
    padded_inputs = torch.stack(padded_inputs)
    padded_targets = torch.stack(padded_targets)

    return clip_vectors, padded_inputs, padded_targets

# Load WritingPrompts data
full_data = torch.load('data/writingprompts/wp_train.pt') # loaded from download_wp.py, contains 'stories' and 'prompts' lists
total = len(full_data['stories']) # should be 20000 based on download_wp.py filtering
train_end = int(total * 0.9) # 90% for training, 10% for validation

# Split into train and val and save separately for easier loading in dataset class
torch.save({
    'stories': full_data['stories'][:train_end],
    'prompts': full_data['prompts'][:train_end],
}, 'data/writingprompts/wp_train_split.pt')

torch.save({
    'stories': full_data['stories'][train_end:],
    'prompts': full_data['prompts'][train_end:],
}, 'data/writingprompts/wp_val_split.pt')

# Load datasets
train_dataset = WritingPromptsDataset('data/writingprompts/wp_train_split.pt', max_len=max_len)
val_dataset = WritingPromptsDataset('data/writingprompts/wp_val_split.pt', max_len=max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=fix_story_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=fix_story_size)

print(f"Train size: {len(train_dataset)} stories")
print(f"Val size: {len(val_dataset)} stories")

# Create model and load Stage 1 weights
model = ChaosNarrator(vocab_size, embed_dim, num_heads, num_layers, max_len, clip_embed_dim)
model = model.to(device)

# Load best model from Stage 1
stage1 = torch.load('checkpoints/best_model.pt')
model.load_state_dict(stage1['model_state'])
print(f"Loaded Stage 1 best model from epoch {stage1['epoch']}")

