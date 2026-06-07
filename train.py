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
max_len = 200
clip_embed_dim = 768
batch_size = 16
learning_rate = 3e-4
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# A helper function to fix story sizes in a batch by padding them to the same length
def fix_story_size(batch):
    # Unpacking the batch into separate lists
    clip_vectors = [] # raw CLIP features for each story in the batch
    input_tokens = [] 
    target_tokens = []

    for img, inp, tgt in batch:
        clip_vectors.append(img)
        input_tokens.append(inp)
        target_tokens.append(tgt)

    # Stacking the clip vectors since they are already the same size (CLIP features)
    clip_vectors = torch.stack(clip_vectors)

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

# Load datasets
train_dataset = VISTDataset('data/clip_features/train_features.pt', max_len=max_len)
val_dataset = VISTDataset('data/clip_features/val_features.pt', max_len=max_len)
test_dataset = VISTDataset('data/clip_features/test_features.pt', max_len=max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=fix_story_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=fix_story_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=fix_story_size)

print(f"Train size: {len(train_dataset)} stories")
print(f"Val size: {len(val_dataset)} stories")
print(f"Test size: {len(test_dataset)} stories")

# Creating model
model = ChaosNarrator(vocab_size, embed_dim, num_heads, num_layers, max_len, clip_embed_dim)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# Optimizer: AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Tracking
best_val_loss = float('inf')
train_losses = []
val_losses = []
batch_losses = []  # per-batch training loss for detailed plot
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots', exist_ok=True)


def plot_losses(train_losses, val_losses, batch_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: train vs val loss per epoch
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Right plot: training loss per batch
    ax2.plot(batch_losses, 'b-', linewidth=0.5)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss per Batch')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('plots/loss_curves.png', dpi=150)
    plt.close()
    print("  Plot saved: plots/loss_curves.png")


def run_epoch(loader, training=True):
    if training:
        model.train()
        context_train_or_eval = torch.enable_grad()
    else:
        model.eval()
        context_train_or_eval = torch.no_grad()

    total_loss = 0
    total_batches = 0

    with context_train_or_eval:
        # Loop through each batch of data
        for batch_idx, (clip_vectors, input_tokens, target_tokens) in enumerate(loader):
            # move to gpu if available
            clip_vectors = clip_vectors.to(device)
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            logits, loss = model(input_tokens, clip_vectors, targets=target_tokens)
            if training:
                optimizer.zero_grad() # clear previous gradients
                loss.backward() # backpropagation
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping to prevent exploding gradients max vaue is 1.0
                optimizer.step() # upating model parameters
                batch_losses.append(loss.item()) # track batch loss for plotting

            total_loss += loss.item()
            total_batches += 1

            if training and (batch_idx + 1) % 50 == 0:
                avg = total_loss / total_batches
                print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {avg:.4f}")

    return total_loss / total_batches


# --- TRAINING ---
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'batch_losses': batch_losses,
    }, f'checkpoints/epoch_{epoch+1}.pt')

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'val_loss': val_loss,
        }, 'checkpoints/best_model.pt')
        print(f"  New best model! Val Loss: {val_loss:.4f}")

    # Updating plot after every epoch
    plot_losses(train_losses, val_losses, batch_losses)


# --- TESTING ---
print("\n" + "=" * 50)
print("Running test evaluation with best model")
print("=" * 50)

best = torch.load('checkpoints/best_model.pt')
model.load_state_dict(best['model_state'])
print(f"Loaded best model from epoch {best['epoch']}")

test_loss = run_epoch(test_loader, training=False)
print(f"Test Loss: {test_loss:.4f}")


# --- GENERATING SAMPLE STORIES ---
print("\n" + "=" * 50)
print("Generating sample stories")
print("=" * 50)

import json

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load VIST data to find image URLs for test stories
with open('data/sis/train.story-in-sequence.json') as f:
    vist_data = json.load(f)

# Build lookup: story_id -> image URLs
img_lookup = {}
for img in vist_data['images']:
    if 'url_o' in img:
        img_lookup[img['id']] = img['url_o']

stories_raw = {}
for ann in vist_data['annotations']:
    ann = ann[0]
    sid = ann['story_id']
    if sid not in stories_raw:
        stories_raw[sid] = []
    stories_raw[sid].append(ann)

story_to_urls = {}
for sid, parts in stories_raw.items():
    parts = sorted(parts, key=lambda x: x['worker_arranged_photo_order'])
    urls = []
    for p in parts:
        fid = p['photo_flickr_id']
        if fid in img_lookup:
            urls.append(img_lookup[fid])
    story_to_urls[sid] = urls

model.eval()
with torch.no_grad():
    for i in range(5):
        image_feat, input_tokens, _ = test_dataset[i]
        image_feat = image_feat.unsqueeze(0).to(device)

        start_tokens = torch.tensor([[tokenizer.eos_token_id]]).to(device)

        generated = model.generate(start_tokens, image_feat, max_new_tokens=100, temperature=0.8, top_k=40)
        story = tokenizer.decode(generated[0], skip_special_tokens=True)

        # Find image URLs for this story
        story_id = test_dataset.story_ids[i]
        urls = story_to_urls.get(story_id, [])

        print(f"\nStory {i+1}:")
        print(f"  Images:")
        for url in urls:
            print(f"    {url}")
        print(f"  Original: {test_dataset.stories[i][:200]}...")
        print(f"  Generated: {story[:200]}...")

print("\nDone!")