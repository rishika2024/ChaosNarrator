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
embed_dim = 512  # was 256 earlier
num_heads = 8 # was 4 earlier
num_layers = 8 # was 4 earlier
max_len = 200
clip_embed_dim = 768
batch_size = 128 # was 16 earlier
learning_rate = 1e-4  # changed learning rate from 5e-5 earlier
num_epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def fix_story_size(batch):
    """Pads stories in a batch to the same length so they can be stacked into tensors.

    clip_vectors will be dummy zeros for Stage 2 (no real images).
    Inputs are padded with GPT-2's EOS token (50256); targets are padded with -1
    so those positions are ignored by cross-entropy loss.
    Returns (clip_vectors, padded_inputs, padded_targets) as stacked tensors.
    """
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
full_data = torch.load('data/larger_model/writingprompts/wp_train.pt') # loaded from download_wp.py, contains 'stories' and 'prompts' lists
total = len(full_data['stories']) # should be 20000 based on download_wp.py filtering
train_end = int(total * 0.9) # 90% for training, 10% for validation

# Split into train and val and save separately for easier loading in dataset class
torch.save({
    'stories': full_data['stories'][:train_end],
    'prompts': full_data['prompts'][:train_end],
}, 'data/larger_model/writingprompts/wp_train_split.pt')

torch.save({
    'stories': full_data['stories'][train_end:],
    'prompts': full_data['prompts'][train_end:],
}, 'data/larger_model/writingprompts/wp_val_split.pt')

# Load datasets
train_dataset = WritingPromptsDataset('data/larger_model/writingprompts/wp_train_split.pt', max_len=max_len)
val_dataset = WritingPromptsDataset('data/larger_model/writingprompts/wp_val_split.pt', max_len=max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=fix_story_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=fix_story_size)

print(f"Train size: {len(train_dataset)} stories")
print(f"Val size: {len(val_dataset)} stories")

# Create model and load Stage 1 weights
model = ChaosNarrator(vocab_size, embed_dim, num_heads, num_layers, max_len, clip_embed_dim)
model = model.to(device)

# Load best model from Stage 1
if not os.path.exists('larger_model_checkpoints/best_model.pt'):
    print("Warning: Stage 1 checkpoint not found, using fresh model")
else:
    stage1 = torch.load('larger_model_checkpoints/best_model.pt')
    model.load_state_dict(stage1['model_state'])
    print(f"Loaded Stage 1 best model from epoch {stage1['epoch']}")

# # Freeze the image projection layer to preserve image grounding
# for param in model.embedding.image_embedding.parameters():
#     param.requires_grad = False
# print("Frozen image projection layer")

# .numel() returns the total number of elements in the tensor
# counting how many trainable parameters we have in this model
total_params = 0
for p in model.parameters():
    if p.requires_grad:
        total_params += p.numel()
print(f"Trainable parameters: {total_params / 1e6:.2f}M")


# Optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate
)

# Tracking
best_val_loss = float('inf')
train_losses = []
val_losses = []
batch_losses = []
os.makedirs('larger_model_checkpoints_stage2/unfrozen', exist_ok=True)
os.makedirs('larger_model_plots/unfrozen', exist_ok=True)


def plot_losses(train_losses, val_losses, batch_losses):
    """Saves a two-panel Stage 2 loss plot to larger_model_plots/unfrozen/stage2_loss_curves.png.

    Left panel: train vs validation loss per epoch.
    Right panel: training loss per individual batch.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Left plot: train vs val loss per epoch
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Stage 2: Train vs Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Right plot: training loss per batch
    ax2.plot(batch_losses, 'b-', linewidth=0.5)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Stage 2: Training Loss per Batch')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('larger_model_plots/unfrozen/stage2_loss_curves.png', dpi=150)
    plt.close()
    print("  Plot saved: larger_model_plots/unfrozen/stage2_loss_curves.png")


def run_epoch(loader, training=True):
    """Runs one full pass over the dataset, either training or evaluating.

    In training mode: zeros gradients, runs backprop, clips gradients, and steps
    the optimizer, and appends each batch loss to batch_losses for plotting.
    Returns the average loss over all batches.
    """
    if training:
        model.train()
        context = torch.enable_grad()
    else:
        model.eval()
        context = torch.no_grad()

    total_loss = 0
    total_batches = 0

    with context:
        # Loop through each batch of data
        for batch_idx, (clip_vectors, input_tokens, target_tokens) in enumerate(loader):
            # move to gpu if available
            clip_vectors = clip_vectors.to(device)
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            logits, loss = model(input_tokens, clip_vectors, targets=target_tokens)

            if training:
                optimizer.zero_grad() # zero out gradients before backward pass
                loss.backward() # backpropagation to compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping to prevent exploding gradients, max value is 1.0
                optimizer.step() # update model parameters based on computed gradients
                batch_losses.append(loss.item()) # track batch loss for plotting

            total_loss += loss.item()
            total_batches += 1

            if training and (batch_idx + 1) % 50 == 0:
                avg = total_loss / total_batches
                print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {avg:.4f}")

    return total_loss / total_batches


# --- STAGE 2 TRAINING ---
print("\nStarting Stage 2 Training (WritingPrompts)")
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
    }, f'larger_model_checkpoints_stage2/unfrozen/epoch_{epoch+1}.pt')
    
    # Save best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'val_loss': val_loss,
        }, 'larger_model_checkpoints_stage2/unfrozen/best_model.pt')
        print(f"  New best model! Val Loss: {val_loss:.4f}")
    
    # update the loss plots after each epoch
    plot_losses(train_losses, val_losses, batch_losses)


# --- GENERATING SAMPLE STORIES ---
print("\n" + "=" * 50)
print("Generating stories with images + keywords")
print("=" * 50)

from dataset import VISTDataset
test_dataset = VISTDataset('data/clip_features/test_features.pt', max_len=max_len)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

if not os.path.exists('larger_model_checkpoints_stage2/unfrozen/best_model.pt'):
    print("Warning: Stage 2 checkpoint not found, skipping generation")
else:
    best = torch.load('larger_model_checkpoints_stage2/unfrozen/best_model.pt')
    model.load_state_dict(best['model_state'])

    test_keywords = [
        "dragon battle chaos",
        "time travel apocalypse",
        "alien invasion pizza",
        "wizard zombie dance party",
        "robot falls in love with toaster",
    ]

    print("\n" + "=" * 50)
    print("Stage 2 Run 1 (unfrozen projection, higher temp)")
    print("=" * 50)

    model.eval()
    with torch.no_grad():
        for i in range(5):
            clip_vector, input_tokens, target_tokens = test_dataset[i]
            clip_vector = clip_vector.unsqueeze(0).to(device)

            keywords = test_keywords[i]
            start_tokens = tokenizer(keywords, return_tensors="pt").input_ids.to(device)
        
        

            generated = model.generate(start_tokens, clip_vector, max_new_tokens=200, temperature=0.9, top_k=40)

            story = tokenizer.decode(generated[0], skip_special_tokens=True)

            print(f"\nStory {i+1}:")
            print(f"  Keywords: {keywords}")
            print(f"  Generated: {story[:300]}...")