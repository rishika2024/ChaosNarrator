from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
import torch.nn as nn
import torch
from PIL import Image
import requests
import math
import torch.nn.functional as F

# Loading the models


tokenizer = AutoTokenizer.from_pretrained("gpt2")

# patch32 is the smallest CLIP, so fastest to test
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") 
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class MultimodalTokenAndPositionEmbedding(nn.Module):
    def __init__(self, max_len, vocab_size, embed_dim, clip_embed_dim=768):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.image_embedding = nn.Linear(clip_embed_dim, embed_dim)

    def forward(self, token_ids, image_features):
        text_tokens = self.token_embedding(token_ids)
        image_tokens = self.image_embedding(image_features)   
        # Concatenating image and text tokens     
        combined = torch.cat([image_tokens, text_tokens], dim=1) 
        # torch.arange(9) creates indices [0, 1, 2, 3, 4, 5, 6, 7, 8] for 2 image token + 7 text tokens
        positions = torch.arange(combined.size(1), device=combined.device) 
        # takes indices and maps to position embeddings
        positions = self.position_embedding(positions)
        return combined + positions # adding position embeddings to the combined tokens
    
# Testing with real data

# Downloading 2 real images from CLIP github
urls = [
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "https://raw.githubusercontent.com/pytorch/hub/master/images/deeplab1.png",
]
images = []
for url in urls:
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    images.append(img)
    print(f"Loaded image: {img.size}")

# Extracting the frozen CLIP features
clip_inputs = clip_processor(images=images, return_tensors="pt")
with torch.no_grad():
    output = clip.vision_model(pixel_values=clip_inputs["pixel_values"])
    image_features = output.pooler_output

print(f"CLIP output shape: {image_features.shape}")

# Add batch dimension: (2, 512) -> (1, 2, 512) because 1 example with 2 images
image_features = image_features.unsqueeze(0)
print(f"After adding batch dim: {image_features.shape}")

# Tokenize real text keywords
text = "dragon gets hit by a bus"
token_ids = tokenizer(text, return_tensors="pt").input_ids
print(f"Text: '{text}'")
print(f"Token IDs: {token_ids}")
print(f"Token IDs shape: {token_ids.shape}")

# Run through embedding layer
embedding = MultimodalTokenAndPositionEmbedding(
    max_len=100,
    vocab_size=tokenizer.vocab_size,
    embed_dim=256
)

output = embedding(token_ids, image_features)

print(f"\nOutput shape: {output.shape}")
num_img = image_features.shape[1]
num_txt = token_ids.shape[1]
print(f"That is {num_img} image tokens + {num_txt} text tokens = {num_img + num_txt} total")


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        # making sure the embedding dimension is divisible by the number of heads
        # to avoid decimal head dimensions
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Weights for query, key, value projections
        # nn.Linear creates matrix of weights and bias for linear transformation
        self.W_q = nn.Linear(embed_dim, embed_dim) 
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # output projection => combining the heads back together
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def causal_attention_mask(self, T, device):
        i = torch.arange(T, device=device)[:, None] # [:, None] adds a new dimension to make it a column vector
        j = torch.arange(T, device=device)
        # comparing i, j
        # if i>= j, we want to attend (mask value 1), else mask value 0
        # this creates a lower triangular matrix of 1s
        mask = (i >= j).float() 
        return mask.view(1, 1, T, T)
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension

        # creating Q, K, V separately
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # splitting into multiple heads 
        # view: (B, T, C) -> (B, T, num_heads, head_dim)
        # transpose: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim) for attention calculation
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # calculating attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # creating and applying causal mask
        mask = self.causal_attention_mask(T, x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # applying softmax and dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # calculating weighted sum of values
        out = weights @ v

        # combining heads back together
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # final projection
        out = self.resid_dropout(self.out_proj(out))

        return out
        
# Test CausalSelfAttention
attention = CausalSelfAttention(embed_dim=256, num_heads=4)

# Use the output from your embedding layer as input
# output shape was (1, 8, 256) from earlier
test_output = attention(output)

print(f"Attention input shape: {output.shape}")
print(f"Attention output shape: {test_output.shape}")


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)  # layer norm 1
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)  # self attention layer
        self.ln_2 = nn.LayerNorm(embed_dim)  # layer norm 2
        # feedforward network linear layer -> GELU -> linear layer -> dropout
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # expanding dimension of ffn to 4x for more capacity
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x = x + attention(layer 1 norm of x)
        # x = x + feedforward(layer 2 norm of x) x is the updated x from previous line
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x
    
# Test TransformerBlock
block = TransformerBlock(embed_dim=256, num_heads=4)

# output shape (1, 8, 256)
block_output = block(output)

print(f"Block input shape: {output.shape}")
print(f"Block output shape: {block_output.shape}")

class ChaosNarrator(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len, clip_embed_dim=768, dropout=0.1):
        super().__init__()

        # multimodal embedding
        self.embedding = MultimodalTokenAndPositionEmbedding(max_len, vocab_size, embed_dim, clip_embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        # stacking transformer blocks together
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(embed_dim, num_heads, dropout))

        # final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # output head: mapping embed_dim back to vocab_size
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # store for generate function
        self.max_len = max_len

    def forward(self, token_ids, image_features, targets=None):
        # embedding: images + text + positions
        # then dropout
        x = self.embedding(token_ids, image_features)
        x = self.embed_dropout(x)

        # passing through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # final layer norm
        x = self.ln_f(x)

        # predicting next token
        logits = self.lm_head(x)

        # computing loss if targets provided
        loss = None
        if targets is not None:
            # F.cross_entropy compares each prediction against the correct answer        
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # reshape to (batch_size * seq_len, vocab_size) for loss calculation
                targets.view(-1), # reshape to (batch_size * seq_len) to match logits
                ignore_index=-1 # if target token is -1, it will be ignored
            )

        return logits, loss
    
    @torch.no_grad() # gradients not needed for generation
    def generate(self, token_ids, image_features, max_new_tokens, temperature=1.0, top_k=None):
        self.eval() # set model to evaluation mode
        for _ in range(max_new_tokens):
            # crop if sequence is too long by removing the oldest tokens
            if token_ids.size(1) + image_features.size(1) > self.max_len:
                token_ids = token_ids[:, -(self.max_len - image_features.size(1)):]

            # forward pass
            logits, dummy = self(token_ids, image_features)

            # getting logits for last position and scaling by temperature/randomness
            logits = logits[:, -1, :] / temperature

            # top_k filtering: except the top_k most likely tokens, set the rest to -inf
            #  so they won't be sampled
            if top_k is not None:
                v, dummy = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            # softmax to get probabilities (-inf will become 0 probability)
            probs = F.softmax(logits, dim=-1)

            # sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # append to sequence
            token_ids = torch.cat([token_ids, next_token], dim=1)

        return token_ids

# TESTING THE FULL MODEL
# Settings
vocab_size = 50257
embed_dim = 256
num_heads = 4
num_layers = 4
max_len = 100

# Create model
model = ChaosNarrator(vocab_size, embed_dim, num_heads, num_layers, max_len)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# Test forward pass with real data from earlier
logits, loss = model(token_ids, image_features)
print(f"Logits shape: {logits.shape}")
# Expected: (1, 8, 50257) — for each of the 8 tokens, a score for every word in vocab

# Test generate
generated = model.generate(token_ids, image_features, max_new_tokens=20, temperature=0.8, top_k=40)
print(f"Generated token IDs shape: {generated.shape}")
print(f"Generated text: {tokenizer.decode(generated[0])}")

    
