from transformers import AutoTokenizer, CLIPModel
import torch.nn as nn
import torch


tokenizer = AutoTokenizer.from_pretrained("gpt2")
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

class MultimodalTokenAndPositionEmbedding(nn.Module):
    def __init__(self, max_len, vocab_size, embed_dim, clip_embed_dim=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.image_embedding = nn.Linear(clip_embed_dim, embed_dim)

    def forward(self, token_ids, image_features):
        
        text_tokens = self.token_embedding(token_ids)
        image_tokens = self.image_embedding(image_features)

        combined = torch.cat([image_tokens, text_tokens], dim=1)

        positions = torch.arange(combined.size(1), device=combined.device)
        positions = self.position_embedding(positions)

        return combined + positions
    
# Settings
batch_size = 2
num_images = 3
seq_len = 10
vocab_size = 50257
embed_dim = 256
max_len = 100

# Create the layer
embedding = MultimodalTokenAndPositionEmbedding(max_len, vocab_size, embed_dim)

# Fake data
fake_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
fake_image_features = torch.randn(batch_size, num_images, 512)

# Run it
output = embedding(fake_token_ids, fake_image_features)

print(f"Token IDs shape: {fake_token_ids.shape}")
print(f"Image features shape: {fake_image_features.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected: ({batch_size}, {num_images + seq_len}, {embed_dim})")