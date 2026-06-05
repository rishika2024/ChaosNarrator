from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
import torch.nn as nn
import torch
from PIL import Image
import requests

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
    
# # Testing with real data

# # Downloading 2 real images from CLIP github
# urls = [
#     "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
#     "https://raw.githubusercontent.com/pytorch/hub/master/images/deeplab1.png",
# ]
# images = []
# for url in urls:
#     img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#     images.append(img)
#     print(f"Loaded image: {img.size}")

# # Extracting the frozen CLIP features
# clip_inputs = clip_processor(images=images, return_tensors="pt")
# with torch.no_grad():
#     output = clip.vision_model(pixel_values=clip_inputs["pixel_values"])
#     image_features = output.pooler_output

# print(f"CLIP output shape: {image_features.shape}")

# # Add batch dimension: (2, 512) -> (1, 2, 512) because 1 example with 2 images
# image_features = image_features.unsqueeze(0)
# print(f"After adding batch dim: {image_features.shape}")

# # Tokenize real text keywords
# text = "dragon gets hit by a bus"
# token_ids = tokenizer(text, return_tensors="pt").input_ids
# print(f"Text: '{text}'")
# print(f"Token IDs: {token_ids}")
# print(f"Token IDs shape: {token_ids.shape}")

# # Run through embedding layer
# embedding = MultimodalTokenAndPositionEmbedding(
#     max_len=100,
#     vocab_size=tokenizer.vocab_size,
#     embed_dim=256
# )

# output = embedding(token_ids, image_features)

# print(f"\nOutput shape: {output.shape}")
# num_img = image_features.shape[1]
# num_txt = token_ids.shape[1]
# print(f"That is {num_img} image tokens + {num_txt} text tokens = {num_img + num_txt} total")


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