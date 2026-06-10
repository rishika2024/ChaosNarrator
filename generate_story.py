import torch
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
from transformer import ChaosNarrator
from PIL import Image

vocab_size = 50257
embed_dim = 512
num_heads = 8
num_layers = 8
max_len = 200
clip_embed_dim = 768
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load CLIP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = ChaosNarrator(vocab_size, embed_dim, num_heads, num_layers, max_len, clip_embed_dim)
model = model.to(device)

best = torch.load('best_model.pt', map_location=device)
model.load_state_dict(best['model_state'])

test_keywords = [
    "cosmic battle between cats and dogs",
    "hit by a bus, wakes up with superpowers",
    "pizza grew legs and ran away",
    "the love between a fish and a bird",
    "a world where gravity suddenly reverses"
]

print("\n" + "=" * 50)
print("Generating stories with images + keywords")
print("=" * 50)

model.eval()
with torch.no_grad():
    for i in range(5):
        # load test image i and extract CLIP features
        image = Image.open(f'data/test_images/test_{i}.jpg').convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        feat = clip_model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output  # (1, 768)
        clip_vector = feat.unsqueeze(1).repeat(1, 5, 1)  # (1, 5, 768)

        keywords = test_keywords[i]
        start_tokens = tokenizer(keywords, return_tensors="pt").input_ids.to(device)

        generated = model.generate(start_tokens, clip_vector, max_new_tokens=200, temperature=1.0, top_k=40)
        story = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\nStory {i+1} (test_{i}.jpg):")
        print(f"  Keywords: {keywords}")
        print(f"  Generated: {story[:300]}...")