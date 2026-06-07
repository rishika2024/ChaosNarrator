import torch
import os
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip.eval()

os.makedirs('data/clip_features', exist_ok=True)

print("Loading Flickr30k dataset...")
dataset = load_dataset("nlphuji/flickr30k", split="test")
print(f"Total examples: {len(dataset)}")

all_features = []
all_stories = []
all_story_ids = []
batch_size = 32
max_stories = 20000

for i in range(0, min(len(dataset), max_stories), batch_size):
    batch = dataset[i:i + batch_size]

    images = []
    captions = []
    ids = []

    for j in range(len(batch['image'])):
        try:
            img = batch['image'][j].convert('RGB')
            caps = batch['caption'][j]
            if isinstance(caps, list):
                caption = ' '.join(caps)
            else:
                caption = str(caps)
            images.append(img)
            captions.append(caption)
            ids.append(str(i + j))
        except:
            continue

    if len(images) == 0:
        continue

    try:
        inputs = clip_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = clip.vision_model(pixel_values=inputs["pixel_values"])
            features = output.pooler_output.cpu()

        for k in range(len(images)):
            single_feat = features[k].unsqueeze(0).repeat(5, 1)
            all_features.append(single_feat)
            all_stories.append(captions[k])
            all_story_ids.append(ids[k])

    except Exception as e:
        print(f"  Batch failed: {e}")
        continue

    if len(all_features) % 1000 == 0 and len(all_features) > 0:
        print(f"Processed {len(all_features)} stories")

# Final save
final = {
    'features': torch.stack(all_features),
    'stories': all_stories,
    'story_ids': all_story_ids
}
torch.save(final, 'data/clip_features/train_features.pt')
print(f"\nDone! Saved {len(all_features)} stories")
print(f"Feature tensor shape: {final['features'].shape}")