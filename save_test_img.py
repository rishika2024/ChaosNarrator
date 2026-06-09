from datasets import load_dataset
import torch
import os

dataset = load_dataset("nlphuji/flickr30k", split="test")
test_data = torch.load('data/clip_features/test_features.pt')

os.makedirs('data/test_images', exist_ok=True)

for i in range(min(10, len(test_data['story_ids']))):
    idx = int(test_data['story_ids'][i])
    img = dataset[idx]['image'].convert('RGB')
    img.save(f'data/test_images/test_{i}.jpg')
    print(f"test_{i}.jpg: {test_data['stories'][i][:80]}...")

print("Done!")