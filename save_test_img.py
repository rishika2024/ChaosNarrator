from datasets import load_dataset
import os

dataset = load_dataset("nlphuji/flickr30k", split="test")

os.makedirs('data/test_images', exist_ok=True)

for i in range(10):
    img = dataset[i]['image'].convert('RGB')
    img.save(f'data/test_images/test_{i}.jpg')
    print(f"Saved test_{i}.jpg")

print("Done!")