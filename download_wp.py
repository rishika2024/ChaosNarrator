from datasets import load_dataset
import torch
import os

print("Loading WritingPrompts...")
dataset = load_dataset("euclaise/writingprompts", split="train")
print(f"Total examples: {len(dataset)}")

stories = []
prompts = []

for i in range(len(dataset)):
    story = dataset[i]['story']
    prompt = dataset[i]['prompt']

    word_count = len(story.split())
    if 100 <= word_count <= 500:
        stories.append(story)
        prompts.append(prompt)

    if i % 10000 == 0:
        print(f"Scanned {i} | Kept {len(stories)}")

print(f"\nKept {len(stories)} stories")

os.makedirs('data/larger_model/writingprompt', exist_ok=True)
torch.save({
    'stories': stories,
    'prompts': prompts,
}, 'data/larger_model/writingprompt/wp_train.pt')

print("Saved to data/larger_model/writingprompt/wp_train.pt")
print(f"\nSample prompt: {prompts[0][:100]}...")
print(f"Sample story: {stories[0][:300]}...")