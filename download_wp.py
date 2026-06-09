from datasets import load_dataset
import torch
import os

print("Loading WritingPrompts...")
dataset = load_dataset("euclaise/writingprompts", split="train")
print(f"Total examples: {len(dataset)}")

## Filter for "wild" stories with certain keywords in the prompt and length between 100-500 words

wild_keywords = ['dragon', 'magic', 'alien', 'monster', 'demon', 'wizard',
                 'zombie', 'robot', 'god', 'immortal', 'dimension', 'portal',
                 'superpower', 'apocalypse', 'time travel', 'universe',
                 'villain', 'hero', 'destroy', 'chaos', 'explosion',
                 'quest', 'kingdom', 'sword', 'battle', 'curse', 'spell',
                 'ghost', 'witch', 'vampire', 'werewolf', 'pirate',
                 'spaceship', 'galaxy', 'mutant', 'prophecy', 'throne',
                 'dungeon', 'warrior', 'sorcerer', 'enchanted', 'undead',
                 'invasion', 'summoned', 'reincarnated', 'transmigration',
                 'isekai', 'overlord', 'demon king', 'chosen one']

stories = []
prompts = []
skipped = 0

for i in range(len(dataset)):
    story = dataset[i]['story']
    prompt = dataset[i]['prompt'].lower()

    word_count = len(story.split())
    if 100 <= word_count <= 500:
        if any(kw in prompt for kw in wild_keywords):
            stories.append(story)
            prompts.append(prompt)
        else:
            skipped += 1

    if len(stories) >= 20000:
        break

    if i % 10000 == 0:
        print(f"Scanned {i} | Kept {len(stories)} | Skipped {skipped}")

print(f"\nKept {len(stories)} wild stories")
print(f"Skipped {skipped} boring stories")

os.makedirs('data/wild/writingprompts', exist_ok=True)
torch.save({
    'stories': stories,
    'prompts': prompts,
}, 'data/wild/writingprompts/wp_train.pt')
print("Saved to data/writingprompts/wp_train.pt")

print(f"\nSample wild prompts:")
for i in range(min(5, len(prompts))):
    print(f"  {i+1}: {prompts[i][:100]}...")
print(f"\nSample story: {stories[0][:300]}...")