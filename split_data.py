import torch

data = torch.load('data/clip_features/train_features.pt')
total = len(data['stories'])
print(f"Total stories: {total}")

# 80% train, 10% val, 10% test
train_end = int(total * 0.8)
val_end = int(total * 0.9)

for split, start, end in [('train', 0, train_end), ('val', train_end, val_end), ('test', val_end, total)]:
    split_data = {
        'features': data['features'][start:end],
        'stories': data['stories'][start:end],
        'story_ids': data['story_ids'][start:end]
    }
    torch.save(split_data, f'data/clip_features/{split}_features.pt')
    print(f"{split}: {end - start} stories")

print("Done!")