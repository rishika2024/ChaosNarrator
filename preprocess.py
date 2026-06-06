import json

with open('data/sis/train.story-in-sequence.json') as f:
    data = json.load(f)

# Build image URL lookup: flickr_id -> url
img_lookup = {}
for img in data['images']:
    if 'url_o' in img:
        img_lookup[img['id']] = img['url_o']

# Group annotations by story_id
stories_raw = {}
for ann in data['annotations']:
    ann = ann[0]
    sid = ann['story_id']
    if sid not in stories_raw:
        stories_raw[sid] = []
    stories_raw[sid].append(ann)

# Build clean dataset: each entry has image URLs + full story text
dataset = []
skipped = 0
for sid, parts in stories_raw.items():
    parts = sorted(parts, key=lambda x: x['worker_arranged_photo_order'])
    
    urls = []
    for p in parts:
        fid = p['photo_flickr_id']
        if fid in img_lookup:
            urls.append(img_lookup[fid])
        else:
            urls.append(None)
    
    if None in urls:
        skipped += 1
        continue
    
    story = ' '.join([p['text'] for p in parts])
    
    dataset.append({
        'story_id': sid,
        'image_urls': urls,
        'story': story
    })

print(f"Total usable stories: {len(dataset)}")
print(f"Skipped (missing images): {skipped}")
print()
print("Example entry:")
print(f"  Story: {dataset[0]['story']}")
print(f"  Images: {len(dataset[0]['image_urls'])}")
for url in dataset[0]['image_urls']:
    print(f"    {url}")