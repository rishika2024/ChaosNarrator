import json
import torch
import requests
import os
import sys
from PIL import Image
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor
from concurrent.futures import ThreadPoolExecutor

# Which splits to process
if len(sys.argv) > 1:
    splits = sys.argv[1:]
else:
    splits = ['train', 'val', 'test']

# Load CLIP on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip.eval()

os.makedirs('data/clip_features', exist_ok=True)

def download_image(url):
    try:
        resp = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        return Image.open(BytesIO(resp.content)).convert('RGB')
    except:
        return None

def process_split(split):
    print(f"\n{'='*50}")
    print(f"Processing split: {split}")
    print(f"{'='*50}")

    with open(f'data/sis/{split}.story-in-sequence.json') as f:
        data = json.load(f)

    # Build image URL lookup
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

    # Build dataset
    dataset = []
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
            continue
        story = ' '.join([p['text'] for p in parts])
        dataset.append({
            'story_id': sid,
            'image_urls': urls,
            'story': story
        })

    print(f"Total stories to process: {len(dataset)}")

    save_path = f'data/clip_features/{split}_features.pt'

    # Check for resume
    done_ids = set()
    all_features = []
    all_stories = []
    all_story_ids = []

    if os.path.exists(save_path):
        existing = torch.load(save_path)
        done_ids = set(existing['story_ids'])
        all_features = list(existing['features'])
        all_stories = existing['stories']
        all_story_ids = existing['story_ids']
        print(f"Resuming from checkpoint: {len(done_ids)} stories already done")

    batch_size = 20
    batch_entries = []
    batch_images = []
    batch_image_counts = []
    failed = 0
    save_every = 500
    processed = 0

    for i, entry in enumerate(dataset):
        if entry['story_id'] in done_ids:
            continue

        with ThreadPoolExecutor(max_workers=5) as executor:
            images = list(executor.map(download_image, entry['image_urls']))

        if None in images:
            failed += 1
            continue

        batch_entries.append(entry)
        batch_images.extend(images)
        batch_image_counts.append(len(images))

        if len(batch_entries) >= batch_size:
            try:
                inputs = clip_processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    output = clip.vision_model(pixel_values=inputs["pixel_values"])
                    features = output.pooler_output.cpu()

                idx = 0
                for entry_b, count in zip(batch_entries, batch_image_counts):
                    story_features = features[idx:idx + count]
                    all_features.append(story_features)
                    all_stories.append(entry_b['story'])
                    all_story_ids.append(entry_b['story_id'])
                    idx += count

                processed += len(batch_entries)

            except Exception as e:
                failed += len(batch_entries)
                print(f"  Batch failed: {e}")

            batch_entries = []
            batch_images = []
            batch_image_counts = []

            if processed % 20 == 0:
                print(f"[{split}] Processed {processed} | Total saved {len(all_features)} | Failed {failed}")

            if len(all_features) % save_every < batch_size and len(all_features) > 0:
                checkpoint = {
                    'features': torch.stack(all_features),
                    'stories': all_stories,
                    'story_ids': all_story_ids
                }
                torch.save(checkpoint, save_path)
                print(f"  Checkpoint saved: {len(all_features)} stories")

    # Process remaining
    if len(batch_entries) > 0:
        try:
            inputs = clip_processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = clip.vision_model(pixel_values=inputs["pixel_values"])
                features = output.pooler_output.cpu()

            idx = 0
            for entry_b, count in zip(batch_entries, batch_image_counts):
                story_features = features[idx:idx + count]
                all_features.append(story_features)
                all_stories.append(entry_b['story'])
                all_story_ids.append(entry_b['story_id'])
                idx += count
        except Exception as e:
            failed += len(batch_entries)

    # Final save
    if len(all_features) > 0:
        final = {
            'features': torch.stack(all_features),
            'stories': all_stories,
            'story_ids': all_story_ids
        }
        torch.save(final, save_path)
        print(f"\nDone with {split}! Saved {len(all_features)} stories")
        print(f"Failed: {failed}")
        print(f"Feature tensor shape: {final['features'].shape}")
        print(f"Saved to: {save_path}")

# Process all splits
for split in splits:
    process_split(split)

print("\nAll splits complete!")