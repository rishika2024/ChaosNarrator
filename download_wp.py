import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class WritingPromptsDataset(Dataset):
    def __init__(self, features_path, max_len=200, tokenizer_name="gpt2"):
        data = torch.load(features_path)
        self.stories = data['stories']
        self.prompts = data['prompts']
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = max_len

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        # No images in Stage 2, use dummy zeros
        # 5 images of 768-dim each, all zeros
        dummy_image_feat = torch.zeros(5, 768)

        # Tokenize the story
        tokens = self.tokenizer(
            self.stories[idx],
            truncation=True,
            max_length=self.max_len - 5,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

        return dummy_image_feat, input_tokens, target_tokens