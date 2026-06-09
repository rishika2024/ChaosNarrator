import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class VISTDataset(Dataset):
    """PyTorch Dataset for the VIST dataset — loads CLIP image features and tokenizes story text."""

    def __init__(self, features_path, max_len=100, tokenizer_name="gpt2"):
        """
        Args:
            features_path: path to a .pt file with keys 'features', 'stories', 'story_ids'.
            max_len: maximum token sequence length (image tokens are reserved separately).
            tokenizer_name: HuggingFace tokenizer identifier, defaults to GPT-2.
        """
        if not os.path.exists(features_path):
            os.makedirs(os.path.dirname(features_path) or '.', exist_ok=True)
            torch.save({'features': torch.zeros(0, 5, 768), 'stories': [], 'story_ids': []}, features_path)
            print(f"Created empty dataset file: {features_path}")
        data = torch.load(features_path) # {'features': ..., 'stories': ..., 'story_ids': ...}
        self.clip_vectors = data['features']   # (num_stories, 5, 768)
        self.stories = data['stories']
        self.story_ids = data['story_ids'] 
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) # loading gpt2 tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token # set pad token to eos for GPT-2
        self.max_len = max_len

    def __len__(self):
        """Returns the number of stories in the dataset."""
        # how many stories do we have in this dataset?
        return len(self.stories)

    def __getitem__(self, idx):
        """Returns (clip_vectors, input_tokens, target_tokens) for one story."""
        # Get CLIP features for this story
        clip_vectors = self.clip_vectors[idx]  # (5, 768)

        # Tokenize the story
        tokens = self.tokenizer(
            self.stories[idx], # the text of the story
            truncation=True, # truncate if story is too long
            max_length=self.max_len - 5, # reserve space for image tokens
            return_tensors="pt"
        ).input_ids.squeeze(0)  # (seq_len,) remove batch dimension

        # Input is everything except the last token
        input_tokens = tokens[:-1]

        # Target is everything except the first token
        target_tokens = tokens[1:]

        return clip_vectors, input_tokens, target_tokens