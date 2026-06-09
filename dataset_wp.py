import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class WritingPromptsDataset(Dataset):
    # how to load the stories and prompts
    def __init__(self, features_path, max_len=200, tokenizer_name="gpt2"):
        data = torch.load(features_path) # {'stories': ..., 'prompts': ...} loaded from saved dataset downloaded in download_wp.py
        self.stories = data['stories'] # list of story texts
        self.prompts = data['prompts'] # list of prompt texts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) # loading gpt2 tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token # set pad token to eos for GPT-2
        self.max_len = max_len

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        # Since writing prompts doesn't have associated images
        # we will returning dummy image feature vector of zero
        # total 5 images of 768-dim each, all zeros
        dummy_image_feat = torch.zeros(5, 768)

        # Extracting a few keywords from the prompt
        # Remove common filler words and take first 5 meaningful words
        prompt = self.prompts[idx]

        # Cleaning up the prompt: remove [WP], [EU], etc.
        prompt = prompt.replace('[ wp ]', '').replace('[ eu ]', '').replace('[ tt ]', '')
        prompt = prompt.replace('``', '').replace("''", '').strip()

        # Take first few words as keywords
        words = prompt.split()[:8]
        keywords = ' '.join(words)

        # Combine keywords + story with a separator
        combined = keywords + " | " + self.stories[idx]

        # Tokenizing the story
        tokens = self.tokenizer( 
            combined, # the text to tokenize (keywords + story)            
            truncation=True, # truncate if story is too long
            max_length=self.max_len - 5, # reserve space for image tokens
            return_tensors="pt" # return PyTorch tensors
        ).input_ids.squeeze(0) # (seq_len,) remove batch dimension

        # Input is everything except the last token
        input_tokens = tokens[:-1]

        # Target is everything except the first token
        target_tokens = tokens[1:]

        return dummy_image_feat, input_tokens, target_tokens