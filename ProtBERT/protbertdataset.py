import torch
from torch.utils.data import Dataset

from transformers import BertModel, BertTokenizer
import re

class ProtBERTDataset(Dataset):
    def __init__(self, sequences, labels, max_length=512, model_name=BertModel.from_pretrained("Rostlab/prot_bert")):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Tokenize the sequence
        encoding = self.tokenizer(sequence, max_length=self.max_length, padding="max_length", truncation=True)

        return {
            'features': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }

    def collate_fn(self, batch):
        features = [item['features'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        features = torch.stack(features)
        attention_masks = torch.stack(attention_masks)

        return {
            'features': features,
            'attention_mask': attention_masks,
            'labels': torch.stack(labels)
        }
