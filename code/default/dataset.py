import pickle
import pandas as pd

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_df, tokenizer, token_max_len=256, e_token=None):
        self.data_df = data_df
        self.e_token = e_token
        self.tokenizer = self.init_tokenizer(tokenizer, e_token)
        self.token_max_len = token_max_len

        self.preprocessed = self.preprocessing()
        self.tokenized_dataset = self.tokenizing()

    def __getitem__(self, idx):
        encoded = {k: v[idx] for k, v in self.tokenized_dataset.items()}
        label = torch.tensor(self.preprocessed["labels"][idx])

        return encoded, label

    def __len__(self) -> int:
        return len(self.data_df)

    def init_tokenizer(self, tokenizer, e_token):
        if e_token is not None:
            e_token_dict = {"additional_special_tokens": [e_token]}
            tokenizer.add_special_tokens(e_token_dict)

        return tokenizer

    def preprocessing(self):
        with open("data/label_type.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        label = []
        for v in self.data_df[8]:
            if v == "blind":
                label.append(100)
            else:
                label.append(label_encoder[v])
        preprocessed_dataset = {
            "sentence": self.data_df[1].to_numpy(),
            "entity_a": self.data_df[2].to_numpy(),
            "entity_b": self.data_df[5].to_numpy(),
            "labels": label,
        }

        return preprocessed_dataset

    def tokenizing(self):
        if self.e_token is not None:
            concat_entity = [
                e_a + self.e_token + e_b + self.e_token + sent
                for e_a, e_b, sent in zip(self.preprocessed["entity_a"], self.preprocessed["entity_b"], list(self.preprocessed["sentence"]))
            ]
            tokenized_sentences = self.tokenizer(
                text=concat_entity,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.token_max_len,
                add_special_tokens=True,
            )
        else:
            concat_entity = [e_a + self.tokenizer.sep_token + e_b for e_a, e_b in zip(self.preprocessed["entity_a"], self.preprocessed["entity_b"])]
            tokenized_sentences = self.tokenizer(
                text=concat_entity,
                text_pair=list(self.preprocessed["sentence"]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.token_max_len,
                add_special_tokens=True,
            )

        return tokenized_sentences
