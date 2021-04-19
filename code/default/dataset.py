import pickle
import pandas as pd

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_df, tokenizer, token_max_len=256):
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.token_max_len = token_max_len

        self.preprocessed = self.preprocessing()
        self.tokenized_dataset = self.tokenizing()

    def __getitem__(self, idx):
        # Tokenizer's input sentence would have a form like "[CLS], entity_a's name, [SEP], entity_b's name, [SEP] sentence [SEP]".
        encoded = {k: v[idx] for k, v in self.tokenized_dataset.items()}
        label = torch.tensor(self.preprocessed["labels"][idx])

        return encoded, label

    def __len__(self) -> int:
        return len(self.data_df)

    def preprocessing(self):
        with open("data/label_type.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        label = []
        for v in self.data_df[8]:
            if v == "blind":
                label.append(100)
            else:
                label.append(label_encoder[v])
        preprocessed_dataset = {"sentence": self.data_df[1].to_numpy(), "entity_a": self.data_df[2].to_numpy(), "entity_b": self.data_df[5].to_numpy(), "labels": label}

        return preprocessed_dataset

    def tokenizing(self):
        concat_entity = [e_a + "[SEP]" + e_b for e_a, e_b in zip(self.preprocessed["entity_a"], self.preprocessed["entity_b"])]
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


class TrainDataset(Dataset):
    def __init__(self, data_df, tokenizer, token_max_len=256, max_label_size=300):
        self.tokenizer = tokenizer
        self.token_max_len = 256

        self.preprocessed_ = self.preprocessing(data_df, max_label_size)
        self.tokenized_ = self.tokenizing()

    def __getitem__(self, idx):
        # Tokenizer's input sentence would have a form like "[CLS], entity_a's name, [SEP], entity_b's name, [SEP] sentence [SEP]".
        encoded = {k: v[idx] for k, v in self.tokenized_.items()}
        label = torch.tensor(self.preprocessed_["labels"][idx])

        return encoded, label

    def __len__(self):
        return self.preprocessed_["sentence"].size

    def preprocessing(self, data_df, max_label_size):
        preprocessed_ = pd.DataFrame(columns=data_df.columns)

        label_num = len(data_df[8].value_counts())
        label_keys = [k for k in data_df[8].value_counts().keys()]

        for i in range(label_num):
            label_data_df = data_df.loc[data_df[8] == label_keys[i], :]
            if len(label_data_df) >= max_label_size:
                sample_df = label_data_df.sample(n=max_label_size)
            else:
                sample_df = label_data_df.sample(n=max_label_size, replace=True)
            preprocessed_ = pd.concat([preprocessed_, sample_df], ignore_index=True)

        with open("data/label_type.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        label = []
        for v in preprocessed_[8]:
            if v == "blind":
                label.append(100)
            else:
                label.append(label_encoder[v])

        preprocessed_ = {
            "sentence": preprocessed_[1].to_numpy(),
            "e1_name": preprocessed_[2].to_numpy(),
            "e1_idx": preprocessed_[[3, 4]].to_numpy(),
            "e2_name": preprocessed_[5].to_numpy(),
            "e2_idx": preprocessed_[[6, 7]].to_numpy(),
            "labels": label
        }

        return preprocessed_

    def tokenizing(self):
        concat_entity = [e_a + "[SEP]" + e_b for e_a, e_b in zip(self.preprocessed_["e1_name"], self.preprocessed_["e2_name"])]
        tokenized_sentences = self.tokenizer(
            text=concat_entity,
            text_pair=list(self.preprocessed_["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.token_max_len,
            add_special_tokens=True,
        )

        return tokenized_sentences
