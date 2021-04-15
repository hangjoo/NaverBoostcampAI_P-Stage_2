import pickle as pkl
import numpy as np

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_df, tokenizer, max_len=312):
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.preprocessed_ = self.preprocessing()
        self.tokenized_, self.e_token_ids = self.tokenizing()

    def __getitem__(self, idx):
        encoded = {k: v[idx] for k, v in self.tokenized_.items()}
        e_token_ids = {k: torch.tensor(v[idx]) for k, v in self.e_token_ids.items()}
        label = torch.tensor(self.preprocessed_["label"][idx])

        return encoded, e_token_ids, label

    def __len__(self):
        return len(self.data_df)

    def preprocessing(self):
        sentence = self.data_df.iloc[:, 1].to_list()

        e1_name = self.data_df.iloc[:, 2].to_list()
        e1_si = [v - s[:v].count(" ") for v, s in zip(self.data_df.iloc[:, 3], self.data_df.iloc[:, 1])]
        e1_ei = [v - s[:v].count(" ") for v, s in zip(self.data_df.iloc[:, 4], self.data_df.iloc[:, 1])]

        e2_name = self.data_df.iloc[:, 5].to_list()
        e2_si = [v - s[:v].count(" ") for v, s in zip(self.data_df.iloc[:, 6], self.data_df.iloc[:, 1])]
        e2_ei = [v - s[:v].count(" ") for v, s in zip(self.data_df.iloc[:, 7], self.data_df.iloc[:, 1])]
        with open("data/label_type.pkl", "rb") as f:
            label_encoder = pkl.load(f)

        label = []
        for v in self.data_df[8]:
            if v == "blind":
                label.append(100)
            else:
                label.append(label_encoder[v])

        preprocessed_ = {
            "sentence": sentence,
            "e1_name": e1_name,
            "e1_si": e1_si,
            "e1_ei": e1_ei,
            "e2_name": e2_name,
            "e2_si": e2_si,
            "e2_ei": e2_ei,
            "label": label
        }

        return preprocessed_

    def tokenizing(self):
        tokenized_ = self.tokenizer(
            text=self.preprocessed_["sentence"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
        )
        e_token_ids = {}

        seq_size = tokenized_.input_ids.shape[-1]
        e1_token_ids = []
        e2_token_ids = []
        for sent_idx, sent in enumerate(self.preprocessed_["sentence"]):
            tokenized_sent = self.tokenizer.tokenize(sent)
            e1_idx_ = []
            e2_idx_ = []

            chr_count = 0
            for token_idx, word in enumerate(tokenized_sent):
                for word_idx in range(len(word) - word.count("#")):
                    if chr_count + word_idx in range(self.preprocessed_["e1_si"][sent_idx], self.preprocessed_["e1_ei"][sent_idx] + 1):
                        e1_idx_.append(token_idx)
                        break
                for word_idx in range(len(word) - word.count("#")):
                    if chr_count + word_idx in range(self.preprocessed_["e2_si"][sent_idx], self.preprocessed_["e2_ei"][sent_idx] + 1):
                        e2_idx_.append(token_idx)
                        break
                chr_count += len(word)
                chr_count -= word.count("#")

            e1_idx_ = [True if i in e1_idx_ else False for i in range(seq_size)]
            e2_idx_ = [True if i in e2_idx_ else False for i in range(seq_size)]

            e1_token_ids.append(e1_idx_)
            e2_token_ids.append(e2_idx_)

        e_token_ids["e1_token_ids"] = e1_token_ids
        e_token_ids["e2_token_ids"] = e2_token_ids

        return tokenized_, e_token_ids
