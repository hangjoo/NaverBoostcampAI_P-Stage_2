import pickle as pkl
import numpy as np

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_df, tokenizer, token_max_len=312, e_tokens=["E1, E2"]):
        self.data_df = data_df
        self.tokenizer = self.init_tokenizer(tokenizer, e_tokens)
        self.token_max_len = token_max_len

        assert len(e_tokens) == 2, "Entity special token nums must be 2."
        self.e1_st_id, self.e1_et_id = self.get_e_token_ids(e_tokens[0])
        self.e2_st_id, self.e2_et_id = self.get_e_token_ids(e_tokens[1])

        self.preprocessed_ = self.preprocessing()
        self.tokenized_, self.e_token_mask = self.tokenizing()

    def __getitem__(self, idx):
        encoded = {k: v[idx] for k, v in self.tokenized_.items()}
        e_token_mask = {k: torch.tensor(v[idx]) for k, v in self.e_token_mask.items()}
        label = torch.tensor(self.preprocessed_["label"][idx])

        return encoded, e_token_mask, label

    def __len__(self):
        return len(self.data_df)

    def preprocessing(self):
        sentence = self.data_df.iloc[:, 1].to_numpy()

        e1_name = self.data_df.iloc[:, 2].to_numpy()
        e1_si = self.data_df.iloc[:, 3].to_numpy()
        e1_ei = self.data_df.iloc[:, 4].to_numpy()

        e2_name = self.data_df.iloc[:, 5].to_numpy()
        e2_si = self.data_df.iloc[:, 6].to_numpy()
        e2_ei = self.data_df.iloc[:, 7].to_numpy()

        for idx in range(len(sentence)):
            if e1_si[idx] < e2_si[idx]:
                prev_e_id = 1
                prev_e_si = e1_si[idx]
                prev_e_ei = e1_ei[idx]

                next_e_id = 2
                next_e_si = e2_si[idx]
                next_e_ei = e2_ei[idx]
            else:
                prev_e_id = 2
                prev_e_si = e2_si[idx]
                prev_e_ei = e2_ei[idx]

                next_e_id = 1
                next_e_si = e1_si[idx]
                next_e_ei = e1_ei[idx]

            # wrap entities by entity tokens like "... [E1]entity1[/E1] ... [E2]entity2[/E2] ..."
            sentence[idx] = sentence[idx][:prev_e_si] + \
                f"[E{prev_e_id}]" + sentence[idx][prev_e_si:prev_e_ei + 1] + f"[/E{prev_e_id}]" + \
                sentence[idx][prev_e_ei + 1:next_e_si] + \
                f"[E{next_e_id}]" + sentence[idx][next_e_si:next_e_ei + 1] + f"[/E{next_e_id}]" + \
                sentence[idx][next_e_ei + 1:]

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
            text=list(self.preprocessed_["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.token_max_len,
            add_special_tokens=True,
        )

        e1_mask = []
        e2_mask = []
        e1_flag = False
        e2_flag = False
        for input_ids in tokenized_.input_ids:
            e1_mask_ = []
            e2_mask_ = []
            for encoded in input_ids:
                if encoded == self.e1_st_id:
                    e1_flag = True
                if encoded == self.e1_et_id:
                    e1_flag = False
                if encoded == self.e2_st_id:
                    e2_flag = True
                if encoded == self.e2_et_id:
                    e2_flag = False
                e1_mask_.append(e1_flag)
                e2_mask_.append(e2_flag)
            e1_mask.append(e1_mask_)
            e2_mask.append(e2_mask_)

        e_token_mask = {"e1_token_mask": e1_mask, "e2_token_mask": e2_mask}

        return tokenized_, e_token_mask

    def init_tokenizer(self, tokenizer, e_tokens):
        e1_st = "[" + e_tokens[0] + "]"
        e1_et = "[/" + e_tokens[0] + "]"
        e2_st = "[" + e_tokens[1] + "]"
        e2_et = "[/" + e_tokens[1] + "]"

        e_tokens_dict = {"additional_special_tokens": [e1_st, e1_et, e2_st, e2_et]}
        tokenizer.add_special_tokens(e_tokens_dict)

        return tokenizer

    def get_e_token_ids(self, e_token):
        e_st = "[" + e_token + "]"
        e_et = "[/" + e_token + "]"

        e_st_id = self.tokenizer.encode(e_st)[1]
        e_et_id = self.tokenizer.encode(e_et)[1]

        return e_st_id, e_et_id


class BaseDataset_BACKUP(Dataset):
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


if __name__ == "__main__":
    import os
    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    tsv_path = os.path.join("data", "train", "train.tsv")
    data_df = pd.read_csv(tsv_path, sep="\t", header=None)

    data_set = BaseDataset(data_df=data_df, tokenizer=tokenizer, token_max_len=288, e_tokens=["E1", "E2"])
    print(data_set.preprocessed_["sentence"])
