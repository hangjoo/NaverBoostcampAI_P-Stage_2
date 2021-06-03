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
