import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from dataset import BaseDataset
from creators import ClassifierModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval():
    exp_name = "PSTAGE2-k_fold-0"
    epoch_idx = "7"

    save_path = os.path.join("output", exp_name)
    with open(os.path.join(save_path, "configs.json"), "r") as config_file:
        config = json.load(config_file)

    print("Preparing data set for testing ... ", end="")
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL"]["MODEL_NAME"])
    test_path = os.path.join("data", "test", "test.tsv")
    test_df = pd.read_csv(test_path, sep="\t", header=None)
    test_set = BaseDataset(data_df=test_df, tokenizer=tokenizer, token_max_len=config["SESSION"]["INPUT_MAX_LEN"])
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    print("done.")

    print("Inference test starts ...")
    output_logits = []
    for fold_idx in range(config["SESSION"]["K_FOLD_NUM"]):
        if fold_idx == 1:
            continue
        model = ClassifierModel(model_type=config["MODEL"]["MODEL_TYPE"], model_name=config["MODEL"]["MODEL_NAME"], dropout_rate=0.2).to(device)
        model.load_state_dict(torch.load(os.path.join(save_path, f"epoch-{epoch_idx}_fold-{fold_idx}_model.pth")))
        model.eval()

        fold_logits = []
        for idx, (encoded, _) in enumerate(test_loader):
            with torch.no_grad():
                encoded = {k: v.to(device) for k, v in encoded.items()}

                outputs = model(**encoded)
                fold_logits.extend(outputs.cpu().tolist())

            print(f"[fold idx {fold_idx + 1}/{config['SESSION']['K_FOLD_NUM']}] - [iter idx {idx + 1}/{len(test_loader)}]", end="\r")

        output_logits.append(fold_logits)
    print()

    output_logits = np.sum(output_logits, axis=0)
    output_pred = np.argmax(output_logits, axis=-1)

    submission = pd.DataFrame(output_pred, columns=["pred"])
    submission.to_csv(os.path.join(save_path, f"{exp_name}_epoch-{epoch_idx}_submisson.csv"), index=False)


if __name__ == "__main__":
    eval()
