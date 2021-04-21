import os
import json
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from dataset import BaseDataset
from creators import create_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval():
    exp_name = "PSTAGE2-default-8"
    weight_type = "f1-score"

    save_path = os.path.join("output", exp_name)
    with open(os.path.join(save_path, "configs.json"), "r") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(config["MODEL"]["MODEL_NAME"])
    test_path = os.path.join("data", "test", "test.tsv")
    test_df = pd.read_csv(test_path, sep="\t", header=None)
    test_set = BaseDataset(data_df=test_df, tokenizer=tokenizer, token_max_len=config["SESSION"]["INPUT_MAX_LEN"], e_token=config["DATA"]["ENTITY_TOKEN"])
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = create_model(model_type=config["MODEL"]["MODEL_TYPE"], model_name=config["MODEL"]["MODEL_NAME"], dropout_rate=0.2, embedding_size=len(test_set.tokenizer)).to(device)
    model.load_state_dict(torch.load(os.path.join(save_path, f"best_{weight_type}_model.pth")))
    model.eval()

    output_pred = []

    for idx, (encoded, _) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            pred_label = outputs.argmax(dim=-1)

            output_pred.extend(pred_label.cpu().tolist())

    submission = pd.DataFrame(output_pred, columns=["pred"])
    submission.to_csv(os.path.join(save_path, f"{exp_name}_{weight_type}_submisson.csv"), index=False)


if __name__ == "__main__":
    eval()
