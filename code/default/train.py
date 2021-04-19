import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

import neptune.new as neptune
import neptune_config
from dataset import BaseDataset
# from dataset import TrainDataset
from creators import create_criterion, create_optimizer, ClassifierModel
from utils import fix_random_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fix_random_seed()


def train():
    # configs
    CONFIGS = {
        "MODEL": {
            "MODEL_TYPE": "Electra",
            "MODEL_NAME": "monologg/koelectra-base-v3-discriminator"
        },
        "DATA": {
            "DATA_VER": "train",
        },
        "SESSION": {
            "EPOCH_NUM": 10,
            "BATCH_SIZE": 64,
            "LEARNING_RATE": 1e-6,
            "INPUT_MAX_LEN": 288,

            "CRITERION_NAME": "LabelSmoothingLoss",
            "CRITERION_PARAMS": {},

            "OPTIMIZER_NAME": "MADGRAD",
            "OPTIMIZER_PARAMS": {},

            "SCHEDULER_NAME": "cosine_schedule_with_warmup",
            "SCHEDULER_PARAMS": {
                "WARMUP_RATIO": 0.01,
            },
        },
    }

    for c_k, c_v in CONFIGS.items():
        print(f"[SESSION LOG] {c_k} Configs:")
        for k, v in c_v.items():
            print(f"[SESSION LOG]\t {k}: {v}")

    # neptune
    neptune_run = neptune.init(project="hangjoo/PSTAGE-2", api_token=neptune_config.token,)
    neptune_run["CONFIGS"] = CONFIGS

    # save dir
    save_path = os.path.join("output", neptune_run["sys/id"].fetch())
    os.makedirs(save_path, exist_ok=True)
    print(f"[SESSION LOG] All record results would be saved in {save_path}.")

    # save configs to json file
    with open(os.path.join(save_path, "configs.json"), "w") as config_json:
        json.dump(CONFIGS, config_json, indent=4)

    # dataset
    print("[SESSION LOG] Generate training and validation data sets ... ", end="")
    tokenizer = AutoTokenizer.from_pretrained(CONFIGS["MODEL"]["MODEL_NAME"])
    tsv_path = os.path.join("data", "train", f"{CONFIGS['DATA']['DATA_VER']}.tsv")
    data_df = pd.read_csv(tsv_path, sep="\t", header=None)
    train_df, valid_df = train_test_split(data_df, train_size=0.9)

    train_set = BaseDataset(data_df=data_df, tokenizer=tokenizer, token_max_len=CONFIGS["SESSION"]["INPUT_MAX_LEN"])
    valid_set = BaseDataset(data_df=valid_df, tokenizer=tokenizer, token_max_len=CONFIGS["SESSION"]["INPUT_MAX_LEN"])

    # train_set = TrainDataset(data_df=train_df, tokenizer=tokenizer, token_max_len=CONFIGS["SESSION"]["INPUT_MAX_LEN"], max_label_size=250)
    # valid_set = BaseDataset(data_df=valid_df, tokenizer=tokenizer, token_max_len=CONFIGS["SESSION"]["INPUT_MAX_LEN"])

    train_loader = DataLoader(train_set, batch_size=CONFIGS["SESSION"]["BATCH_SIZE"], shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=CONFIGS["SESSION"]["BATCH_SIZE"], shuffle=True, num_workers=0, drop_last=True)
    print("done.")

    # model
    model = ClassifierModel(model_type=CONFIGS["MODEL"]["MODEL_TYPE"], model_name=CONFIGS["MODEL"]["MODEL_NAME"], dropout_rate=0.2).to(device)

    # criterion
    criterion = create_criterion(criterion_name=CONFIGS["SESSION"]["CRITERION_NAME"], **CONFIGS["SESSION"]["CRITERION_PARAMS"]).to(device)

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = create_optimizer(
        optimizer_name=CONFIGS["SESSION"]["OPTIMIZER_NAME"],
        params=optimizer_grouped_parameters,
        lr=CONFIGS["SESSION"]["LEARNING_RATE"],
        **CONFIGS["SESSION"]["OPTIMIZER_PARAMS"],
    )

    # scheduler
    t_total = len(train_loader) * CONFIGS["SESSION"]["EPOCH_NUM"]
    warmup_step = int(t_total * CONFIGS["SESSION"]["SCHEDULER_PARAMS"]["WARMUP_RATIO"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    # training session
    print("[SESSION LOG] Training Process starts ... ")
    best_loss = np.inf
    best_acc = 0
    best_f1_score = 0
    for epoch_idx in range(1, CONFIGS["SESSION"]["EPOCH_NUM"] + 1):
        time_ckpt = time.time()

        iter_train_loss = []
        iter_train_acc = []
        iter_train_f1_score = []

        iter_valid_loss = []
        iter_valid_acc = []
        iter_valid_f1_score = []

        model.train()
        for iter_idx, (encoded, label) in enumerate(train_loader, 1):
            encoded = {k: v.to(device) for k, v in encoded.items()}
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(**encoded)
            train_loss = criterion(outputs, label)
            pred_label = outputs.argmax(dim=-1)

            train_loss.backward()
            optimizer.step()
            scheduler.step()

            iter_train_loss.append(train_loss.item())
            iter_train_acc.extend((pred_label == label).cpu().tolist())

            iter_f1_score = f1_score(y_pred=pred_label.cpu().numpy(), y_true=label.cpu().numpy(), average="macro")
            iter_train_f1_score.append(iter_f1_score)

            print(
                f"[SESSION LOG] Epoch {epoch_idx}/{CONFIGS['SESSION']['EPOCH_NUM']} - model training iteration {iter_idx}/{len(train_loader)}     ",
                end="\r",
            )

        for iter_idx, (encoded, label) in enumerate(valid_loader, 1):
            with torch.no_grad():
                model.eval()

                encoded = {k: v.to(device) for k, v in encoded.items()}
                label = label.to(device)

                outputs = model(**encoded)
                valid_loss = criterion(outputs, label)
                pred_label = outputs.argmax(dim=-1)

                iter_valid_loss.append(valid_loss.item())
                iter_valid_acc.extend((pred_label == label).cpu().tolist())

                iter_f1_score = f1_score(y_pred=pred_label.cpu().numpy(), y_true=label.cpu().numpy(), average="macro")
                iter_valid_f1_score.append(iter_f1_score)

                print(
                    f"[SESSION LOG] Epoch {epoch_idx}/{CONFIGS['SESSION']['EPOCH_NUM']} - model validation iteration {iter_idx}/{len(valid_loader)}     ",
                    end="\r",
                )

        epoch_train_loss = np.mean(iter_train_loss)
        epoch_train_acc = np.mean(iter_train_acc) * 100
        epoch_train_f1_score = np.mean(iter_train_f1_score)

        epoch_valid_loss = np.mean(iter_valid_loss)
        epoch_valid_acc = np.mean(iter_valid_acc) * 100
        epoch_valid_f1_score = np.mean(iter_valid_f1_score)

        time_taken = time.time() - time_ckpt

        print(
            f"[SESSION LOG] Epoch {epoch_idx}/{CONFIGS['SESSION']['EPOCH_NUM']} - time taken : {time_taken:.2f}sec" + " " * 20 + "\n"
            f"[SESSION LOG]\t train loss : {epoch_train_loss:.4f} | train accuracy : {epoch_train_acc:.2f}% | train f1-score : {epoch_train_f1_score:.4f}\n"
            f"[SESSION LOG]\t valid loss : {epoch_valid_loss:.4f} | valid accuracy : {epoch_valid_acc:.2f}% | valid f1-score : {epoch_valid_f1_score:.4f}",
        )

        best_fits = []
        if epoch_valid_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(save_path, "best_loss_model.pth"))
            best_loss = epoch_valid_loss
            best_fits.append("loss")
        if epoch_valid_acc > best_acc:
            torch.save(model.state_dict(), os.path.join(save_path, "best_acc_model.pth"))
            best_acc = epoch_valid_acc
            best_fits.append("accuracy")
        if epoch_valid_f1_score > best_f1_score:
            torch.save(model.state_dict(), os.path.join(save_path, "best_f1-score_model.pth"))
            best_f1_score = epoch_valid_f1_score
            best_fits.append("f1-score")
        if best_fits:
            print(f"[SESSION LOG]\t Record best validation {'/'.join(best_fits)}. Model weight file saved.")

        neptune_run["Results/training loss"].log(value=epoch_train_loss, step=epoch_idx)
        neptune_run["Results/training accuracy"].log(value=epoch_train_acc, step=epoch_idx)
        neptune_run["Results/training f1-score"].log(value=epoch_train_f1_score, step=epoch_idx)

        neptune_run["Results/validation loss"].log(value=epoch_valid_loss, step=epoch_idx)
        neptune_run["Results/validation accuracy"].log(value=epoch_valid_acc, step=epoch_idx)
        neptune_run["Results/validation f1-score"].log(value=epoch_valid_f1_score, step=epoch_idx)

    neptune_run.stop()


if __name__ == "__main__":
    train()
