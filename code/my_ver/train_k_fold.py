import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import wandb

import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from dataset import BaseDataset
from creators import ClassifierModel, create_criterion, create_optimizer
from utils import fix_random_seed, log_print, progress_sign

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fix_random_seed()


def main(args_mode=False):
    # configs
    CONFIGS = {
        "MODEL": {"MODEL_TYPE": "Electra", "MODEL_NAME": "monologg/koelectra-base-v3-discriminator"},
        "DATA": {"DATA_VER": "train", "ENTITY_TOKEN": ["E1", "E2"]},
        "SESSION": {
            "EPOCH_NUM": 10,
            "BATCH_SIZE": 64,
            "BACKBONE_LEARNING_RATE": 1e-6,
            "CLASSIFIER_LEARNING_RATE": 1e-3,
            "INPUT_MAX_LEN": 288,
            "K_FOLD_NUM": 5,
            "CRITERION_NAME": "LabelSmoothingLoss",
            "CRITERION_PARAMS": {},
            "OPTIMIZER_NAME": "AdamW",
            "OPTIMIZER_PARAMS": {},
            "SCHEDULER_NAME": "cosine_schedule_with_warmup",
            "SCHEDULER_PARAMS": {"WARMUP_RATIO": 0.01},
        },
    }
    for c_k, c_v in CONFIGS.items():
        log_print(f"{c_k} Configs:")
        for k, v in c_v.items():
            log_print(f"\t {k}: {v}")

    # save path
    save_i = 0
    while True:
        if not os.path.isdir(f"output/PSTAGE2-k_fold-{save_i}"):
            save_path = f"output/PSTAGE2-k_fold-{save_i}"
            break
        save_i += 1
    os.makedirs(save_path, exist_ok=True)

    # wandb
    wandb.init(project="PSTAGE-2-k_fold", name=f"PSTAGE-2-{save_i}")
    for c_v in CONFIGS.values():
        for k, v in c_v.items():
            wandb.config[k] = v

    # save configs to json file
    log_print(f"All record results would be saved in {save_path}.")
    with open(os.path.join(save_path, "configs.json"), "w") as config_json:
        json.dump(CONFIGS, config_json, indent=4)

    # dataset
    log_print("Generate training and validation data sets ... ", end="")
    tokenizer = AutoTokenizer.from_pretrained(CONFIGS["MODEL"]["MODEL_NAME"])
    tsv_path = os.path.join("data", "train", f"{CONFIGS['DATA']['DATA_VER']}.tsv")
    data_df = pd.read_csv(tsv_path, sep="\t", header=None)
    kf = KFold(n_splits=CONFIGS["SESSION"]["K_FOLD_NUM"], shuffle=True)
    data_loader = []
    for train_idx, valid_idx in kf.split(data_df):
        train_set = BaseDataset(
            data_df=data_df.iloc[train_idx, :],
            tokenizer=tokenizer,
            token_max_len=CONFIGS["SESSION"]["INPUT_MAX_LEN"],
            e_tokens=CONFIGS["DATA"]["ENTITY_TOKEN"],
        )
        valid_set = BaseDataset(
            data_df=data_df.iloc[valid_idx, :],
            tokenizer=tokenizer,
            token_max_len=CONFIGS["SESSION"]["INPUT_MAX_LEN"],
            e_tokens=CONFIGS["DATA"]["ENTITY_TOKEN"],
        )

        train_loader = DataLoader(train_set, batch_size=CONFIGS["SESSION"]["BATCH_SIZE"], shuffle=True, num_workers=4, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=CONFIGS["SESSION"]["BATCH_SIZE"], shuffle=True, num_workers=4, drop_last=True)

        data_loader.append({"train": train_loader, "valid": valid_loader})
    print("done.")

    # training session
    log_print("Training Process starts ... ")
    avg_result = {
        "Average_train_loss": [],
        "Average_train_acc": [],
        "Average_train_f1_score": [],
        "Average_valid_loss": [],
        "Average_valid_acc": [],
        "Average_valid_f1_score": [],
        "Average_time_taken": [],
    }
    for fold_idx in range(CONFIGS["SESSION"]["K_FOLD_NUM"]):
        # model
        model = ClassifierModel(model_type=CONFIGS["MODEL"]["MODEL_TYPE"], model_name=CONFIGS["MODEL"]["MODEL_NAME"], dropout_rate=0.2).to(device)
        model.backbone.resize_token_embeddings(len(train_set.tokenizer))

        # criterion
        criterion = create_criterion(criterion_name=CONFIGS["SESSION"]["CRITERION_NAME"], **CONFIGS["SESSION"]["CRITERION_PARAMS"]).to(device)

        # optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        backbone_parameters = [
            {
                "params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
                "lr": CONFIGS["SESSION"]["BACKBONE_LEARNING_RATE"],
            },
            {
                "params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": CONFIGS["SESSION"]["BACKBONE_LEARNING_RATE"],
            },
        ]
        classifier_parameters = [
            {
                "params": [p for p in model.connector.parameters()] + [p for p in model.classifier.parameters()],
                "weight_decay": 0.01,
                "lr": CONFIGS["SESSION"]["CLASSIFIER_LEARNING_RATE"],
            }
        ]
        optimizer = create_optimizer(
            optimizer_name=CONFIGS["SESSION"]["OPTIMIZER_NAME"],
            params=backbone_parameters + classifier_parameters,
            **CONFIGS["SESSION"]["OPTIMIZER_PARAMS"],
        )

        # scheduler
        t_total = len(train_loader) * CONFIGS["SESSION"]["EPOCH_NUM"]
        warmup_step = int(t_total * CONFIGS["SESSION"]["SCHEDULER_PARAMS"]["WARMUP_RATIO"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        best_loss = np.inf
        best_acc = 0
        best_f1_score = 0
        for epoch_idx in range(1, CONFIGS["SESSION"]["EPOCH_NUM"] + 1):
            result, model_state = train(model, criterion, optimizer, scheduler, data_loader[fold_idx])

            log_print(
                f"Epoch {epoch_idx}/{CONFIGS['SESSION']['EPOCH_NUM']} - Fold {fold_idx + 1}/{CONFIGS['SESSION']['K_FOLD_NUM']} - time taken : {result['time_taken']:.2f}\n"
                f"\t| train loss : {result['train_loss']:.4f} | train accuracy : {result['train_acc']:.2f}% | train f1-score : {result['train_f1_score']:.4f} |\n"
                f"\t| valid loss : {result['valid_loss']:.4f} | valid accuracy : {result['valid_acc']:.2f}% | valid f1-score : {result['valid_f1_score']:.4f} |"
            )
            best_fits = []
            if result["valid_loss"] < best_loss:
                torch.save(model_state, os.path.join(save_path, f"best_loss_{fold_idx}_fold_model.pth"))
                best_loss = result["valid_loss"]
                best_fits.append("loss")
            if result["valid_acc"] > best_acc:
                torch.save(model_state, os.path.join(save_path, f"best_acc_{fold_idx}_fold_model.pth"))
                best_acc = result["valid_acc"]
                best_fits.append("accuracy")
            if result["valid_f1_score"] > best_f1_score:
                torch.save(model_state, os.path.join(save_path, f"best_f1-score_{fold_idx}_fold_model.pth"))
                best_f1_score = result["valid_f1_score"]
                best_fits.append("f1-score")
            if best_fits:
                log_print(f"\t-> 🚀 Record best validation {'/'.join(best_fits)} 🚀 Model weight file saved.")

            for k, v in result.items():
                avg_result[f"Average_{k}"].append(v)

            for k, v in result.items():
                wandb.log({f"{fold_idx}-fold {k}": v})

    for k, v in avg_result.items():
        avg_result[k] = np.mean(v)

    log_print(
        f"Epoch {epoch_idx}/{CONFIGS['SESSION']['EPOCH_NUM']} - ** ⭐️ Average All Fold ⭐️ ** - time taken : {avg_result['Average_time_taken']:.2f}\n"
        f"\t| train loss : {avg_result['Average_train_loss']:.4f} | train accuracy : {avg_result['Average_train_acc']:.2f} | train f1-score : {avg_result['Average_train_f1_score']:.4f} |\n"
        f"\t| valid loss : {avg_result['Average_valid_loss']:.4f} | valid accuracy : {avg_result['Average_valid_acc']:.2f} | valid f1-score : {avg_result['Average_valid_f1_score']:.4f} |"
    )

    for k, v in avg_result.items():
        wandb.log({k: v})


def train(model, criterion, optimizer, scheduler, data_loader):
    time_ckpt = time.time()

    model.to(device)
    train_loader = data_loader["train"]
    valid_loader = data_loader["valid"]

    iter_train_loss = []
    iter_train_acc = []
    iter_train_f1_score = []

    iter_valid_loss = []
    iter_valid_acc = []
    iter_valid_f1_score = []

    model.train()
    for iter_idx, (encoded, e_token_mask, label) in enumerate(train_loader, 1):
        encoded = {k: v.to(device) for k, v in encoded.items()}
        e_token_mask = {k: v.to(device) for k, v in e_token_mask.items()}
        label = label.to(device)

        optimizer.zero_grad()
        outputs = model(e_token_mask=e_token_mask, **encoded)
        train_loss = criterion(outputs, label)
        pred_label = outputs.argmax(dim=-1)

        train_loss.backward()
        optimizer.step()
        scheduler.step()

        iter_train_loss.append(train_loss.item())
        iter_train_acc.extend((pred_label == label).cpu().tolist())

        iter_f1_score = f1_score(y_pred=pred_label.cpu().numpy(), y_true=label.cpu().numpy(), average="macro")
        iter_train_f1_score.append(iter_f1_score)

        log_print(f"train iteration {progress_sign[iter_idx % 4]} {iter_idx}/{len(train_loader)}" + " " * 10, end="\r")

    model.eval()
    with torch.no_grad():
        for iter_idx, (encoded, e_token_mask, label) in enumerate(valid_loader, 1):
            encoded = {k: v.to(device) for k, v in encoded.items()}
            e_token_mask = {k: v.to(device) for k, v in e_token_mask.items()}
            label = label.to(device)

            outputs = model(e_token_mask=e_token_mask, **encoded)
            valid_loss = criterion(outputs, label)
            pred_label = outputs.argmax(dim=-1)

            iter_valid_loss.append(valid_loss.item())
            iter_valid_acc.extend((pred_label == label).cpu().tolist())

            iter_f1_score = f1_score(y_pred=pred_label.cpu().numpy(), y_true=label.cpu().numpy(), average="macro")
            iter_valid_f1_score.append(iter_f1_score)

            log_print(f"train iteration {progress_sign[iter_idx % 4]} {iter_idx}/{len(valid_loader)}" + " " * 10, end="\r")

    epoch_train_loss = np.mean(iter_train_loss)
    epoch_train_acc = np.mean(iter_train_acc) * 100
    epoch_train_f1_score = np.mean(iter_train_f1_score)

    epoch_valid_loss = np.mean(iter_valid_loss)
    epoch_valid_acc = np.mean(iter_valid_acc) * 100
    epoch_valid_f1_score = np.mean(iter_valid_f1_score)

    time_taken = time.time() - time_ckpt

    result = {
        "train_loss": epoch_train_loss,
        "train_acc": epoch_train_acc,
        "train_f1_score": epoch_train_f1_score,
        "valid_loss": epoch_valid_loss,
        "valid_acc": epoch_valid_acc,
        "valid_f1_score": epoch_valid_f1_score,
        "time_taken": time_taken,
    }
    model_state = model.state_dict()

    return result, model_state


if __name__ == "__main__":
    main()
