import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from load_data import load_data, tokenized_dataset, RE_Dataset

from utils import fix_random_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fix_random_seed(random_seed=42)


# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


def train():
    # load model and tokenizer
    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    data_df = pd.read_csv("data/train/train.tsv", sep="\t", header=None)
    train_df, test_df = train_test_split(data_df, train_size=0.8, shuffle=True, random_state=42)
    train_dataset = load_data(train_df)
    valid_dataset = load_data(test_df)
    train_label = train_dataset["label"].values
    valid_label = valid_dataset['label'].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(valid_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

    # setting model hyperparameter
    bert_config = BertConfig.from_pretrained(MODEL_NAME)
    bert_config.num_labels = 42
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir="./results",
        save_total_limit=3,
        save_steps=500,
        num_train_epochs=10,
        learning_rate=5e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_steps=500
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RE_train_dataset,
        eval_dataset=RE_valid_dataset,
        compute_metrics=compute_metrics
    )

    # train model
    trainer.train()


def main():
    train()


if __name__ == "__main__":
    main()
