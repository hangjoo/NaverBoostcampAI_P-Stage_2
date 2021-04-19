
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, SGD
from madgrad import MADGRAD
from adamp import AdamP, SGDP
from transformers import AdamW

from transformers import ElectraConfig, ElectraForSequenceClassification
from transformers import BertConfig, BertForSequenceClassification


def create_model(model_name):
    if "electra" in model_name.lower():
        electra_config = ElectraConfig.from_pretrained(model_name)
        electra_config.num_labels = 42
        model_ft = ElectraForSequenceClassification(electra_config)

        return model_ft

    elif "bert" in model_name.lower():
        bert_config = BertConfig.from_pretrained(model_name)
        bert_config.num_labels = 42
        model_ft = BertForSequenceClassification(bert_config)

        return model_ft


def create_criterion(criterion_name, *args, **kwargs):
    if criterion_name == "CrossEntropyError":
        criterion = nn.CrossEntropyLoss(*args, **kwargs)
    elif criterion_name == "MSE":
        criterion = nn.MSELoss(*args, **kwargs)
    elif criterion_name == "FocalLoss":
        criterion = FocalLoss(*args, **kwargs)
    elif criterion_name == "KLDiv":
        criterion = nn.KLDivLoss(*args, **kwargs)
    elif criterion_name == "LabelSmoothingLoss":
        criterion = LabelSmoothingLoss(*args, **kwargs)
    else:
        raise Exception(f"{criterion_name} does not exist in criterion_list.")

    return criterion


def create_optimizer(optimizer_name, **kwargs):
    if optimizer_name == "Adam":
        optimizer = Adam(**kwargs)
    elif optimizer_name == "SGD":
        optimizer = SGD(**kwargs)
    elif optimizer_name == "MADGRAD":
        optimizer = MADGRAD(**kwargs)
    elif optimizer_name == "AdamP":
        optimizer = AdamP(**kwargs)
    elif optimizer_name == "SGDP":
        optimizer = SGDP(**kwargs)
    elif optimizer_name == "AdamW":
        optimizer = AdamW(**kwargs)
    else:
        raise Exception(f"{optimizer_name} does not exist in optimizer_list.")

    return optimizer


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(((1 - prob) ** self.gamma) * log_prob, target_tensor, weight=self.weight, reduction=self.reduction)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=42, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class ClassifierModel(nn.Module):
    def __init__(self, model_type, model_name, class_num=42, fc_size=256, dropout_rate=None):
        super(ClassifierModel, self).__init__()

        model_config = getattr(import_module("transformers"), model_type + "Config").from_pretrained(model_name)
        self.model_type = model_type
        # backbone.
        self.backbone = getattr(import_module("transformers"), model_type + "Model").from_pretrained(model_name)
        # flatten
        self.flatten = nn.Flatten(0, -1)
        # connector
        self.connector = nn.Linear(model_config.hidden_size, fc_size)
        # classifier.
        self.classifier = nn.Linear(fc_size * 3, class_num)
        # dropout
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else None
        # activation
        self.tanh = nn.Tanh()

    def forward(self, e_token_mask, **kwargs):
        # Reference : https://github.com/monologg/R-BERT
        e1_token_mask = e_token_mask["e1_token_mask"]
        e2_token_mask = e_token_mask["e2_token_mask"]

        if e1_token_mask is None or e2_token_mask is None:
            raise Exception("ERROR! Model must be feed e1_token_ids, e2_token_ids")

        outputs = self.backbone(**kwargs).last_hidden_state  # (batch_size, max_len, hidden_size)

        cls_output = outputs[:, 0, :]
        e1_output = torch.sum(outputs * e1_token_mask.unsqueeze(-1), dim=1) / torch.sum(e1_token_mask, dim=1, keepdim=True)
        e2_output = torch.sum(outputs * e2_token_mask.unsqueeze(-1), dim=1) / torch.sum(e2_token_mask, dim=1, keepdim=True)
        if self.dropout:
            cls_output = self.dropout(cls_output)
            e1_output = self.dropout(e1_output)
            e2_output = self.dropout(e2_output)
        cls_output = self.connector(self.tanh(cls_output))
        e1_output = self.connector(self.tanh(e1_output))
        e2_output = self.connector(self.tanh(e2_output))

        combine_output = torch.cat([cls_output, e1_output, e2_output], dim=-1)
        if self.dropout:
            combine_output = self.dropout(combine_output)

        out = self.classifier(combine_output)

        return out
