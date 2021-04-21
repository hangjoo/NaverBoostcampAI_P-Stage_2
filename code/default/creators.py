from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, SGD
from madgrad import MADGRAD
from adamp import AdamP, SGDP
from transformers import AdamW


def create_model(*args, **kwargs):
    model_type = kwargs.get("model_type", None)
    if model_type is None:
        raise Exception("create_model must have model_name argument.")

    if model_type == "Bert":
        model_ft = BertClassifierModel(*args, **kwargs)
        return model_ft
    if model_type == "Electra":
        model_ft = ElectraClassifierModel(*args, **kwargs)
        return model_ft
    if model_type == "XLMRoberta":
        model_ft = XLMRobertaClassifierModel(*args, **kwargs)
        return model_ft
    else:
        raise Exception(f"{model_type} does not exist.")


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


class BertClassifierModel(nn.Module):
    def __init__(self, model_type, model_name, class_num=42, dropout_rate=0.2, embedding_size=None):
        super(BertClassifierModel, self).__init__()

        model_config = getattr(import_module("transformers"), model_type + "Config").from_pretrained(model_name)
        self.model_type = model_type
        # backbone.
        self.backbone = getattr(import_module("transformers"), model_type + "Model").from_pretrained(model_name)
        if embedding_size is not None:
            self.backbone.resize_token_embeddings(embedding_size)
        # classifier.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model_config.hidden_size, model_config.hidden_size),
            nn.BatchNorm1d(model_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(model_config.hidden_size, class_num)
        )

    def forward(self, *args, **kwargs):
        outputs = self.backbone(*args, **kwargs)

        cls_logits = outputs.pooler_output

        out = self.classifier(cls_logits)

        return out


class ElectraClassifierModel(nn.Module):
    def __init__(self, model_type, model_name, class_num=42, dropout_rate=0.2, embedding_size=None):
        super(ElectraClassifierModel, self).__init__()

        model_config = getattr(import_module("transformers"), model_type + "Config").from_pretrained(model_name)
        self.model_type = model_type
        # backbone.
        self.backbone = getattr(import_module("transformers"), model_type + "Model").from_pretrained(model_name)
        if embedding_size is not None:
            self.backbone.resize_token_embeddings(embedding_size)
        # classifier.
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(model_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(model_config.hidden_size, model_config.hidden_size),
            nn.BatchNorm1d(model_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(model_config.hidden_size, class_num)
        )

    def forward(self, *args, **kwargs):
        outputs = self.backbone(*args, **kwargs)

        cls_logits = outputs.last_hidden_state[:, 0, :]

        out = self.classifier(cls_logits)

        return out


class XLMRobertaClassifierModel(nn.Module):
    def __init__(self, model_type, model_name, class_num=42, dropout_rate=0.2, embedding_size=None):
        super(XLMRobertaClassifierModel, self).__init__()

        model_config = getattr(import_module("transformers"), model_type + "Config").from_pretrained(model_name)
        self.model_type = model_type
        # backbone.
        self.backbone = getattr(import_module("transformers"), model_type + "Model").from_pretrained(model_name)
        if embedding_size is not None:
            self.backbone.resize_token_embeddings(embedding_size)
        # classifier.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model_config.hidden_size, class_num)
        )

    def forward(self, *args, **kwargs):
        outputs = self.backbone(*args, **kwargs)

        cls_logits = outputs.pooler_output

        out = self.classifier(cls_logits)

        return out
