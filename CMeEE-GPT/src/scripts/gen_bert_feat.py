import os
import json
import torch
import argparse
import numpy as np
from transformers import BertTokenizer, AutoTokenizer

from src.models.model import Bert
from src.utils.etqdm import etqdm
from src.utils.settings import get_json_data
from src.utils.const import EE_NUM_LABELS, EE_NUM_LABELS1, EE_NUM_LABELS2, CBLUE_ROOT


def get_model_with_tokenizer(model_args):
    model_type = model_args.model_type
    model_class = Bert

    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_args.model_path, max_len=512)
    elif model_type in ["roberta", "mcbert", "ehr_bert", "rocbert", "assemble_bert", "lstm", "bert_lstm"]:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, use_fast=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS)
    print(f"===> Load model from {model_args.model_path}")

    return model, tokenizer


def model_infer(text, model, tokenizer, device, max_len=512):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    return outputs  # (1, seq_len, 768)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bert")
    parser.add_argument("--model_path", type=str, default="./mcbert")
    parser.add_argument("--data_type", type=str, default="train")
    args = parser.parse_args()

    model, tokenizer = get_model_with_tokenizer(args)
    model = model.to(device).eval()

    train_data = get_json_data(args.data_type)
    res = {}
    for item in etqdm(train_data, desc="Generating BERT features"):
        text = item["text"]
        outputs = model_infer(text, model, tokenizer, device)  # (1, seq_len, 768)
        # remove all zero vectors in sequence
        outputs = outputs[0][outputs[0].sum(axis=1) != 0]
        # average pooling
        outputs = outputs.cpu().numpy().mean(axis=0)
        assert outputs.shape == (768,)
        res[text] = outputs.tolist()

    # save
    path = os.path.join(CBLUE_ROOT, "feat", f"{args.data_type}_bert_feat.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False)
