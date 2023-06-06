"""
Download a model from HuggingFace model hub.

Usage:
    python download_model.py -d {download_path} -m {model_name} -c {cache_dir}

The downloaded model will be saved in {download_path} and cached in {cache_dir}.
Default cache_dir is ~/.cache/huggingface/transformers.
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--download_path", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-c", "--cache_dir", type=str, default=None)
    args = parser.parse_args()

    download_path = args.download_path
    model_name = args.model_name

    # download model in main stream
    snapshot_download(repo_id=args.model_name, local_dir=download_path, local_dir_use_symlinks=False, cache_dir=args.cache_dir)
    print(f"Loaded model: {model_name}")

    model = AutoModelForMaskedLM.from_pretrained(download_path)
    tokenizer = AutoTokenizer.from_pretrained(download_path)

    # prepare input
    text = ["我爱北京天安门", "今天天气不错"]
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # forward pass
    output = model(**encoded_input)
    print(output.keys())
    print(output["logits"].shape)
