"""
Postprocess the json file to add entities that are not annotated in the original data.

Usage:
    python src/datasets/postprocess.py --src <src_json_file> [--output <output_json_file>] [--train_data <train_data_json_file>] [[--grid-search --label <label_json_file>] | [--min-occurrence <min_occurrence>] [--min-length <min_length>]] [--ignore-original-entities]
"""
from typing import List
import os.path as osp
import json
import argparse
import numpy as np
from tqdm import tqdm
import jieba
from copy import deepcopy
from calc_f1_score import main as calc_f1_score


def build_entities_dictionary(args):
    entities_dictionary = set()
    entities_occurrence_dict = {}
    entities_types_dict = {}
    with open(args.train_data, "r") as f:
        train_data = json.load(f)
    for data in train_data:
        for entity in data["entities"]:
            entity_name = entity["entity"]
            entities_dictionary.add(entity_name)
            entities_occurrence_dict[entity_name] = entities_occurrence_dict.get(entity_name, 0) + 1
            if entity_name not in entities_types_dict:
                entities_types_dict[entity_name] = {}
            entities_types_dict[entity_name][entity["type"]] = entities_types_dict[entity_name].get(entity["type"], 0) + 1
    
    entities_type_dict = {}
    for entity_name, types in entities_types_dict.items():
        entities_type_dict[entity_name] = max(types.keys(), key=lambda x: types[x])
    
    entities_dictionary = [entity for entity in entities_dictionary \
        if entities_occurrence_dict[entity] >= args.min_occurrence and len(entity) >= args.min_length]
    return entities_dictionary, entities_type_dict


def get_char_start_idx_list(cut_text: List[str]) -> List[int]:
    char_start_idx_list = [0]
    for token in cut_text:
        char_start_idx_list.append(char_start_idx_list[-1] + len(token))
    return char_start_idx_list


def find_entities(entities_dictionary: List[str], cut_text: List[str], char_start_idx_list: List[int], flag: np.ndarray):
    for entity in entities_dictionary:
        cur_idx = 0
        while entity in cut_text[cur_idx:]:
            cur_idx = cut_text.index(entity, cur_idx)
            start_idx = char_start_idx_list[cur_idx]
            end_idx = start_idx + len(entity)
            if not flag[start_idx:end_idx].all():
                yield entity, start_idx, end_idx
                flag[start_idx:end_idx] = True
            cur_idx += 1


def main(args):
    entities_dictionary, entities_type_dict = build_entities_dictionary(args)
    entities_dictionary = list(entities_dictionary)
    entities_dictionary.sort(key=lambda x: len(x), reverse=True)
    new_json = []
    for entity in entities_dictionary:
        jieba.add_word(entity)

    with open(args.src, "r") as f:
        src_data = json.load(f)
    for data in tqdm(src_data):
        text: str = data["text"]
        cut_text = jieba.lcut(text)
        flag = np.zeros(len(text), dtype=bool)
        if args.ignore_original_entities:
            entities = []
        else:
            entities: list = deepcopy(data.get("entities", []))
        for entity in (data.get("entities", []) if not args.ignore_original_entities else []):
            flag[entity["start_idx"]:entity["end_idx"]+1] = True
        for entity, start_idx, end_idx in find_entities(entities_dictionary, cut_text, get_char_start_idx_list(cut_text), flag):
            entities.append({
                "entity": entity,
                "start_idx": start_idx,
                "end_idx": end_idx-1,
                "type": entities_type_dict[entity]
            })
        entities.sort(key=lambda x: x["start_idx"])
        new_json.append({
            "text": text,
            "entities": entities
        })
    
    with open(args.output, "w") as f:
        json.dump(new_json, f, indent=2, ensure_ascii=False)


def grid_search(args):
    from tabulate import tabulate
    class CalcF1ScoreArgs:
        def __init__(self):
            self.pred = args.output
            self.label = args.label
    calc_f1_score_args = CalcF1ScoreArgs()
    
    print("Grid search...")
    best_min_occ, best_min_len, best_f1 = 0, 0, -1
    table = []
    for min_occurrence in list(range(1, 5+1)) + list(range(6, 20+1, 2)):
        for min_length in range(1, 6+1):
            args.min_length = min_length
            args.min_occurrence = min_occurrence
            main(args)
            f1_score = calc_f1_score(calc_f1_score_args)
            if f1_score > best_f1:
                best_min_occ, best_min_len, best_f1 = min_occurrence, min_length, f1_score
            table.append([min_occurrence, min_length, round(f1_score, 4)])
    args.min_occurrence = best_min_occ
    args.min_length = best_min_len
    main(args)
    table.sort(key=lambda x: x[-1], reverse=True)
    print(tabulate(table[:20], headers=["min_occ", "min_len", "f1"], tablefmt="grid"))
    print(f"Save the best result to {args.output}")


if __name__ == "__main__":
    default_train_data = osp.join(osp.dirname(__file__), "../../data/CBLUEDatasets/CMeEE/CMeEE_train.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True,
        help="The source json file to be processed")
    parser.add_argument("--output", type=str, default=None,
        help="The output json file. Default is the same as src with suffix '_postprocessed.json'")
    parser.add_argument("--train_data", type=str, default=default_train_data,
        help="The train data to build entities dictionary. Default is CMeEE_train.json")
    parser.add_argument("--ignore-original-entities", action="store_true", default=False,
        help="Whether to ignore the original entities. Default is False")
    parser.add_argument("--min-occurrence", type=int, default=1,
        help="The minimum occurrence of entities to be added. Default is 1")
    parser.add_argument("--min-length", type=int, default=1,
        help="The minimum length of entities to be added. Default is 1")
    parser.add_argument("--grid-search", action="store_true", default=False,
        help="Whether to perform grid search. Default is False")
    parser.add_argument("--label", type=str, default=None,
        help="The label json file to calculate f1 score. Default is the same as src")
    args = parser.parse_args()
    if args.output is None:
        args.output = osp.splitext(args.src)[0] + "_postprocessed.json"
    if args.label is None:
        args.label = deepcopy(args.src)
    
    if args.grid_search:  # perform grid search to find the best min_occurrence and min_length.
        grid_search(args)
    else:
        main(args)