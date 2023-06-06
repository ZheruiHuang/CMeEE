"""
Calculate f1 score for the prediction results.

Usage:
    python src/scripts/calc_f1_score.py --pred <pred_json_file> --label <label_json_file>
"""
import argparse
import json


def main(args):
    with open(args.pred, "r", encoding="utf-8") as f:
        pred = json.load(f)
    with open(args.label, "r", encoding="utf-8") as f:
        label = json.load(f)
    assert len(pred) == len(label)
    pred_num, label_num, correct_num = 0, 0, 0
    for pred_data, label_data in zip(pred, label):
        pred_entities = pred_data["entities"]
        label_entities = label_data["entities"]
        pred_num += len(pred_entities)
        label_num += len(label_entities)
        for pred_entity in pred_entities:
            if pred_entity in label_entities:
                correct_num += 1
    f1_score = 2 * correct_num / (pred_num + label_num)
    print(f"f1_score: {f1_score}")
    return f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="The prediction json file")
    parser.add_argument("--label", type=str, default="data_share/select_dev.json", help="The label json file")
    args = parser.parse_args()
    main(args)
