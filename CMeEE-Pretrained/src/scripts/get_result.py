import os
import json
import argparse
from tabulate import tabulate
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="../ckpts")
    args = parser.parse_args()

    # find all trainer_state.json and get the "best_metric"
    # best_metric_dict = {best_model_checkpoint: best_metric}
    ckpt_dir = args.ckpt_dir
    best_metric_dict = defaultdict(list)
    for root, dirs, files in os.walk(ckpt_dir):
        for file in files:
            if file == "trainer_state.json":
                with open(os.path.join(root, file), "r") as f:
                    trainer_state = json.load(f)
                key = root.split("/")[-2]
                best_metric_dict[key].append(trainer_state["best_metric"])
    # show the best_metric_dict in a table
    header = ["model", "best_metric"]
    table = []
    for key, value in best_metric_dict.items():
        table.append([key, max(value)])
    print("\n========= Raw =========")
    # align the table by left
    table.sort(key=lambda x: x[0])
    print(tabulate(table, headers=header, tablefmt="pretty", colalign=("left", "left")))
    # print(tabulate(table, headers=header, tablefmt="latex_raw", colalign=("left", "left")))

    # sort the table by best_metric
    print("\n========= Sorted by best_metric =========")
    table.sort(key=lambda x: x[1], reverse=True)
    print(tabulate(table, headers=header, tablefmt="pretty", colalign=("left", "left")))


if __name__ == "__main__":
    main()
