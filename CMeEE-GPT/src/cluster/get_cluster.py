from src.cluster.random_choose import Random_shots
from src.cluster.bert_feat import BertShots
from src.cluster.backup.bag_of_word import Bag_shots
from src.cluster.linear_search import LS_shots

def get_cluster(mode="random"):
    if mode == "random":
        return Random_shots()
    elif mode == "bert":
        return BertShots()
    elif mode == 'bag':
        return Bag_shots()
    elif mode == 'ls':
        return LS_shots()
    else:
        raise ValueError(f"Unknown cluster mode: {mode}")
