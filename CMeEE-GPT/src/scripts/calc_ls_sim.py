import json
from src.cluster.linear_search import LS_shots
from src.utils.settings import get_json_data

from src.utils.settings import get_json_data
import numpy as np
from src.utils.const import CHINESE_FILTER
from src.utils.etqdm import etqdm

class LS_shots:
    def __init__(self):
        self.data = get_json_data("train")
        self.text_list = []
        for i in range(len(self.data)):
            text = self.data[i]["text"]
            self.text_list.append(text)
        self.filter = CHINESE_FILTER
    
    def get_shots(self, text, nums):
        # 计算两个字符串重复的词的个数
        def get_repetitive_num(text1, text2):
            text1 = set(text1) - set(self.filter)
            text2 = set(text2) - set(self.filter)
            valid = text1 & text2
            score = len(valid)
            return valid, score
        
        score_array = np.zeros(len(self.data))
        inter = []
        for i in range(len(self.data)):
            if len(self.text_list[i]) > 120:
                continue
            intersection, score = get_repetitive_num(text, self.text_list[i])
            score_array[i] = score
            inter.append(intersection)
        
        # 选取前nums个最大的idx and text
        score_array_idx = np.argsort(score_array)[-nums:]
        # print(text)
        sim_text = []
        for i in score_array_idx:
            sim_text.append(self.data[i]["text"])
            # print(score_array[i], inter[i], self.data[i]["text"], )

        return sim_text

data = get_json_data("select_dev")
a = LS_shots()
sim_res = {}
for data_dict in etqdm(data):
    sim_list = a.get_shots(data_dict["text"], 8)
    sim_res[data_dict["text"]] = sim_list


with open("data_share/ls_sim_res.json", "w", encoding="utf-8") as f:
    json.dump(sim_res, f, ensure_ascii=False, indent=4)
