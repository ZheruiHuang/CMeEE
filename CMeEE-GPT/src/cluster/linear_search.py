from src.utils.settings import get_json_data
import numpy as np
from src.utils.const import CHINESE_FILTER
from src.utils.etqdm import etqdm

class LS_shots:
    def __init__(self):
        self.data = get_json_data("ls_sim")
        self.train_data = get_json_data("train")

    def get_shots(self, text, nums):
        if text in self.data:
            data = self.data[text]
            data = [item for item in self.train_data if item["text"] in data]
        else:
            data = self.train_data

        data_list = np.random.choice(data, nums, replace=False)

        assert len(data_list) == nums, "len(data_list) != nums"
        return data_list

if __name__ == "__main__":
    dev_data = get_json_data("select_dev")
    dev_text = [item["text"] for item in dev_data]
    shot_chooser = LS_shots()
    for text in etqdm(dev_text):
        shots = shot_chooser.get_shots(text, 5)
        assert len(shots) == 5

    text = "这是一个测试句子。"
    shots = shot_chooser.get_shots(text, 5)
    assert len(shots) == 5
