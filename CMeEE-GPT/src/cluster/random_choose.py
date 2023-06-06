from src.utils.settings import get_json_data
import random


class Random_shots:
    def __init__(self):
        self.data = get_json_data("train")

    def get_shots(self, text, nums):
        data_list = []
        for i in range(nums):
            idx = random.randint(100, len(self.data) - 1)
            data = self.data[idx]
            data_list.append(data)
        return data_list
