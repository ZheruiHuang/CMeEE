from typing import List, Union
from src.utils.const import TYPE_MAP, NUM_ZH_MAP, NAME_LIST


def prompt_base(data_list, target_text, **kwargs):
    prompt = "以下是些样例，不需要进行预测。\n"
    for i, data in enumerate(data_list):
        prompt += "第{}个样例：\n".format(i + 1)
        text = data["text"]
        entities = data["entities"]

        msg = "|实体|类别|\n"
        for entity in entities:
            entity_type = entity["type"]
            entity_text = entity["entity"]

            msg += "|" + entity_text + "|" + TYPE_MAP[entity_type] + "|\n"

        prompt += text + "\n" + "实体：\n" + msg
        prompt += "\n"
    prompt += "以下是输入文本，请进行预测：\n" + target_text + "\n实体："
    return prompt


class PromptBERTHint:
    def __init__(
        self,
        hint_list: List[List],
        hint_name: Union[str, List[str]] = NAME_LIST,
    ):
        """
        Args:
            hint: list of list, each list is a hint with possible predictions.
        """
        self.hint_list = []
        for hint in hint_list:
            self.hint_list.append({})
            for data in hint:
                self.hint_list[-1][data["text"]] = data
        self.hint_num = 0
        for hint in self.hint_list:
            self.hint_num += len(hint)
        print(f"{self.hint_num} hints loaded.")

        self.hint_name = hint_name
        if isinstance(self.hint_name, str):
            self.hint_name = [self.hint_name]

    def __call__(self, data_list, target_text, **kwargs):
        hint_list = self._get_hint(data_list)
        prompt = self._prompt_bert_hint(data_list, hint_list)
        tar_hint_list = self._get_hint([{"text": target_text}])
        prompt += f"\n\n{self._gen_question(target_text, tar_hint_list)}"
        return prompt

    def _get_hint(self, data_list):
        ret_hint_list = []
        for data in data_list:
            text = data["text"]
            ret_hint_list.append([])
            for hint_name, hint in zip(self.hint_name, self.hint_list):
                if text in hint:
                    ret_hint_list[-1].append((hint_name, hint[text]))
        return ret_hint_list

    def _prompt_bert_hint(self, data_list, hint_list: List[List[tuple]]):
        prompt = "以下是一些样例，不需要进行预测。\n\n"
        for i, (data, hints) in enumerate(zip(data_list, hint_list)):
            prompt += f"第{NUM_ZH_MAP[i+1]}个样例：\n"
            text, entities = data["text"], data["entities"]

            hint_content = ""
            # NOTE: if data is from train set, hints will be [],
            # then hint_content will be empty
            for hint in hints:
                assert hint[1]["text"] == text
                hint_content += f"{hint[0]}给出的回答是：\n"
                hint_content += "|实体|类别|\n"
                for entity in hint[1]["entities"]:
                    entity_type = entity["type"]
                    entity_text = entity["entity"]
                    hint_content += f"|{entity_text}|{TYPE_MAP[entity_type]}|\n"
            hint_content = hint_content.rstrip("\n")

            # ground truth
            gt = f"正确答案是：\n"
            gt += "|实体|类别|\n"
            for entity in entities:
                entity_type = entity["type"]
                entity_text = entity["entity"]
                gt += f"|{entity_text}|{TYPE_MAP[entity_type]}|\n"
            gt = gt.rstrip("\n")

            for content in ("文本：", text, hint_content, gt):
                if content == "":
                    continue
                prompt += f"{content}\n"
            prompt += "\n"

        prompt = prompt.rstrip("\n")
        return prompt

    def _gen_question(self, target_text, hint_list: List[List[tuple]]):
        question = "以下是输入文本，一些人给出了回答，请你以此为提示：\n"
        for hints in hint_list:
            hint_content = ""
            for hint in hints:
                assert hint[1]["text"] == target_text
                hint_content += f"{hint[0]}给出的回答是：\n"
                hint_content += "|实体|类别|\n"
                for entity in hint[1]["entities"]:
                    entity_type = entity["type"]
                    entity_text = entity["entity"]
                    hint_content += f"|{entity_text}|{TYPE_MAP[entity_type]}|\n"
            hint_content = hint_content.rstrip("\n")

            for content in ("文本：", target_text, hint_content):
                if content == "":
                    continue
                question += f"{content}\n"
            question += "\n"

        question = question.rstrip("\n")
        question += f"\n\n现在请你进行预测：\n文本：\n{target_text}\n实体："
        return question


if __name__ == "__main__":
    # from src.cluster.get_cluster import get_cluster
    from src.utils.settings import get_json_data

    hint_list = [get_json_data("bert_pred_dev")]
    hint_list.append(get_json_data("bert_5_base_pred_dev"))
    hint_list.append(get_json_data("bert_large_pred_dev"))
    prompt_gen = PromptBERTHint(hint_list)
    shots = get_json_data("select_50")[:2]
    examples = prompt_gen(shots)
    print(examples)
