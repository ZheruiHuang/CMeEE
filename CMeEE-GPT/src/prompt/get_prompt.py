from typing import List
from src.prompt.build_prompt import prompt_base, PromptBERTHint
from src.utils.settings import get_json_data


def get_prompt(mode="base", hint_list: List[List[dict]] = None):
    if mode == "base":
        return prompt_base
    elif mode == "bert-hint":
        """
        Args:
            hint: list of list, each list is a hint with possible predictions.
            if hint is None, use the predictions of BERT.
        """
        if hint_list is None:
            hint_list = []
            hint_list.append(get_json_data("bert_pred_dev"))
            hint_list.append(get_json_data("bert_large_pred_dev"))
            # hint_list.append(get_json_data("bert_5_base_pred_dev"))
            # hint_list.append(get_json_data("ls_5_base_pred_dev"))
        return PromptBERTHint(hint_list)
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")
