import torch
import numpy as np
from .const import type_inverse_map


def cos_sim(a: np.ndarray or torch.Tensor, b: np.ndarray or torch.Tensor) -> np.ndarray:
    """
    Calculate the cosine similarity between two vectors.
    :param a: vector a, shape: (n, d) or (d, ) (n=1)
    :param b: vector b, shape: (m, d) or (d, ) (m=1)
    :return: cosine similarity, shape: (n, m)
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.norm(a, dim=1, keepdim=True)
    b_norm = torch.norm(b, dim=1, keepdim=True)
    a = a / a_norm
    b = b / b_norm
    return torch.mm(a, b.t()).numpy()


def distance(a: np.ndarray or torch.Tensor, b: np.ndarray or torch.Tensor) -> np.ndarray:
    """
    Calculate the distance between two vectors.
    :param a: vector a, shape: (n, d) or (d, ) (n=1)
    :param b: vector b, shape: (m, d) or (d, ) (m=1)
    :return: distance, shape: (n, m)
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.cdist(a, b).numpy()


def is_digit(n: str):
    return n.isdigit()


def str2int(a):
    int_list = list(filter(is_digit, a))
    result = ""
    for i in int_list:
        result += i
    return int(result)


def search_for_ent(text, entity):
    index_list = []
    for idx, item in enumerate(text):
        if item == entity[0] and idx + len(entity) <= len(text):
            if text[idx : idx + len(entity)] == entity:
                index_list.append(idx)
    return index_list


def back_to_data(response, text):
    res = []
    for idx, item in enumerate(response.split("\n")):
        if item.startswith("|") and idx > 0:
            result = item.split("|")[1:-1]
            if result[1] not in type_inverse_map:
                continue
            # start = str2int(result[2])
            # end = str2int(result[3])
            index_list = search_for_ent(text, result[0])
            for start in index_list:
                end = start + len(result[0]) - 1
                result_dict = {
                    "start_idx": start,
                    "end_idx": end,
                    "type": type_inverse_map[result[1]],
                    "entity": result[0],
                }

                res.append(result_dict)
    return res
