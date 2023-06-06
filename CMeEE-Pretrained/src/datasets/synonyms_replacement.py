import os.path as osp
import random
import json
import jieba
import synonyms
from tqdm import tqdm


def generate_stop_words(in_f, out_f):
    with open(in_f, 'r') as f:
        json_data = json.load(f)
    stop_words = set()
    for entities_dict in json_data:
        text = entities_dict["text"]
        flag = [False for _ in range(len(text))]
        for entity in entities_dict["entities"]:
            start_idx, end_idx = entity["start_idx"], entity["end_idx"]
            flag[start_idx:end_idx+1] = [True for _ in range(end_idx-start_idx+1)]
        words = list(jieba.cut(text))
        char_num = 0
        for word in words:
            if sum(flag[char_num:char_num+len(word)]) > 0:
                stop_words.add(word)
            char_num += len(word)
    with open(out_f, 'w') as f:
        for stop_word in stop_words:
            f.write(stop_word + '\n')


def replace_words_with_synonyms(sentence: str, stop_words: set = set(), max_num_replaced: int = 2) -> str:
    """
    Example:
        >>> sentence = "我爱北京天安门"
        >>> new_sentence = replace_words_with_synonyms(sentence, 2)
        >>> print(new_sentence)  # 你爱天津天安门; 我爱南京宛平城 etc.
    """
    new_words = list(jieba.cut(sentence))
    replacement_words = [_word for _word in set(new_words) if _word not in stop_words]
    random.shuffle(replacement_words)
    num_replaced = 0
    for word in replacement_words:
        synonyms_words = [_word for _word in synonyms.nearby(word, 5)[0][1:] if len(_word) == len(word)]
        if len(synonyms_words) >= 1:
            synonym = random.choice(synonyms_words)
            new_words = [synonym if _word == word else _word for _word in new_words]
            num_replaced += 1
        if num_replaced >= max_num_replaced:
            break

    new_sentence = ''.join(new_words)
    return new_sentence


def generate_synonyms(in_f, out_f, stop_words: set = set(), replace_num: int = 2):
    with open(in_f, 'r') as f:
        json_data = json.load(f)
    new_json_data = []
    for entities_dict in tqdm(json_data):
        new_json_data.append(entities_dict)
        text, entities = entities_dict["text"], entities_dict["entities"]
        texts_set = {text}
        for _ in range(replace_num):
            new_text = replace_words_with_synonyms(text, stop_words)
            if new_text not in texts_set:
                new_json_data.append({"text": new_text, "entities": entities})
                texts_set.add(new_text)
            else:
                break
    with open(out_f, 'w') as f:
        json.dump(new_json_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    data_root = osp.join(osp.dirname(__file__), "../../data/CBLUEDatasets/CMeEE/")
    # generate stop words
    if not osp.exists(osp.join(data_root, "stop_words.txt")):
        generate_stop_words(osp.join(data_root, "CMeEE_train.json"), osp.join(data_root, "stop_words.txt"))
    with open(osp.join(data_root, "stop_words.txt"), 'r') as f:
        stop_words = set([line.strip() for line in f.readlines()])
    # generate synonyms
    if not osp.exists(osp.join(data_root, "CMeEE_train_synonyms.json")):
        print("Generating synonyms...")
        generate_synonyms(osp.join(data_root, "CMeEE_train.json"), osp.join(data_root, "CMeEE_train_synonyms.json"), stop_words, 2)