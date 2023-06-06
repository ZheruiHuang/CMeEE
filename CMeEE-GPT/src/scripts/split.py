import json
import jieba

CBLUE_ROOT = "./data/CBLUEDatasets/CMeEE"
all_data = json.load(open(CBLUE_ROOT + "/CMeEE_dev.json", encoding="utf-8"))

# jieba.enable_paddle()
length = 1000
store_data = all_data[:length]


# 清空一下
with open("jieba_{}.txt".format(length), "w", encoding="utf-8") as result_file:
    result_file.close()

# jieba
for data in store_data:
    text = data["text"]
    seg_list = list(jieba.cut(text, cut_all=False))
    sentences = ""
    for idx, seg in enumerate(seg_list):
        if idx == len(seg_list) - 1:
            sentences += seg + "\n"
        else:
            sentences += seg + "/"

    with open("jieba_{}.txt".format(length), "a", encoding="utf-8") as result_file:
        result_file.write(sentences)

# word_by_word
for data in store_data:
    text = data["text"]
    sentences = ""
    for idx, seg in enumerate(text):
        if idx == len(text) - 1:
            sentences += seg + "\n"
        else:
            sentences += seg + "/"

    with open("word_{}.txt".format(length), "a", encoding="utf-8") as result_file:
        result_file.write(sentences)
        
# with open("dev_{}.json".format(length), "a", encoding="utf-8") as result_file:
#     result_file.write(json.dumps(store_data, indent=4, sort_keys=False, ensure_ascii=False))