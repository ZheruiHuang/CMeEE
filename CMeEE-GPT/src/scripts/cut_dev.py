import json
import random

CBLUE_ROOT = "./data/CBLUEDatasets/CMeEE"
# all_data = json.load(open(CBLUE_ROOT + "/CMeEE_dev.json", encoding="utf-8"))
all_data = json.load(open(CBLUE_ROOT + "/select_dev.json", encoding="utf-8"))

length = 50
id_list = list(range(500))
store_data = []
for i in range(length):
    j = random.choice(list(range(len(id_list))))
    candidate = id_list[j]
    id_list.pop(j)
    store_data.append(all_data[candidate])

with open("dev_{}.json".format(length), "a", encoding="utf-8") as result_file:
    result_file.write(json.dumps(store_data, indent=4, sort_keys=False, ensure_ascii=False))

# all_data = json.load(open("small_dev_data.json", encoding="utf-8"))
# print(len(all_data))
