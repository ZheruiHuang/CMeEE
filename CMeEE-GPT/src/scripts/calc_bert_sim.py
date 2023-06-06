import os
import json
import numpy as np

from src.utils.const import CBLUE_ROOT
from src.utils.etqdm import etqdm
from src.utils.utils import cos_sim

if __name__ == "__main__":
    train_data = json.load(open(os.path.join(CBLUE_ROOT, "feat", "train_bert_feat.json"), encoding="utf-8"))
    train_feat = [feat for text, feat in train_data.items()]
    train_feat = np.array(train_feat)  # (N, 768)
    train_text = [text for text, feat in train_data.items()]
    print(f"===> Train Size: {len(train_data)}")

    dev_data = json.load(open(os.path.join(CBLUE_ROOT, "feat", "select_dev_bert_feat.json"), encoding="utf-8"))
    print(f"===> Dev Size: {len(dev_data)}")

    sim_res = {}
    for text, feat in etqdm(dev_data.items()):
        feat = np.array(feat)  # (768,)
        sim = cos_sim(feat, train_feat).reshape(-1)
        idx = np.argsort(sim)[::-1][:8]
        sim_res[text] = [train_text[i] for i in idx]
    json.dump(
        sim_res,
        open(os.path.join("data_share", "sim_res.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )
