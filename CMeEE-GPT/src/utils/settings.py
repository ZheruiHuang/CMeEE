import os
import json
from .const import CBLUE_ROOT


def set_proxy(port=33210):
    PROXY = "127.0.0.1:{}".format(port)
    os.environ["HTTP_PROXY"] = os.environ["http_proxy"] = PROXY
    os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = PROXY
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost,.local"


def get_json_data(data_type="all"):
    data = {}
    if data_type == "all":
        data = json.load(open(CBLUE_ROOT + "/CMeEE_dev.json", encoding="utf-8"))
    elif data_type == "train":
        data = json.load(open(CBLUE_ROOT + "/CMeEE_train.json", encoding="utf-8"))
    elif data_type == "select_dev":
        data = json.load(open(CBLUE_ROOT + "/select_dev.json", encoding="utf-8"))
    elif data_type == "bert_sim":
        data = json.load(open(os.path.join("data_share", "sim_res.json"), encoding="utf-8"))
    elif data_type == "ls_sim":
        data = json.load(open(os.path.join("data_share", "ls_sim_res.json"), encoding="utf-8"))
    elif data_type == "select_50":
        data = json.load(open(os.path.join("data_share", "select_dev_50.json"), encoding="utf-8"))
    elif data_type == "bert_pred_dev":
        data = json.load(open(os.path.join("data_share", "BERT_CMeEE_dev_pred.json"), encoding="utf-8"))
    elif data_type == "bert_5_base_pred_dev":
        data = json.load(open(os.path.join("data_share", "bert_5_base_select_dev.json"), encoding="utf-8"))
    elif data_type == "ls_5_base_pred_dev":
        data = json.load(open(os.path.join("data_share", "ls_5_base_select_dev.json"), encoding="utf-8"))
    elif data_type == "bert_large_pred_dev":
        data = json.load(open(os.path.join("data_share", "BERT_large_CMeEE_dev_pred.json"), encoding="utf-8"))
    elif data_type == "train_bert_pred":
        data = json.load(open(os.path.join(CBLUE_ROOT, "BERT_large_CMeEE_train.json"), encoding="utf-8"))
    else:
        raise NotImplementedError

    return data
