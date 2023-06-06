CBLUE_ROOT = "./data/CBLUEDatasets/CMeEE"

TYPE_MAP = {
    "dep": "科室",
    "equ": "医疗设备",
    "mic": "微生物",
    "ite": "医学检验项目",
    "dru": "药物",
    "pro": "医疗程序",
    "sym": "临床表现",
    "dis": "疾病",
    "bod": "身体",
}

NUM_ZH_MAP = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
}

type_inverse_map = {value: key for key, value in TYPE_MAP.items()}

KEY_LIST = [
    # put OpenAI API tokens here
]


NER_PAD, NO_ENT = "[PAD]", "O"

LABEL1 = ["dep", "equ", "mic", "ite", "dru", "pro", "dis", "bod"]  # 按照出现频率从低到高排序
LABEL2 = ["sym"]
LABEL = ["dep", "equ", "mic", "ite", "dru", "pro", "sym", "dis", "bod"]

EE_id2label1 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL1 for P in ("B", "I")]
EE_id2label2 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL2 for P in ("B", "I")]
EE_id2label = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL for P in ("B", "I")]

EE_label2id1 = {b: a for a, b in enumerate(EE_id2label1)}
EE_label2id2 = {b: a for a, b in enumerate(EE_id2label2)}
EE_label2id = {b: a for a, b in enumerate(EE_id2label)}

EE_NUM_LABELS1 = len(EE_id2label1)
EE_NUM_LABELS2 = len(EE_id2label2)
EE_NUM_LABELS = len(EE_id2label)

CHINESE_FILTER = ['，','。','（','）','【','】','有','其','之','些','只','还','就','这','那','能','你','我','他','她','是','的','得','地','了','呢','吧','啊','哦','嗯','噢','哈','呀','嘛','嘻','嘿',]

NAME_LIST = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
