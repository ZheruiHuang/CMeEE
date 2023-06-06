from transformers import BertConfig
from .model import BertForCRFHeadNER, BertForLinearHeadNER, BertForCRFHeadNestedNER, BertForLinearHeadNestedNER


class RobertaForLinearHeadNER(BertForLinearHeadNER):
    config_class = BertConfig
    base_model_prefix = "roberta"


class RobertaForLinearHeadNestedNER(BertForLinearHeadNestedNER):
    config_class = BertConfig
    base_model_prefix = "roberta"


class RobertaForCRFHeadNER(BertForCRFHeadNER):
    config_class = BertConfig
    base_model_prefix = "roberta"


class RobertaForCRFHeadNestedNER(BertForCRFHeadNestedNER):
    config_class = BertConfig
    base_model_prefix = "roberta"
