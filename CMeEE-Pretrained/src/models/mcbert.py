from transformers import BertConfig
from .model import BertForCRFHeadNER, BertForLinearHeadNER, BertForCRFHeadNestedNER, BertForLinearHeadNestedNER


class McbertForLinearHeadNER(BertForLinearHeadNER):
    config_class = BertConfig
    base_model_prefix = "mcbert"


class McbertForLinearHeadNestedNER(BertForLinearHeadNestedNER):
    config_class = BertConfig
    base_model_prefix = "mcbert"


class McbertForCRFHeadNER(BertForCRFHeadNER):
    config_class = BertConfig
    base_model_prefix = "mcbert"


class McbertForCRFHeadNestedNER(BertForCRFHeadNestedNER):
    config_class = BertConfig
    base_model_prefix = "mcbert"
