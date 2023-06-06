from transformers import BertConfig
from .model import BertForCRFHeadNER, BertForLinearHeadNER, BertForCRFHeadNestedNER, BertForLinearHeadNestedNER


class EHRbertForLinearHeadNER(BertForLinearHeadNER):
    config_class = BertConfig
    base_model_prefix = "ehr_bert"


class EHRbertForLinearHeadNestedNER(BertForLinearHeadNestedNER):
    config_class = BertConfig
    base_model_prefix = "ehr_bert"


class EHRbertForCRFHeadNER(BertForCRFHeadNER):
    config_class = BertConfig
    base_model_prefix = "ehr_bert"


class EHRbertForCRFHeadNestedNER(BertForCRFHeadNestedNER):
    config_class = BertConfig
    base_model_prefix = "ehr_bert"
