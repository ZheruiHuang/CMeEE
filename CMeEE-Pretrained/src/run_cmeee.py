import os
import json
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
from typing import List
import warnings

from transformers import (
    set_seed, BertTokenizer, Trainer, HfArgumentParser, TrainingArguments, RobertaTokenizer, AutoTokenizer, PreTrainedModel
)

from args import ModelConstructArgs, CBLUEDataArgs
from utils.logger import get_logger
from datasets.ee_data import EE_label2id2, EEDataset, EE_NUM_LABELS1, EE_NUM_LABELS2, EE_NUM_LABELS, CollateFnForEE, \
    EE_label2id1, NER_PAD, EE_label2id
from models.model import (
    BertForCRFHeadNER, BertForLinearHeadNER, BertForLinearHeadNestedNER, BertForCRFHeadNestedNER
)
from models.roberta import (
    RobertaForLinearHeadNER, RobertaForLinearHeadNestedNER, RobertaForCRFHeadNER, RobertaForCRFHeadNestedNER
)
from models.mcbert import (
    McbertForLinearHeadNER, McbertForLinearHeadNestedNER, McbertForCRFHeadNER, McbertForCRFHeadNestedNER
)
from models.ehr_bert import (
    EHRbertForLinearHeadNER, EHRbertForLinearHeadNestedNER, EHRbertForCRFHeadNER, EHRbertForCRFHeadNestedNER
)
from models.rocbert import (
    RoCBertForLinearHeadNER, RoCBertForLinearHeadNestedNER, RoCBertForCRFHeadNER, RoCBertForCRFHeadNestedNER
)
from models.assemble_bert import (
    AssemblebertForLinearHeadNER, AssemblebertForLinearHeadNestedNER, AssemblebertForCRFHeadNER, AssemblebertForCRFHeadNestedNER   
)
from models.lstm import (
    LSTMForLinearHeadNER, LSTMForLinearHeadNestedNER, LSTMForCRFHeadNER, LSTMForCRFHeadNestedNER
)
from models.bert_lstm import (
    BertLSTMForLinearHeadNER, BertLSTMForLinearHeadNestedNER, BertLSTMForCRFHeadNER, BertLSTMForCRFHeadNestedNER   
)
from metrics import ComputeMetricsForNER, ComputeMetricsForNestedNER, extract_entities
from torch.nn import LSTM
from torch.optim import AdamW

warnings.filterwarnings("ignore")

MODEL_CLASS = {
    'bert_linear': BertForLinearHeadNER, 
    'bert_linear_nested': BertForLinearHeadNestedNER,
    'bert_crf': BertForCRFHeadNER,
    'bert_crf_nested': BertForCRFHeadNestedNER,
    'roberta_linear': RobertaForLinearHeadNER,
    'roberta_linear_nested': RobertaForLinearHeadNestedNER,
    'roberta_crf': RobertaForCRFHeadNER,
    'roberta_crf_nested': RobertaForCRFHeadNestedNER,
    'mcbert_linear': McbertForLinearHeadNER,
    'mcbert_linear_nested': McbertForLinearHeadNestedNER,
    'mcbert_crf': McbertForCRFHeadNER,
    'mcbert_crf_nested': McbertForCRFHeadNestedNER,
    'ehr_bert_linear': EHRbertForLinearHeadNER,
    'ehr_bert_linear_nested': EHRbertForLinearHeadNestedNER,
    'ehr_bert_crf': EHRbertForCRFHeadNER,
    'ehr_bert_crf_nested': EHRbertForCRFHeadNestedNER,
    'rocbert_linear': RoCBertForLinearHeadNER,
    'rocbert_linear_nested': RoCBertForLinearHeadNestedNER,
    'rocbert_crf': RoCBertForCRFHeadNER,
    'rocbert_crf_nested': RoCBertForCRFHeadNestedNER,
    'assemble_bert_linear': AssemblebertForLinearHeadNER,
    'assemble_bert_linear_nested': AssemblebertForLinearHeadNestedNER,
    'assemble_bert_crf': AssemblebertForCRFHeadNER,
    'assemble_bert_crf_nested': AssemblebertForCRFHeadNestedNER,
    'lstm_linear': LSTMForLinearHeadNER,
    'lstm_linear_nested': LSTMForLinearHeadNestedNER,
    'lstm_crf': LSTMForCRFHeadNER,
    'lstm_crf_nested': LSTMForCRFHeadNestedNER,
    'bert_lstm_linear': BertLSTMForLinearHeadNER,
    'bert_lstm_linear_nested': BertLSTMForLinearHeadNestedNER,
    'bert_lstm_crf': BertLSTMForCRFHeadNER,
    'bert_lstm_crf_nested': BertLSTMForCRFHeadNestedNER,
}

def get_logger_and_args(logger_name: str, _args: List[str] = None):
    parser = HfArgumentParser([TrainingArguments, ModelConstructArgs, CBLUEDataArgs])
    
    train_args, model_args, data_args = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(logger_name, exp_dir=train_args.logging_dir, rank=train_args.local_rank)
    for _log_name, _logger in logger.manager.loggerDict.items():
        # 在4.6.0版本的transformers中无效
        if _log_name.startswith("transformers.trainer"):
            # Redirect other loggers' output
            _logger.addHandler(logger.handlers[0])

    logger.info(f"==== Train Arguments ==== {train_args.to_json_string()}")
    logger.info(f"==== Model Arguments ==== {model_args.to_json_string()}")
    logger.info(f"==== Data Arguments ==== {data_args.to_json_string()}")

    return logger, train_args, model_args, data_args


def get_model_with_tokenizer(model_args):
    model_type = model_args.model_type
    head_type = model_args.head_type
    model_class: PreTrainedModel = MODEL_CLASS[f"{model_type}_{head_type}"]
    
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(model_args.model_path)
    elif model_type in ['roberta', 'mcbert', 'ehr_bert', 'rocbert', 'assemble_bert', 'lstm', 'bert_lstm']:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, use_fast=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    ntokens = len(tokenizer)
    
    if 'nested' not in model_args.head_type:
        if model_type in ['assemble_bert', 'lstm']:
            model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS, ntokens=ntokens)
        else:
            model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS)
    else:
        if model_type in ['assemble_bert', 'lstm']:
            model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1, num_labels2=EE_NUM_LABELS2, ntokens=ntokens)
        else:
            model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1, num_labels2=EE_NUM_LABELS2)

    return model, tokenizer


def generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=False, name="test"):
    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    if not for_nested_ner:
        pred_entities1 = extract_entities(predictions[:, 1:], for_nested_ner=False)
        pred_entities2 = [[]] * len(pred_entities1)
    else:
        pred_entities1 = extract_entities(predictions[:, 1:, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities(predictions[:, 1:, 1], for_nested_ner=True, first_labels=False)

    final_answer = []

    for p1, p2, example in zip(pred_entities1, pred_entities2, test_dataset.examples):
        text = example.text
        entities = []
        for start_idx, end_idx, entity_type in p1 + p2:
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": entity_type,
                "entity": text[start_idx: end_idx + 1],
            })
        final_answer.append({"text": text, "entities": entities})

    # if the output path exists, add index to the file name
    path = join(train_args.output_dir, f"CMeEE_{name}.json")
    index = 1
    while os.path.exists(path):
        path = join(train_args.output_dir, f"CMeEE_{name}_{index}.json")
        index += 1

    with open(path, "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`{path}` saved")


def main(_args: List[str] = None):
    # ===== Parse arguments =====
    logger, train_args, model_args, data_args = get_logger_and_args(__name__, _args)

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    model, tokenizer = get_model_with_tokenizer(model_args)
    for_nested_ner = 'nested' in model_args.head_type

    # ===== Get optimizer =====
    if train_args.do_train and model_args.optimizer:
        # implement layer-wise learning rate decay for BERT
        grouped_parameters = []
        lr = train_args.learning_rate
        lr_decay = 0.98
        layer_names = [name for name, _ in model.named_parameters()]
        layer_names.reverse()
        for name in layer_names:
            grouped_parameters.append({'params': [p for n, p in model.named_parameters() if name in n], 'lr': lr})
            lr *= lr_decay
        optimizer = AdamW(grouped_parameters, lr=train_args.learning_rate)
    else:
        optimizer = None
    
    # ===== Get datasets =====
    if train_args.do_train:
        train_dataset = EEDataset(data_args.cblue_root, "train", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner, synonyms_replacement=data_args.synonyms_replacement)
        dev_dataset = EEDataset(data_args.cblue_root, "dev", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner, synonyms_replacement=False)
        logger.info(f"Trainset: {len(train_dataset)} samples")
        logger.info(f"Devset: {len(dev_dataset)} samples")
    else:
        train_dataset = dev_dataset = None

    # ===== Trainer =====
    compute_metrics = ComputeMetricsForNestedNER() if for_nested_ner else ComputeMetricsForNER()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    if train_args.do_train:
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

    if train_args.do_predict:
        test_dataset = EEDataset(data_args.cblue_root, "test", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner, synonyms_replacement=False)
        logger.info(f"Testset: {len(test_dataset)} samples")
        if dev_dataset is None:
            dev_dataset = EEDataset(data_args.cblue_root, "dev", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner, synonyms_replacement=False)
            logger.info(f"Devset: {len(dev_dataset)} samples")

        # np.ndarray, None, None
        predictions, _labels, _metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=for_nested_ner)

        predictions, _labels, _metrics = trainer.predict(dev_dataset, metric_key_prefix="predict")
        generate_testing_results(train_args, logger, predictions, dev_dataset, for_nested_ner=for_nested_ner, name="dev")


if __name__ == '__main__':
    main()
