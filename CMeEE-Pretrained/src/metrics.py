import numpy as np

from typing import List, Union, NamedTuple, Tuple, Counter
from datasets.ee_data import EE_label2id, EE_label2id1, EE_label2id2, EE_id2label1, EE_id2label2, EE_id2label, NER_PAD, _LABEL_RANK, NO_ENT


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


def get_stat_entity(entity_types: dict) -> str:
    max_times = max(entity_types.values())
    possible_entities = [_entity_type for _entity_type in entity_types.keys() if entity_types[_entity_type] == max_times]
    entity_type = max(possible_entities, key=lambda x: _LABEL_RANK[x])
    return entity_type


def process_pred_or_label(pred_or_label: np.ndarray, id2label: list) -> set:
    label2id = {label: id for id, label in enumerate(id2label)}
    entities = set()
    in_entity, start_idx, entity_types = False, None, {}
    for idx, id in enumerate(pred_or_label):
        if id == label2id[NER_PAD]:
            if in_entity:
                entities.add((start_idx, idx-1, get_stat_entity(entity_types)))
                in_entity, start_idx, entity_types = False, None, {}
            break
        BIO, entity_type = ('O', None) if id == label2id[NO_ENT] else id2label[id].split('-')
        if not in_entity:
            if BIO == 'B':
                in_entity, start_idx, entity_types = True, idx, {entity_type: 1}
        else:  # in_entity
            if BIO == 'I':
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            elif BIO == 'B':
                entities.add((start_idx, idx-1, get_stat_entity(entity_types)))
                in_entity, start_idx, entity_types = True, idx, {entity_type: 1}
            elif BIO == 'O':
                entities.add((start_idx, idx-1, get_stat_entity(entity_types)))
                in_entity, start_idx, entity_types = False, None, {}
    if in_entity:
        entities.add((start_idx, idx, get_stat_entity(entity_types)))
    return entities


class ComputeMetricsForNER: # training_args  `--label_names labels `
    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels[labels == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        #'''NOTE: You need to finish the code of computing f1-score.
        assert predictions.shape == labels.shape
        
        pred_entities_list = []
        for prediction in predictions:
            pred_entities_list.append(process_pred_or_label(prediction, EE_id2label))
        
        label_entities_list = []
        for label in labels:
            label_entities_list.append(process_pred_or_label(label, EE_id2label))
        
        pred_num, label_num, pred_true_num = 0, 0, 0
        for pred_entities, label_entities in zip(pred_entities_list, label_entities_list):
            pred_num += len(pred_entities)
            label_num += len(label_entities)
            for pred_entity in pred_entities:
                if pred_entity in label_entities:
                    pred_true_num += 1
        
        f1_score = 2 * pred_true_num / (pred_num + label_num)
        #'''

        return { "f1": f1_score }


class ComputeMetricsForNestedNER: # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        predictions, (labels1, labels2) = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len, 2]
        labels1[labels1 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels2[labels2 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        # '''NOTE: You need to finish the code of computing f1-score.
        assert predictions.shape[0] == labels1.shape[0]
        assert predictions.shape[0] == labels2.shape[0]
        
        pred_entities_list = []
        for prediction in predictions:
            pred_entities_list.append(
                process_pred_or_label(prediction[:, 0], EE_id2label1) \
                | process_pred_or_label(prediction[:, 1], EE_id2label2)
            )
        
        label_entities_list = []
        for label1, label2 in zip(labels1, labels2):
            label_entities_list.append(
                process_pred_or_label(label1, EE_id2label1) \
                | process_pred_or_label(label2, EE_id2label2)
            )
        
        pred_num, label_num, pred_true_num = 0, 0, 0
        for pred_entities, label_entities in zip(pred_entities_list, label_entities_list):
            pred_num += len(pred_entities)
            label_num += len(label_entities)
            for pred_entity in pred_entities:
                if pred_entity in label_entities:
                    pred_true_num += 1
                    
        f1_score = 2 * pred_true_num / (pred_num + label_num)
        # '''

        return { "f1": f1_score }


def extract_entities(batch_labels_or_preds: np.ndarray, for_nested_ner: bool = False, first_labels: bool = True) -> List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。
    
    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.  
        for_nested_ner:        Whether the input label ids is about Nested NER. 
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id1[NER_PAD]  # [batch, seq_len]

    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2

    batch_entities = []  # List[List[(start_idx, end_idx, type)]]
    
    # '''NOTE: You need to finish this function of extracting entities for generating results and computing metrics.
    for label_or_pred in batch_labels_or_preds:
        batch_entities.append(list(process_pred_or_label(label_or_pred, id2label)))
    # '''
    return batch_entities


if __name__ == '__main__':

    # Test for ComputeMetricsForNER
    predictions = np.load('../test_files/predictions.npy')
    labels = np.load('../test_files/labels.npy')

    metrics = ComputeMetricsForNER()(EvalPrediction(predictions, labels))

    if abs(metrics['f1'] - 0.606179116) < 1e-8:
        print('You passed the test for ComputeMetricsForNER.')
    else:
        print('The result of ComputeMetricsForNER is not right.')
    
    # Test for ComputeMetricsForNestedNER
    predictions = np.load('../test_files/predictions_nested.npy')
    labels1 = np.load('../test_files/labels1_nested.npy')
    labels2 = np.load('../test_files/labels2_nested.npy')

    metrics = ComputeMetricsForNestedNER()(EvalPrediction(predictions, (labels1, labels2)))

    if abs(metrics['f1'] - 0.60333644) < 1e-8:
        print('You passed the test for ComputeMetricsForNestedNER.')
    else:
        print('The result of ComputeMetricsForNestedNER is not right.')
