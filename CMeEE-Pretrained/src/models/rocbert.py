from transformers import RoCBertConfig, RoCBertModel, RoCBertTokenizer, RoCBertPreTrainedModel
from .model import LinearClassifier, CRFClassifier, _group_ner_outputs


class RoCBertForLinearHeadNER(RoCBertPreTrainedModel):
    config_class = RoCBertConfig
    base_model_prefix = "rocbert"

    def __init__(self, config: RoCBertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.roc_bert = RoCBertModel(config)

        self.classifier = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels2=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        no_decode=False,
    ):
        sequence_output = self.roc_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, labels, no_decode=no_decode)
        return output


class RoCBertForLinearHeadNestedNER(RoCBertPreTrainedModel):
    config_class = RoCBertConfig
    base_model_prefix = "rocbert"

    def __init__(self, config: RoCBertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.roc_bert = RoCBertModel(config)
        self.classifier1 = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = LinearClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels2=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        no_decode=False,
    ):
        sequence_output = self.roc_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output1 = self.classifier1.forward(sequence_output, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(sequence_output, labels2, no_decode=no_decode)
        output = _group_ner_outputs(output1, output2)
        return output


class RoCBertForCRFHeadNER(RoCBertPreTrainedModel):
    config_class = RoCBertConfig
    base_model_prefix = "rocbert"

    def __init__(self, config: RoCBertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.roc_bert = RoCBertModel(config)
        self.classifier = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels2=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        no_decode=False,
    ):
        sequence_output = self.roc_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, attention_mask, labels, no_decode=no_decode)

        return output


class RoCBertForCRFHeadNestedNER(RoCBertPreTrainedModel):
    config_class = RoCBertConfig
    base_model_prefix = "rocbert"

    def __init__(self, config: RoCBertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.roc_bert = RoCBertModel(config)
        self.classifier1 = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = CRFClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels2=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        no_decode=False,
    ):
        sequence_output = self.roc_bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output1 = self.classifier1.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(sequence_output, attention_mask, labels2, no_decode=no_decode)
        output = _group_ner_outputs(output1, output2)
        return output
