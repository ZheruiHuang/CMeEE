from transformers import BertConfig, BertModel, BertPreTrainedModel
from .model import _group_ner_outputs, LinearClassifier, CRFClassifier
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, ntokens, ninp, nhid, nlayers, dropout) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntokens, ninp)
        self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.nlayers = nlayers
        self.nhid = nhid
        self.init_weights()

    def forward(self, input_ids):
        emb = self.drop(self.encoder(input_ids))
        hidden = (torch.zeros((self.nlayers, input_ids.size(1), self.nhid)).to(input_ids.device),
                  torch.zeros((self.nlayers, input_ids.size(1), self.nhid)).to(input_ids.device),)
        lstm_output, hidden = self.lstm(emb, hidden)
        lstm_output = self.drop(lstm_output)
        return lstm_output

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

class AssemblebertForLinearHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "assemble_bert"
    
    def __init__(self, config: BertConfig, num_labels1: int, ntokens: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        # set hyperparameters for lstm
        self.lstm = LSTM(ntokens, config.hidden_size, config.hidden_size, 2, 0.2)
        self.classifier = LinearClassifier(config.hidden_size * 2, num_labels1, config.hidden_dropout_prob)
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
        sequence_output = self.bert(
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

        lstm_output = self.lstm(input_ids)

        output = torch.concat([sequence_output, lstm_output], dim=-1)
        output = self.classifier.forward(output, labels, no_decode=no_decode)
        return output
        
class AssemblebertForCRFHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "assemble_bert"
    
    def __init__(self, config: BertConfig, num_labels1: int, ntokens: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        # set hyperparameters for lstm
        self.lstm = LSTM(ntokens, config.hidden_size, config.hidden_size, 2, 0.2)
        self.classifier = CRFClassifier(config.hidden_size * 2, num_labels1, config.hidden_dropout_prob)
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
        sequence_output = self.bert(
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

        lstm_output = self.lstm(input_ids)

        output = torch.concat([sequence_output, lstm_output], dim=-1)
        output = self.classifier.forward(output, attention_mask, labels, no_decode=no_decode)
        return output

class AssemblebertForLinearHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "assemble_bert"
    
    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int, ntokens: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        # set hyperparameters for lstm
        self.lstm = LSTM(ntokens, config.hidden_size, config.hidden_size, 2, 0.2)
        self.classifier1 = LinearClassifier(config.hidden_size * 2, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = LinearClassifier(config.hidden_size * 2, num_labels2, config.hidden_dropout_prob)
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
        sequence_output = self.bert(
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

        lstm_output = self.lstm(input_ids)

        output = torch.concat([sequence_output, lstm_output], dim=-1)
        
        output1 = self.classifier1.forward(output, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(output, labels2, no_decode=no_decode)
        output = _group_ner_outputs(output1, output2)
        return output

class AssemblebertForCRFHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "assemble_bert"
    
    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int, ntokens: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        # set hyperparameters for lstm
        self.lstm = LSTM(ntokens, config.hidden_size, config.hidden_size, 2, 0.2)
        self.classifier1 = CRFClassifier(config.hidden_size * 2, num_labels1, config.hidden_dropout_prob)
        self.classifier2 = CRFClassifier(config.hidden_size * 2, num_labels2, config.hidden_dropout_prob)
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
        sequence_output = self.bert(
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

        lstm_output = self.lstm(input_ids)

        output = torch.concat([sequence_output, lstm_output], dim=-1)
        
        output1 = self.classifier1.forward(output, attention_mask, labels, no_decode=no_decode)
        output2 = self.classifier2.forward(output, attention_mask, labels2, no_decode=no_decode)
        output = _group_ner_outputs(output1, output2)
        return output
