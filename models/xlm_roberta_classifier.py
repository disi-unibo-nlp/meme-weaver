# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch XLM-RoBERTa model."""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLM_ROBERTA_START_DOCSTRING,
    XLMRobertaPreTrainedModel,
    XLMRobertaModel,
    XLM_ROBERTA_INPUTS_DOCSTRING,

)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "XLMRobertaConfig"


class Rs_GCN(nn.Module):
    def __init__(self, hidden_size):
        """
        Initialize the Rs_GCN module.

        Args:
            hidden_size (int): Dimensionality of the input features.
        """
        super(Rs_GCN, self).__init__()
        # Fully connected layer to compute φ(·) for transforming input embeddings.
        self.phi = nn.Linear(hidden_size, hidden_size)
        # Fully connected layer to compute γ(·) for transforming input embeddings.
        self.gamma = nn.Linear(hidden_size, hidden_size)
        # Linear transformation applied to the node features after aggregation.
        self.W_g = nn.Linear(hidden_size, hidden_size)
        # Residual weights linear layer applied on the aggregated features.
        self.W_r = nn.Linear(hidden_size, hidden_size)

    def forward(self, features):
        """
        Forward pass for the Rs_GCN layer.

        Args:
            embeddings (torch.Tensor): Input tensor of shape (batch_size, hidden_size)
                                       Each row represents a node's feature.
        Returns:
            torch.Tensor: Updated node features of shape (batch_size, hidden_size)
        """
        # Transform input features using φ and γ functions.
        phi_out = self.phi(features)  # Shape: (batch_size, hidden_size)
        gamma_out = self.gamma(features)  # Shape: (batch_size, hidden_size)

        # Compute the affinity matrix R as the dot product between transformed features.
        R = torch.matmul(phi_out, gamma_out.t())  # Shape: (batch_size, batch_size)

        # Normalize the affinity matrix by dividing by the number of nodes (i.e., last dimension size).
        R_norm = R / R.size(-1)

        # Apply a linear transformation on the original features.
        features_v = self.W_g(features)  # Shape: (batch_size, hidden_size)

        # Aggregate neighboring features using the normalized affinity matrix.
        RV = torch.matmul(R_norm, features_v)  # Shape: (batch_size, hidden_size)

        # Apply a second linear transformation on the aggregated features.
        transformed = self.W_r(RV)  # Shape: (batch_size, hidden_size)

        # Add a residual connection from the original features.
        out = transformed + features

        return out



# Copied from transformers.models.roberta.modeling_roberta.RobertaClassificationHead with Roberta->XLMRoberta
class XLMRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)


        self.rs_gcn_layers = nn.ModuleList(
            [Rs_GCN(config.hidden_size) for _ in range(config.num_gcn_layers)]
        )

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])

        for gcn in self.rs_gcn_layers:
            x = gcn(x)

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForSequenceClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = XLMRobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
