import os
import pickle
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    add_start_docstrings,
    ModelOutput,
)

import torch.nn.init as init
from transformers.activations import ACT2FN                           

from transformers.modeling_outputs import BaseModelOutputWithPooling

from transformers.models.clip.configuration_clip import CLIPConfig

from transformers.models.clip.modeling_clip import (
    CLIP_START_DOCSTRING,
    CLIP_TEXT_INPUTS_DOCSTRING,
    CLIP_VISION_INPUTS_DOCSTRING,
    CLIP_INPUTS_DOCSTRING,
    CLIPTextModel,
    CLIPVisionModel,
    CLIPTextConfig,
    CLIPVisionConfig,
    CLIPPreTrainedModel,
    CLIPMLP
)

from models.custom_modules import gcn_map


@dataclass
class CLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    fused_features: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
    

# Copied from transformers.models.roberta.modeling_roberta.RobertaClassificationHead with Roberta->XLMRoberta
class ClipClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        custom_gcn = gcn_map[config.custom_gcn]
        self.rs_gcn_layers = nn.ModuleList(
            [custom_gcn(config) for _ in range(config.num_gcn_layers)]
        )

        self.apply_ffw = config.apply_ffw

        if self.apply_ffw:
            self.mlp = CLIPMLP(config)

    def forward(self, x, **kwargs):

        R_norm = None
        for gcn in self.rs_gcn_layers:
            # x = self.dropout(x)
            x, _ = gcn(x)

            if self.apply_ffw:
                x_fw = self.mlp(x)
                # Residual connection
                x = x_fw + x

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def feed_forward_chunk(self, x):
        intermediate_output = self.intermediate(x)
        layer_output = self.output(intermediate_output, x)
        return layer_output




@add_start_docstrings(CLIP_START_DOCSTRING)
class CLIPForMultimodalClassification(CLIPPreTrainedModel):
    config_class = CLIPConfig
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer", "CLIPVisionEmbeddings"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        self.num_labels = config.num_labels

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        text_model = CLIPTextModel._from_config(text_config, attn_implementation=config._attn_implementation)
        self.text_model = text_model.text_model

        vision_model = CLIPVisionModel._from_config(vision_config, attn_implementation=config._attn_implementation)
        self.vision_model = vision_model.vision_model

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.classifier = nn.Linear(self.projection_dim * 2, config.num_labels)
        # self.classifier = ClipClassificationHead(config)

        self.loss_fct = nn.CrossEntropyLoss()

        self.num_text_gcn_layers = config.num_text_gcn_layers
        self.num_image_gcn_layers = config.num_image_gcn_layers

        custom_gcn = gcn_map[config.custom_gcn]
        self.rs_gcn_layers = nn.ModuleList(
            [custom_gcn(self.projection_dim * 2) for _ in range(config.num_gcn_layers)]
        )


        self.text_gcn_layers = nn.ModuleList(
            [custom_gcn(self.projection_dim) for _ in range(config.num_text_gcn_layers)]
        )

        self.image_gcn_layers = nn.ModuleList(
            [custom_gcn(self.projection_dim) for _ in range(config.num_image_gcn_layers)]
        )

        self.apply_ffw = config.apply_ffw
        if self.apply_ffw:
            config.hidden_act = "quick_gelu"
            config.hidden_size = self.projection_dim * 2
            config.intermediate_size = self.projection_dim * 4
            config.num_hidden_layers = 12
            self.mlp = CLIPMLP(config)

        self.output_dir = config.output_dir
        self.batch_size = config.batch_size
        self.save_affinity = config.save_affinity

        # Initialize weights and apply final processing
        self.post_init()


    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def apply_gcn(self, fused_features, gcn_layers):
        """
        Apply the GCN layers (and optional feed-forward) to the fused features.
        Returns:
            fused_features (torch.Tensor): The updated features after GCN and optional FFN.
            R_norm (optional): Any second output from the last GCN layer (if applicable).
        """
        R_norm = None
        for gcn in gcn_layers:
            fused_features, R_norm = gcn(fused_features)
            if self.apply_ffw:
                x_fw = self.mlp(fused_features)
                # Residual connection
                fused_features = x_fw + fused_features

        return fused_features, R_norm

    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=CLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        image_embeds, R_norm_image = self.apply_gcn(image_embeds, self.image_gcn_layers)
        text_embeds, R_norm_text = self.apply_gcn(text_embeds, self.text_gcn_layers)

        # Multimodal fusion (concatenation in this case)
        fused_features = torch.cat((text_embeds, image_embeds), dim=-1)

        # Apply GCN layers (and optional feed-forward) to the fused features.
        fused_features_upd, R_norm = self.apply_gcn(fused_features, self.rs_gcn_layers)

        # Compute logits for classification.
        logits = self.classifier(fused_features_upd)

        if self.save_affinity:
            output_path = os.path.join(self.output_dir, f"affinity_matrices_{self.batch_size}_bs")
            os.makedirs(output_path, exist_ok=True)
            
            proc_files = len(os.listdir(output_path))

            # Save data
            with open(os.path.join(output_path, f'affinity_batch_{proc_files + 1}.pkl'), 'wb') as f:
                pickle.dump({'R_norm': R_norm, 
                             'features': fused_features.cpu(),
                             'features_upd': fused_features_upd.cpu(),
                             'labels': labels.cpu(), 
                             "image_embeds": image_embeds.cpu(), 
                             "text_embeds": text_embeds.cpu(),
                             "logits": logits.cpu()}, f)

        loss = None
        import pdb; pdb.set_trace()
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits=logits,
            fused_features=fused_features,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
