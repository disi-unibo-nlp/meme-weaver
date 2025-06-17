import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig, PreTrainedModel, PretrainedConfig
from models.modular_modernbert import ModernBertModel, ModernBertConfig

from models.custom_modules import gcn_map
from models.modality_fusers import fuser_map


class MultiModalConfig(PretrainedConfig):
    # If you already have a custom config class, extend it. Otherwise you can just pass
    # model names directly to the model __init__ and skip a custom config.
    # Here we show how to include text and vision model names in config.
    model_type = "multimodal-classifier"

    def __init__(
        self,
        text_model_name_or_path: str,
        vision_model_name_or_path: str,
        projection_dim: int = 768,
        num_labels: int = 2,
        logit_scale_init_value: float = 2.6592,
        # you can add more hyperparameters here
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_model_name_or_path = text_model_name_or_path
        self.vision_model_name_or_path = vision_model_name_or_path
        self.projection_dim = projection_dim
        self.num_labels = num_labels
        self.logit_scale_init_value = logit_scale_init_value
        self.vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "text_model_name_or_path": self.text_model_name_or_path,
            "vision_model_name_or_path": self.vision_model_name_or_path,
            "projection_dim": self.projection_dim,
            "num_labels": self.num_labels,
            "logit_scale_init_value": self.logit_scale_init_value,
        })
        return d
    
    def to_diff_dict(self):
        # Don’t attempt to instantiate a new default instance
        # Just return the full dict (i.e. no diff-optimization).
        return self.to_dict()


class CustomMultiModalForClassification(PreTrainedModel):
    config_class = MultiModalConfig

    def __init__(self, config: MultiModalConfig):
        super().__init__(config)
        # 1. Load text encoder via AutoModel

        if "ModernBERT" in config.text_model_name_or_path:
            text_conf = ModernBertConfig.from_pretrained(config.text_model_name_or_path)
            self.text_model = ModernBertModel.from_pretrained(config.text_model_name_or_path)
        else:
            text_conf = AutoConfig.from_pretrained(config.text_model_name_or_path)
            self.text_model = AutoModel.from_pretrained(config.text_model_name_or_path, config=text_conf)
        # 2. Load vision encoder via AutoModel
        vision_conf = AutoConfig.from_pretrained(config.vision_model_name_or_path)
        self.vision_model = AutoModel.from_pretrained(config.vision_model_name_or_path, config=vision_conf)
        # 3. Projection layers
        text_hidden = text_conf.hidden_size
        vision_hidden = vision_conf.hidden_size
        self.text_proj = nn.Linear(text_hidden, config.projection_dim, bias=False)
        self.vision_proj = nn.Linear(vision_hidden, config.projection_dim, bias=False)
        # 4. Optional logit scale if you later want similarity-based fusion
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))
        # 5. Classifier on concatenated projections
        self.classifier = nn.Linear(config.projection_dim * 2, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

        custom_gcn = gcn_map[config.custom_gcn]
        self.rs_gcn_layers = nn.ModuleList(
            [custom_gcn(config.projection_dim * 2) for _ in range(config.num_gcn_layers)]
        )


        self.text_gcn_layers = nn.ModuleList(
            [custom_gcn(config.projection_dim) for _ in range(config.num_text_gcn_layers)]
        )

        self.image_gcn_layers = nn.ModuleList(
            [custom_gcn(config.projection_dim) for _ in range(config.num_image_gcn_layers)]
        )

        self.output_dir = config.output_dir
        self.batch_size = config.batch_size
        self.save_affinity = config.save_affinity

        self.modality_fuser = fuser_map[config.modality_fuser](config)


        self.post_init()  # initialize weights if needed

    def get_text_features(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if hasattr(self.text_model.config, "type_vocab_size") else None,
            return_dict=True,
            **kwargs,
        )
        # Most AutoModel text models put pooled output at out.pooler_output if present.
        # If the model has no pooler (e.g., some RoBERTa variants), you may take e.g. the first token:
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output  # shape (batch, hidden)
        else:
            # fallback: use CLS token hidden state
            pooled = out.last_hidden_state[:, 0]
        feats = self.text_proj(pooled)  # (batch, proj_dim)
        return feats

    def get_image_features(self, pixel_values=None, images=None, **kwargs):
        # Depending on how you preprocess images: typically you pass pixel_values tensor.
        # If your vision_model expects 'pixel_values', use that.
        # If it expects 'pixel_values' or other inputs, adapt accordingly.
        # Here assume pixel_values is a tensor from a feature extractor.
        out = self.vision_model(pixel_values=pixel_values, return_dict=True, **kwargs)
        # Many vision AutoModels (e.g., ViTModel) have pooler_output; else take first token:
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        feats = self.vision_proj(pooled)
        return feats
    
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

        return fused_features, R_norm
    
    def gradient_checkpointing_enable(self, **kwargs):
        """
        Override the default no-op so that Trainer.train() can
        actually enable checkpointing in the text & vision encoders.
        """
        # turn on checkpointing in each HF sub-model
        if hasattr(self.text_model, "gradient_checkpointing_enable"):
            self.text_model.gradient_checkpointing_enable(**kwargs)
        if hasattr(self.vision_model, "gradient_checkpointing_enable"):
            self.vision_model.gradient_checkpointing_enable(**kwargs)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pixel_values=None,
        labels=None,
        **kwargs
    ):
        # 1. Text features
        text_feats = self.get_text_features(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        
        # 2. Image features
        image_feats = self.get_image_features(pixel_values=pixel_values, **kwargs)
        
        # 3. Normalize (optional but often helpful)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        image_feats = image_feats / image_feats.norm(p=2, dim=-1, keepdim=True)
        
        # 4. apply any per-modality GCN here
        text_feats, _ = self.apply_gcn(text_feats, self.text_gcn_layers)
        image_feats, _ = self.apply_gcn(image_feats, self.image_gcn_layers)
        
        # 5. Fuse by concatenation
        # if self.training and (torch.rand(1).item() < 0.5):
        #     image_feats = image_feats * 0.0

        fused = self.modality_fuser(text_feats, image_feats)
        
        # 6. fused GCN / FFN
        fused_upd, _ = self.apply_gcn(fused, self.rs_gcn_layers)
        
        # 7. Classification
        logits = self.classifier(fused_upd)  # (batch, num_labels)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return {"loss": loss, "logits": logits, "text_feats": text_feats, "image_feats": image_feats}
