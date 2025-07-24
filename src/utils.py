import sys
sys.path.append('./')

import os
import math
import json
import torch
import numpy as np
import pandas as pd
from torch.nn import init
from scipy.special import expit
from codecarbon import EmissionsTracker
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

from transformers import AutoModelForSequenceClassification, EvalPrediction
# from models.xlm_roberta_classifier import XLMRobertaForSequenceClassification
# from models.modernbert_classifier import ModernBertForSequenceClassification
# from models.llama_classifier import LlamaForSequenceClassification

def get_optimizer_and_scheduler(config, model, train_loader):
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        config.training_settings['learning_rate'],
    )


    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.training_settings['learning_rate'],
        epochs=config.training_settings['epochs'],
        steps_per_epoch=len(train_loader),
        anneal_strategy=config.training_settings['anneal_strategy'],
    )


    return optimizer, scheduler


def collate_fn(examples):
    pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    labels = torch.tensor([example["labels"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def evaluate_model(model_output):
    '''Implementation for Task 4 in a soft-soft evaluation.
    '''
    
    report = {
            "precision": round(100 * precision_score(model_output['labels'], model_output['preds']), 2),
            "recall": round(100 * recall_score(model_output['labels'], model_output['preds']), 2),
            "F1": round(100 * f1_score(model_output['labels'], model_output['preds']), 2),
            "accuracy":  round(100 * accuracy_score(model_output['labels'], model_output['preds']), 2),
            "avg_loss": model_output['loss']/len(model_output['labels']),
        }
    
    return report

def get_carburacy(score, emission_train, emission_test, alpha=10, beta_train=1, beta_test=100):
    carburacy_train = None
    if emission_train is not None:
        carburacy_train = math.exp(math.log(score/100, alpha)) / (1 + emission_train * beta_train)
        carburacy_train = round(100 * carburacy_train, 2)
    carburacy_test = None
    if emission_test is not None:
        carburacy_test = math.exp(math.log(score/100, alpha)) / (1 + emission_test * beta_test)
        carburacy_test = round(100 * carburacy_test, 2)
    carburacy = None
    if carburacy_train is not None and carburacy_test is not None:
        carburacy = (2 * carburacy_train * carburacy_test) / (carburacy_train + carburacy_test)
        carburacy = round(100 * carburacy, 2)
    return carburacy_train, carburacy_test, carburacy


def predict_class(trainer, predict_dataset, max_predict_samples, training_args, split, target_column=None):
    test_tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
    test_tracker.start()
    predict_results = trainer.predict(predict_dataset, metric_key_prefix=split)
    test_emissions = test_tracker.stop()

    probs = predict_results.predictions
    if probs.ndim > 1:
        preds = (probs >= 0.5).astype(int)
    else:
        preds = (probs > predict_results.metrics[f"{split}_threshold"]).astype(int)

    all_pred_dicts = []
    predict_dataset.reset_format()
    for i in range(len(probs)):
        inst_id = predict_dataset[training_args.id_column][i].split(".")[0]
        if target_column == "soft_label_task4":
            value = probs[i].item() 
        else:

            if probs.ndim > 1:
                value = preds[i].tolist()
            else:
                value = int(preds[i])

        pred_dict = {"id": inst_id, "value": value}
            
        all_pred_dicts.append(pred_dict)

    output_prediction_file = os.path.join(training_args.output_dir, f"generated_{split}_set.json")
    with open(output_prediction_file, 'w', encoding='utf-8') as f:
        json.dump(all_pred_dicts, f, ensure_ascii=False, indent=4)
    
    if split != "test_challenge":

        metrics = predict_results.metrics

        metrics[f"{split}_samples"] = min(max_predict_samples, len(predict_dataset))
        metrics[f"{split}_emissions"] = test_emissions

        trainer.save_metrics(split, metrics)

        return metrics


def get_model(model_name, model_kwargs):
    if any(substring in model_name for substring in model_constructors):
        for substring, model_constructor in model_constructors.items():
            if substring in model_name:
                model = model_constructor.from_pretrained(model_name, **model_kwargs)
                break
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    
    return model


def evaluate_thresholds(probs, labels, num=50):
    """
    Evaluate metrics at various thresholds and return a DataFrame of results
    and the best result by accuracy.
    """
    thresholds = np.linspace(0, 1, num=num)
    results = []
    for x in thresholds:
        preds = (probs > x).astype(int)
        results.append({
            'threshold': round(x, 4),
            'precision_macro': round(100 * precision_score(labels, preds, average='macro', zero_division=0), 2),
            'recall_macro': round(100 * recall_score(labels, preds, average='macro', zero_division=0), 2),
            'F1_macro': round(100 * f1_score(labels, preds, average='macro', zero_division=0), 2),
            'accuracy': round(100 * accuracy_score(labels, preds), 2),
            'roc_auc': round(100 * roc_auc_score(labels, probs), 2),
        })
    df = pd.DataFrame(results)
    best = df.loc[df['accuracy'].idxmax()].to_frame().T.iloc[0].to_dict()
    
    return best


def preprocess_logits_for_metrics(logits, labels):
    """
    Convert raw model logits into a 1-D tensor of positive-class probabilities,
    detached and moved to CPU so we don’t hold on to any GPU graph.
    """

    # If Trainer returned a tuple (loss, logits), grab logits:
    logits = logits[0] if isinstance(logits, tuple) else logits
    if labels.dim() > 1:
        probs = expit(logits.detach().cpu())
    else:
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()

    return probs


def compute_metrics(eval_pred: EvalPrediction):
    # eval_pred.predictions is now a 1-D numpy array of positive-class probs
    probs = eval_pred.predictions
    labels = eval_pred.label_ids
    
    if labels.ndim > 1:
        preds = (probs >= 0.5).astype(int)

        results = {
            'precision_macro': round(100 * precision_score(labels, preds,
                                                        average='macro',
                                                        zero_division=0), 2),
            'recall_macro':    round(100 * recall_score(labels, preds,
                                                        average='macro',
                                                        zero_division=0), 2),
            'f1_macro':        round(100 * f1_score(labels, preds,
                                                    average='macro',
                                                    zero_division=0), 2),
            'accuracy':        round(100 * accuracy_score(labels, preds), 2),   
            'roc_auc_macro':   round(100 * roc_auc_score(labels, preds,
                                                        average='macro'), 2),
        }
    
    else:
        # run your threshold sweep
        results = evaluate_thresholds(probs, labels, num=100)

    return results


def init_gcn_layer(layer):
    """Apply Xavier init to the four core weight tensors of a single GCN layer."""
    for submodule in (layer.phi, layer.psi_param, layer.W_g, layer.W_r):
        init.xavier_uniform_(submodule.weight)


def set_config_from_args(config, model_args, data_args, training_args, config_json=None):
    """
    Populate a configuration object from model, training, and data argument namespaces.

    Args:
        config: An object with attributes to be set.
        model_args: Namespace containing model-specific arguments.
        training_args: Namespace containing training-specific arguments.
        data_args: Namespace containing data-specific arguments.

    Returns:
        The updated config object.
    """
    if config_json is None:
        # Training-time initialization
        # GCN layer settings
        config.num_gcn_layers = model_args.num_gcn_layers
        config.num_text_gcn_layers = model_args.num_text_gcn_layers
        config.num_image_gcn_layers = model_args.num_image_gcn_layers
        config.custom_gcn = model_args.custom_gcn
        config.save_affinity = model_args.save_affinity

        # Feature fusion and output settings
        config.apply_ffw = model_args.apply_ffw
        config.modality_fuser = model_args.modality_fuser

        # Training output and batch size
        config.output_dir = training_args.output_dir
        config.batch_size = training_args.per_device_eval_batch_size

        config.image_caption = data_args.image_caption
        config.soft_labels = True if "soft" in data_args.target_column else False
        config.multi_label = data_args.multi_label

    else:
        # Inference-time initialization from JSON
        config.num_gcn_layers = config_json["num_gcn_layers"]
        config.num_text_gcn_layers = config_json["num_text_gcn_layers"]
        config.num_image_gcn_layers = config_json["num_image_gcn_layers"]
        config.custom_gcn = config_json["custom_gcn"]
        config.modality_fuser = config_json['modality_fuser']

        # Output and inference settings
        config.output_dir = config_json.get("output_dir")
        config.apply_ffw = config_json.get("apply_ffw")
        config.image_caption = config_json.get("image_caption")
        config.soft_labels = config_json.get("soft_labels")

        # Use values from model_args / training_args when present
        config.save_affinity = model_args.save_affinity
        config.batch_size = training_args.per_device_eval_batch_size
        config.multi_label = data_args.multi_label

    return config

def model_xavier_init(config, model, model_args):
    # try to init parameters in a different way

    if model_args.classifier_xavier_init:
        init.xavier_uniform_(model.classifier.weight)

    if model_args.checkpoint_path is None:
        if config.num_gcn_layers > 0:
            init.xavier_uniform_(model.rs_gcn_layers[0].phi.weight)
            init.xavier_uniform_(model.rs_gcn_layers[0].psi_param.weight)
            init.xavier_uniform_(model.rs_gcn_layers[0].W_g.weight)
            init.xavier_uniform_(model.rs_gcn_layers[0].W_r.weight)

        if config.num_text_gcn_layers > 0:
            init.xavier_uniform_(model.text_gcn_layers[0].phi.weight)
            init.xavier_uniform_(model.text_gcn_layers[0].psi_param.weight)
            init.xavier_uniform_(model.text_gcn_layers[0].W_g.weight)
            init.xavier_uniform_(model.text_gcn_layers[0].W_r.weight)
        if config.num_image_gcn_layers > 0:
            init.xavier_uniform_(model.image_gcn_layers[0].phi.weight)
            init.xavier_uniform_(model.image_gcn_layers[0].psi_param.weight)
            init.xavier_uniform_(model.image_gcn_layers[0].W_g.weight)
            init.xavier_uniform_(model.image_gcn_layers[0].W_r.weight)
        
        if config.modality_fuser == "mfb":
            init.xavier_uniform_(model.modality_fuser.lin_text.weight)
            init.xavier_uniform_(model.modality_fuser.lin_image.weight)
        elif config.modality_fuser == "gmu":
            init.xavier_uniform_(model.modality_fuser.lin_t.weight)
            init.xavier_uniform_(model.modality_fuser.lin_v.weight)
            init.xavier_uniform_(model.modality_fuser.lin_gate.weight)
        elif config.modality_fuser == "cross_attn":
            init.xavier_uniform_(model.modality_fuser.query_lin.weight)
            init.xavier_uniform_(model.modality_fuser.key_lin.weight)
            init.xavier_uniform_(model.modality_fuser.value_lin.weight)

    for param in model.parameters(): param.data = param.data.contiguous()


# model_constructors = {
#     "xlm-roberta": XLMRobertaForSequenceClassification,
#     "ModernBERT": None,
#     "Meta-Llama": LlamaForSequenceClassification,
# }

