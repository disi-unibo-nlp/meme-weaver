import sys
sys.path.append('./')

import os
import math
import json
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from scipy.special import softmax
from codecarbon import EmissionsTracker
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

from transformers import AutoModelForSequenceClassification, EvalPrediction
from models.xlm_roberta_classifier import XLMRobertaForSequenceClassification
# from models.modernbert_classifier import ModernBertForSequenceClassification
from models.llama_classifier import LlamaForSequenceClassification

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

    logits = predict_results.predictions[0]
    preds = np.argmax(logits, axis=1)

    all_pred_dicts = []
    for i in range(len(logits)):
        inst_id = predict_dataset["id"][i].split(".")[0]
        if target_column == "soft_label_task4":
            yes_prob = logits[i][1].item() 
            no_prob = 1 - yes_prob
            value = {"NO": no_prob, "YES": yes_prob}
        else:
            value = "YES" if preds[i] == 1 else "NO" 

        pred_dict = {"test_case": "EXIST2025", "id": inst_id, "value": value}
        # if split != "test_challenge":
        #    pred_dict["target_label"] = "YES" if predict_dataset[target_column][i] == 1 else "NO"
            
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


def compute_metrics(output: EvalPrediction):
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    labels = output.label_ids


    preds = np.argmax(logits, axis=1)
    probs = softmax(logits, axis=1)[:, 1]

    result = {
        "precision_macro": round(100 * precision_score(labels, preds, average='macro'), 2),
        "recall_macro": round(100 * recall_score(labels, preds, average='macro'), 2),
        "F1_macro": round(100 * f1_score(labels, preds, average='macro'), 2),
        "accuracy": round(100 * accuracy_score(labels, preds), 2),
        "roc_auc": round(100 * roc_auc_score(labels, probs), 2),
    }
    
    return result


def init_gcn_layer(layer):
    """Apply Xavier init to the four core weight tensors of a single GCN layer."""
    for submodule in (layer.phi, layer.psi_param, layer.W_g, layer.W_r):
        init.xavier_uniform_(submodule.weight)


model_constructors = {
    "xlm-roberta": XLMRobertaForSequenceClassification,
    "ModernBERT": None,
    "Meta-Llama": LlamaForSequenceClassification,
}

