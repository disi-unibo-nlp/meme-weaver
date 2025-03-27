import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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
