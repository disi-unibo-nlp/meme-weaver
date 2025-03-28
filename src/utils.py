import os
import math
import torch
from codecarbon import EmissionsTracker
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



def predict_class(trainer, predict_dataset, max_predict_samples, training_args, tokenizer, train_emissions, split):
    test_tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
    test_tracker.start()
    predict_results = trainer.predict(predict_dataset, metric_key_prefix=split)
    test_emissions = test_tracker.stop()

    metrics = predict_results.metrics

    metrics[f"{split}_samples"] = min(max_predict_samples, len(predict_dataset))
    metrics[f"{split}_emissions"] = test_emissions

    # trainer.log_metrics(split, metrics)
    trainer.save_metrics(split, metrics)

    predictions = predict_results.predictions
    output_prediction_file = os.path.join(training_args.output_dir, f"generated_{split}_set.txt")
    
    with open(output_prediction_file, 'w') as file:
        # Write each element of the list on a new line
        for item in predictions:
            file.write(f"{item}\n")
