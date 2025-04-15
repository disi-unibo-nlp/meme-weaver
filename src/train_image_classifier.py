import sys
sys.path.append('./')

import wandb
import torch
import argparse
from datasets import load_dataset
from codecarbon import EmissionsTracker

from transformers import (
    AutoModelForImageClassification, 
    AutoImageProcessor,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    get_scheduler
)


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from utils import compute_metrics, predict_class
from arguments import ModelArguments, DataTrainingArguments


def main():

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image_path"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image_path"]]
        return example_batch
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    training_args.output_dir += "/" + training_args.run_name

    wandb.init(mode=data_args.logging,
            name=training_args.run_name,
            project=data_args.dataset_name.split("/")[1] + f"_{data_args.dataset_subset}",
    )

    
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_subset,
        cache_dir="../datasets",
    )

    labels = list(set(dataset["train"][data_args.target_column]))
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    
    image_processor  = AutoImageProcessor.from_pretrained(model_args.model_name_or_path, use_fast=True)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    train_transforms = Compose(
            [
                RandomResizedCrop(crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    predict_dataset = dataset["test"]
    
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(max_train_samples))
    
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.shuffle(seed=training_args.seed).select(range(max_eval_samples))

    if data_args.max_predict_samples is not None:
        max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
        predict_dataset = predict_dataset.shuffle(seed=training_args.seed).select(range(max_predict_samples))

    train_dataset = train_dataset.map(
            preprocess_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False)
    
    
    eval_dataset = eval_dataset.map(
            preprocess_val,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False)
    predict_dataset = predict_dataset.map(  
            preprocess_val,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False)
    
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir="../llms",
        ignore_mismatched_sizes=True, # tells the loader to ignore size mismatches
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")


    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    num_update_steps_per_epoch = len(train_dataset)
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps
    )
    optimizers = (optimizer, lr_scheduler)


    n_steps = len(train_dataset)/training_args.per_device_train_batch_size * training_args.num_train_epochs
    training_args.eval_steps = n_steps // 8
    training_args.save_steps = n_steps // 8

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
    )

    # Training
    if training_args.do_train:

        train_tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
        train_tracker.start()
        train_result = trainer.train()
        train_emissions = train_tracker.stop()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics["train_emissions"] = train_emissions

        trainer.save_model()  # Saves the tokenizer too for easy upload
 
        # trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if training_args.do_eval:
        print("*** Evaluate ***")
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        eval_metrics = predict_class(trainer, eval_dataset, max_eval_samples, training_args, "eval")
        wandb.log(eval_metrics)
    
    if training_args.do_predict:
        print("*** Predict ***")
        max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        predict_metrics = predict_class(trainer, predict_dataset, max_predict_samples, training_args, "predict")
        wandb.log(predict_metrics)



if __name__ == "__main__":

    main()