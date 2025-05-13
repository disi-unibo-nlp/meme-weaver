#!/usr/bin/env python
# coding=utf-8

# adapted from https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text/run_clip.py


# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.


import os
import torch
import wandb
import numpy
import random

from datasets import load_dataset
import torch.nn.init as init
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

from codecarbon import EmissionsTracker

from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    get_scheduler,
    AutoConfig,
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


from utils import compute_metrics, predict_class, init_gcn_layer
from arguments import ModelArguments, DataTrainingArguments
from models.clip_classifier import CLIPForMultimodalClassification

# We use torchvision for faster image pre-processing. The transforms are implemented as nn.Module,
# so we jit it to be faster.
class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    numpy.random.seed(seed=training_args.seed)
    random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_subset,
        cache_dir="../datasets",
    )

    training_args.output_dir += "/" + training_args.run_name

    wandb.init(mode=data_args.logging,
            name=training_args.run_name,
            project=data_args.dataset_name.split("/")[1] + f"_{data_args.dataset_subset}",
    )


    column_names = dataset["train"].column_names
    labels = list(set(dataset["train"][data_args.target_column]))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=len(labels),
        cache_dir="../llms",
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )
    
    config.num_gcn_layers = model_args.num_gcn_layers
    config.num_text_gcn_layers = model_args.num_text_gcn_layers
    config.num_image_gcn_layers = model_args.num_image_gcn_layers
    config.custom_gcn = model_args.custom_gcn
    config.save_affinity = model_args.save_affinity
    config.apply_ffw = model_args.apply_ffw
    config.output_dir = training_args.output_dir
    config.batch_size = training_args.per_device_eval_batch_size

    if model_args.checkpoint_path is not None:
        checkpoint_folder = next(folder for folder in os.listdir(model_args.checkpoint_path) if folder.startswith("checkpoint-"))
        
        # Append the checkpoint folder to the base path
        checkpoint_path = os.path.join(model_args.checkpoint_path, checkpoint_folder)
    else:
        checkpoint_path = model_args.model_name_or_path
    
    model = CLIPForMultimodalClassification.from_pretrained(
        checkpoint_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        config=config,
    )

    # try to init parameters in a different way
    if config.num_gcn_layers > 0 and model_args.checkpoint_path is None:
        init.xavier_uniform_(model.rs_gcn_layers[0].phi.weight)
        init.xavier_uniform_(model.rs_gcn_layers[0].psi_param.weight)
        init.xavier_uniform_(model.rs_gcn_layers[0].W_g.weight)
        init.xavier_uniform_(model.rs_gcn_layers[0].W_r.weight)

    if config.num_text_gcn_layers > 0 and model_args.checkpoint_path is None:
        init.xavier_uniform_(model.text_gcn_layers[0].phi.weight)
        init.xavier_uniform_(model.text_gcn_layers[0].psi_param.weight)
        init.xavier_uniform_(model.text_gcn_layers[0].W_g.weight)
        init.xavier_uniform_(model.text_gcn_layers[0].W_r.weight)
    if config.num_image_gcn_layers > 0 and model_args.checkpoint_path is None:
        init.xavier_uniform_(model.image_gcn_layers[0].phi.weight)
        init.xavier_uniform_(model.image_gcn_layers[0].psi_param.weight)
        init.xavier_uniform_(model.image_gcn_layers[0].W_g.weight)
        init.xavier_uniform_(model.image_gcn_layers[0].W_r.weight)

    for param in model.parameters(): param.data = param.data.contiguous()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
    )

    # Load feature_extractor, in this script we only use this to get the mean and std for normalization.
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
   
    image_size = config.vision_config.image_size
    crop_size = (image_size, image_size)
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
                Resize(image_size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_texts(examples):
        
        captions = [caption for caption in examples[data_args.text_column]]

        if data_args.image_caption is not None: 
            captions = [inp + "[CPT]" + cpt  for inp, cpt in zip(captions, examples[data_args.image_caption])]

        text_inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        examples["label"] = [caption for caption in examples[data_args.target_column]]
        return examples
    
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

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(max_train_samples))
    
        train_dataset = train_dataset.map(
            function=tokenize_texts,
            batched=True,
            remove_columns=[col for col in column_names if col not in [data_args.image_column, data_args.target_column]],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )

        train_dataset = train_dataset.map(  
            preprocess_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False)

        

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a train validation")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.shuffle(seed=training_args.seed).select(range(max_eval_samples))

        eval_dataset = eval_dataset.map(
            function=tokenize_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col not in [data_args.image_column, data_args.target_column]],
            load_from_cache_file=False,
            desc="Running tokenizer on validation dataset",
        )

        eval_dataset = eval_dataset.map(  
            preprocess_val,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False)

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = dataset["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(test_dataset), data_args.max_eval_samples)
            test_dataset = test_dataset.shuffle(seed=training_args.seed).select(range(max_eval_samples))

        test_dataset = test_dataset.map(
            function=tokenize_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col not in [data_args.image_column, data_args.target_column]],
            load_from_cache_file=False,
            desc="Running tokenizer on test dataset",
        )

        test_dataset = test_dataset.map(  
            preprocess_val,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False)
    
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

    def collate_fn(examples):
        pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
        input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
        attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
        labels = torch.tensor([example["label"] for example in examples])
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    n_steps = len(train_dataset)/training_args.per_device_train_batch_size * training_args.num_train_epochs
    training_args.eval_steps = n_steps // 8
    training_args.save_steps = n_steps // 8
    training_args.logging_steps = n_steps // 100

    # 8. Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        optimizers=optimizers,
        compute_metrics=compute_metrics,
    )

    # 9. Training
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

    # 10. Evaluation
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
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(test_dataset)
        )
        predict_metrics = predict_class(trainer, test_dataset, max_predict_samples, training_args, "predict")
        wandb.log(predict_metrics)



if __name__ == "__main__":
    main()