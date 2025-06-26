import sys
sys.path.append('./')

import os 
import json
import torch
import wandb
from functools import partial
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoFeatureExtractor,
)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

from src.arguments import ModelArguments, DataTrainingArguments
from models.clip_classifier import CLIPForMultimodalClassification
from src.utils import compute_metrics, predict_class, set_config_from_args, collate_fn, preprocess_logits_for_metrics
from models.multimodal_classifier import CustomMultiModalForClassification, MultiModalConfig


def main():

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_texts(examples):
        
        captions = [caption for caption in examples[data_args.text_column]]
        if config.image_caption is not None: 
            captions = [inp + "[CPT]" + cpt  for inp, cpt in zip(captions, examples[config.image_caption])]
        
        text_inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        examples["labels"] = [label for label in examples[data_args.target_column]]
        return examples
    
    def apply_transforms_val(examples):
        pixel_values = [
            val_transforms(img.convert("RGB"))
            for img in examples[data_args.image_column]
        ]
        return {
            "input_ids":      examples["input_ids"],
            "attention_mask": examples["attention_mask"],
            "labels":         examples[data_args.target_column],
            "pixel_values":   pixel_values,
            "instance_ids": examples[data_args.id_column],
        }
    
    wandb.init(mode="disabled")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_subset,
            # download_mode="force_redownload",
        )
    

    split_dataset = raw_datasets[data_args.split]
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load feature_extractor, in this script we only use this to get the mean and std for normalization.
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2, # TODO make it dynamic
        cache_dir="../llms",
        trust_remote_code=True,
    )
   
    image_size = config.vision_config.image_size
    crop_size = (image_size, image_size)


    val_transforms = Compose(
            [
                Resize(image_size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )

    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)

    # Find the folder starting with "checkpoint-"
    checkpoint_folder = next(folder for folder in os.listdir(training_args.output_dir) if folder.startswith("checkpoint-"))
    # Append the checkpoint folder to the base path
    checkpoint_path = os.path.join(training_args.output_dir, checkpoint_folder)

    path_config_json = os.path.join(checkpoint_path, "config.json")
    with open(path_config_json, "r") as f:
        config_json = json.load(f)
        
    config = set_config_from_args(config, model_args, data_args, training_args, config_json)
    not_remove_columns = [data_args.image_column, data_args.target_column, data_args.id_column]
    column_names = raw_datasets[data_args.split].column_names
    split_dataset = split_dataset.map(
            function=tokenize_texts,
            batched=True,
            remove_columns=[col for col in column_names if col not in not_remove_columns],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )
    
    split_dataset.set_transform(apply_transforms_val)

    if model_args.text_model_name_or_path is not None and model_args.vision_model_name_or_path is not None:

        model = CustomMultiModalForClassification.from_pretrained(checkpoint_path, config=config,)
    
    else:

        model = CLIPForMultimodalClassification.from_pretrained(
            checkpoint_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            config=config,
        )
        
    for param in model.parameters(): param.data = param.data.contiguous()

    batch_sizes = (
        [training_args.per_device_eval_batch_size]
        if training_args.per_device_eval_batch_size != -1
        else range(1, 120)
    )
    metrics_function = None if data_args.save_inference or config.soft_labels else compute_metrics 
    training_args.remove_unused_columns = False
    
    for batch_size in batch_sizes:

        print(f"Running inference with batch size: {batch_size}")
        training_args.per_device_eval_batch_size = batch_size
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=metrics_function,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            tokenizer=tokenizer,
            data_collator=collate_fn,
        )

        if data_args.save_inference:
            predict_class(trainer, split_dataset, len(split_dataset), training_args, data_args.split, target_column=data_args.target_column)
        else:
            predict_results = trainer.predict(split_dataset, metric_key_prefix="predict")
            metrics = predict_results.metrics

            metric_output_path = os.path.join(training_args.output_dir, "inferences")
            os.makedirs(metric_output_path, exist_ok=True)
            # Save the metrics to json 
            metrics_file = os.path.join(metric_output_path, f"{data_args.split}_results_batch{training_args.per_device_eval_batch_size}.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)

        
if __name__ == "__main__":
    main()