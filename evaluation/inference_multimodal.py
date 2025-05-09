import sys
sys.path.append('./')

import os 
import json
import torch
import wandb
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
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


from src.utils import get_model, compute_metrics
from src.arguments import ModelArguments, DataTrainingArguments
from models.clip_classifier import CLIPForMultimodalClassification


def main():

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_texts(examples):
        
        captions = [caption for caption in examples[data_args.text_column]]
        text_inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True)
        examples["input_ids"] = text_inputs.input_ids
        examples["attention_mask"] = text_inputs.attention_mask
        return examples
    
    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image_path"]]
        return example_batch
    
    def collate_fn(examples):
        pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
        input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
        attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
        labels = torch.tensor([example[data_args.target_column] for example in examples])
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
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

    labels = list(set(split_dataset[data_args.target_column]))
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(labels),
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

    column_names = raw_datasets[data_args.split].column_names
    split_dataset = split_dataset.map(
            function=tokenize_texts,
            batched=True,
            remove_columns=[col for col in column_names if col not in [data_args.image_column, data_args.target_column]],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )

    split_dataset = split_dataset.map(  
        preprocess_val,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False)

    output_path = os.path.join(training_args.output_dir, training_args.run_name)

    # Find the folder starting with "checkpoint-"
    checkpoint_folder = next(folder for folder in os.listdir(output_path) if folder.startswith("checkpoint-"))
    # Append the checkpoint folder to the base path
    checkpoint_path = os.path.join(output_path, checkpoint_folder)

    path_config_json = os.path.join(checkpoint_path, "config.json")
    with open(path_config_json, "r") as f:
        config_json = json.load(f)
        
    config.num_gcn_layers = config_json["num_gcn_layers"]
    config.num_text_gcn_layers = config_json["num_text_gcn_layers"]
    config.num_image_gcn_layers = config_json["num_image_gcn_layers"]
    config.custom_gcn = config_json["custom_gcn"]
    config.output_dir = config_json["output_dir"]
    config.apply_ffw = config_json["apply_ffw"]
    config.save_affinity = model_args.save_affinity
    config.batch_size = training_args.per_device_eval_batch_size


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
        else range(1, 200)
    )

    for batch_size in batch_sizes:

        training_args.per_device_eval_batch_size = batch_size
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=collate_fn,
        )


        predict_results = trainer.predict(split_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics

        metric_output_path = os.path.join(output_path, "inferences")
        os.makedirs(metric_output_path, exist_ok=True)
        # Save the metrics to json 
        metrics_file = os.path.join(metric_output_path, f"{data_args.split}_results_batch{training_args.per_device_eval_batch_size}.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        

if __name__ == "__main__":
    main()