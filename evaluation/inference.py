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
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from src.utils import get_model, compute_metrics
from src.arguments import ModelArguments, DataTrainingArguments


def main():

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs1, targets = [], []
        inputs2 = None
        for i in range(len(examples[data_args.input_column])):
            inputs1.append(examples[data_args.input_column][i])
            targets.append(examples[data_args.target_column][i])
        
        if data_args.add_caption: 
            inputs1 = [inp + "[CPT]" + cpt  for inp, cpt in zip(inputs1, examples["qwen25vl_caption"])]

        # Tokenize the texts
        args = ((inputs1,) if inputs2 is None else (inputs1, inputs2))
        result = tokenizer(*args, padding=False, truncation=True)
        # Map labels to IDs
        if label_to_id is not None and data_args.target_column in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples[data_args.target_column]]
        return result
    
    wandb.init(mode="disabled")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_subset,
            # download_mode="force_redownload",
        )
    
    label_list = list(set(raw_datasets["train"][data_args.target_column]))
    num_labels = len(label_list)
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    split_dataset = raw_datasets[data_args.split]
    column_names = raw_datasets[data_args.split].column_names
    split_dataset = split_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on prediction dataset",
    )

    output_path = os.path.join(training_args.output_dir, training_args.run_name)

    # Find the folder starting with "checkpoint-"
    checkpoint_folder = next(folder for folder in os.listdir(output_path) if folder.startswith("checkpoint-"))
    # Append the checkpoint folder to the base path
    checkpoint_path = os.path.join(output_path, checkpoint_folder)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir="../llms",
        trust_remote_code=True,
    )

    path_config_json = os.path.join(checkpoint_path, "config.json")
    with open(path_config_json, "r") as f:
        config_json = json.load(f)
        
    config.num_gcn_layers = config_json["num_gcn_layers"]
    config.custom_gcn = config_json["custom_gcn"]
    config.save_affinity = config_json["save_affinity"]
    config.output_dir = config_json["output_dir"]

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    # Custom config hyperparameters
    model_kwargs = dict(
            torch_dtype="auto",
            # use_cache=False, # set to False as we're going to use gradient checkpointing
            device_map=device_map,
            quantization_config=None,
            config=config,
            cache_dir="../llms",
        )
    model = get_model(checkpoint_path, model_kwargs)
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    for batch_size in range(1, 406):

        training_args.per_device_eval_batch_size = batch_size
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
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