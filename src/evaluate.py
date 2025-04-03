import sys
sys.path.append('./')

import os 
import torch
import argparse

from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    DataCollatorWithPadding,
)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.utils import get_model

def main():
    raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_subset,
            # download_mode="force_redownload",
        )
    
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    predict_dataset = raw_datasets["test"]
    column_names = raw_datasets["test"].column_names
    predict_dataset = predict_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on prediction dataset",
    )

    label_list = list(set(raw_datasets["train"][args.target_column]))
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        cache_dir="../llms",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    # Custom config hyperparameters
    config.num_gcn_layers = args.num_gcn_layers

    model_kwargs = dict(
            torch_dtype="auto",
            # use_cache=False, # set to False as we're going to use gradient checkpointing
            device_map=device_map,
            quantization_config=None,
            config=config,
            cache_dir="../llms",
        )

    # Find the folder starting with "checkpoint-"
    checkpoint_folder = next(folder for folder in os.listdir(args.output_dir) if folder.startswith("checkpoint-"))
    # Append the checkpoint folder to the base path
    checkpoint_path = os.path.join(args.output_dir, checkpoint_folder)
    model = get_model(checkpoint_path, model_kwargs)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs1, targets = [], []
        inputs2 = None
        for i in range(len(examples[args.input_column])):
            inputs1.append(examples[args.input_column][i])
            targets.append(examples[args.target_column][i])
        
        if args.add_caption: 
            inputs1 = [inp + "[CPT]" + cpt  for inp, cpt in zip(inputs1, examples["qwen25vl_caption"])]

        # Tokenize the texts
        args = ((inputs1,) if inputs2 is None else (inputs1, inputs2))
        result = tokenizer(*args, padding=False, truncation=True)
        # Map labels to IDs
        if label_to_id is not None and args.target_column in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples[args.target_column]]
        return result

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = {
            "precision": round(100 * precision_score(p.label_ids, preds), 2),
            "recall": round(100 * recall_score(p.label_ids, preds), 2),
            "F1": round(100 * f1_score(p.label_ids, preds), 2),
            "accuracy": round(100 * accuracy_score(p.label_ids, preds), 2),
        }
        
        return result
    
    trainer = Trainer(
        model=model,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dataset_subset", type=str, required=True, help="Subset of the dataset")
    parser.add_argument("--target_column", type=str, required=True, help="Target column name")
    parser.add_argument("--input_column", type=str, required=True, help="Input column name")
    parser.add_argument("--add_caption", action="store_true", help="Add caption to the input")
    parser.add_argument("--no_peft", action="store_true", help="Disable PEFT")
    parser.add_argument("--num_gcn_layers", type=int, default=1, help="Number of GCN layers")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    main()