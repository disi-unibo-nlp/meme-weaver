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
    EvalPrediction,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from typing import Optional
from dataclasses import dataclass, field
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.utils import get_model, predict_class


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for tasks."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_name_local: Optional[str] = field(
        default=None, metadata={"help": "The name of the local dataset to use."}
    )
    input_column: Optional[str] = field(
        default="input",
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    target_column: Optional[str] = field(
        default="output",
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    logging : Optional[str] = field(
        default="disabled",
        metadata={
            "help": (
                "Set 'disabled' to disable wandb logging, or else select logging 'online' or 'offline'"
            )
        },
    )
    dataset_subset: Optional[str] = field(
        default=None, metadata={"help": "The subset of the dataset to use."}
    )   
    add_caption: bool = field(
        default=False,
        metadata={"help": "Add references to the input."},
    )
    split: Optional[str] = field(
        default=None, metadata={"help": "The split of the dataset to use (train, test, validation)."}
    ) 


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    no_peft: bool = field(
        default=False,
        metadata={"help": "Do not use PEFT."},
    )
    num_gcn_layers: Optional[int] = field(
        default=0,
        metadata={"help": "The number of Rs_GCN layers to use."},
    )
    save_affinity: bool = field(
        default=False,
        metadata={"help": "Do not use PEFT."},
    )

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

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = {
            "precision": round(100 * precision_score(p.label_ids, preds), 2),
            "recall": round(100 * recall_score(p.label_ids, preds), 2),
            "F1": round(100 * f1_score(p.label_ids, preds), 2),
            "accuracy": round(100 * accuracy_score(p.label_ids, preds), 2),
        }
        
        return result
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)

        return pred_ids
    
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

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir="../llms",
        trust_remote_code=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    output_path = os.path.join(training_args.output_dir, training_args.run_name)

    # Custom config hyperparameters
    config.num_gcn_layers = model_args.num_gcn_layers
    config.save_affinity = model_args.save_affinity
    config.output_dir = output_path 

    model_kwargs = dict(
            torch_dtype="auto",
            # use_cache=False, # set to False as we're going to use gradient checkpointing
            device_map=device_map,
            quantization_config=None,
            config=config,
            cache_dir="../llms",
        )

    # Find the folder starting with "checkpoint-"
    checkpoint_folder = next(folder for folder in os.listdir(output_path) if folder.startswith("checkpoint-"))
    # Append the checkpoint folder to the base path
    checkpoint_path = os.path.join(output_path, checkpoint_folder)
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
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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