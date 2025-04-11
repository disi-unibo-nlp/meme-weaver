import sys
sys.path.append('./')

import torch
# Set PyTorch to use deterministic algorithms for CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import logging
import os
import sys
import json
import math
import wandb

import datasets
from peft import LoraConfig
from datasets import load_dataset
from codecarbon import EmissionsTracker


import transformers
from transformers import (
    BitsAndBytesConfig,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    MBartTokenizer,
    MBartTokenizerFast,
    get_scheduler,
    Trainer,
)

from arguments import ModelArguments, DataTrainingArguments
from utils import predict_class, get_model, compute_metrics
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from peft import get_peft_model, prepare_model_for_kbit_training

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.30.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")

logger = logging.getLogger(__name__)

    
def get_carburacy(score, emission_train, emission_test, alpha=10, beta_train=1, beta_test=100):
    score = score + sys.float_info.epsilon
    carburacy_train = None
    if emission_train is not None:
        carburacy_train = math.exp(math.log(score/100, alpha)) / (1 + emission_train * beta_train)
    carburacy_test = None
    if emission_test is not None:
        carburacy_test = math.exp(math.log(score/100, alpha)) / (1 + emission_test * beta_test)
    carburacy = None
    if carburacy_train is not None and carburacy_test is not None:
        carburacy = (2 * carburacy_train * carburacy_test) / (carburacy_train + carburacy_test)
    return carburacy_train, carburacy_test, carburacy


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    training_args.output_dir += "/" + training_args.run_name

    assert not os.path.exists(training_args.output_dir), "Output directory already exists"
    
    wandb.init(mode=data_args.logging,
            name=training_args.run_name,
            project=data_args.dataset_name.split("/")[1] + f"_{data_args.dataset_subset}",
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    
    import numpy
    import random
    numpy.random.seed(seed=training_args.seed)
    random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_subset,
        # download_mode="force_redownload",
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )


    label_list = list(set(raw_datasets["train"][data_args.target_column]))
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir="../llms",
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        # llm_int8_skip_modules=["rs_gcn_layers"]
        )

    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    # Custom config hyperparameters
    config.num_gcn_layers = model_args.num_gcn_layers
    config.custom_gcn = model_args.custom_gcn
    config.save_affinity = model_args.save_affinity
    config.apply_ffw = model_args.apply_ffw
    config.output_dir = training_args.output_dir

    model_kwargs = dict(
            torch_dtype="auto",
            # use_cache=False, # set to False as we're going to use gradient checkpointing
            device_map=device_map,
            quantization_config= None if model_args.no_peft else quantization_config,
            config=config,
            use_flash_attention_2=False,
            cache_dir="../llms",
        )
    
    if training_args.do_train:
        
        model = get_model(model_args.model_name_or_path, model_kwargs)
        if not model_args.no_peft:
            # based on config
            peft_config = LoraConfig(
                    r=64,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    bias="none",
                    task_type="SEQ_CLS",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "phi", "gamma", "W_g", "W_r"],
            )

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

            # add LoRA adaptor
            model = get_peft_model(model, peft_config)
    else:
        # Find the folder starting with "checkpoint-"
        checkpoint_folder = next(folder for folder in os.listdir(training_args.output_dir) if folder.startswith("checkpoint-"))
        # Append the checkpoint folder to the base path
        checkpoint_path = os.path.join(training_args.output_dir, checkpoint_folder)
        model = get_model(checkpoint_path, model_kwargs)
        
    if model.config.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # TODO: check if also the other models need such settings
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_seq_length
            and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast))
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_seq_length}."
            )
            model.resize_position_embeddings(data_args.max_seq_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_seq_length)
        else:
            raise ValueError(
                f"`--max_seq_length` is set to {data_args.max_seq_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_seq_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_seq_length
    ):
        if model_args.resize_position_embeddings is None and "umberto" not in model_args.model_name_or_path:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_seq_length}."
            )
            model.resize_position_embeddings(data_args.max_seq_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_seq_length)
        else:
            raise ValueError(
                f"`--max_seq_length` is set to {data_args.max_seq_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_seq_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs1, targets = [], []
        inputs2 = None
        for i in range(len(examples[data_args.input_column])):
            inputs1.append(examples[data_args.input_column][i])
            targets.append(examples[data_args.target_column][i])
        
        if data_args.add_caption: 
            inputs1 = [inp + "[CPT]" + cpt  for inp, cpt in zip(inputs1, examples["qwen25vl_caption"])]

        inputs1 = [prefix + inp for inp in inputs1]
        # Tokenize the texts
        args = ((inputs1,) if inputs2 is None else (inputs1, inputs2))
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        # Map labels to IDs
        if label_to_id is not None and data_args.target_column in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples[data_args.target_column]]
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None    

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
            #load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",)

        print(f"Training set size: {len(train_dataset)}")

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
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
    else:
        optimizers = (None, None)

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.shuffle(seed=training_args.seed).select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.shuffle(seed=training_args.seed).select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    
    # Compute frequency of evaluation
    n_steps = len(train_dataset)/training_args.per_device_train_batch_size * training_args.num_train_epochs
    training_args.eval_steps = n_steps // 8
    training_args.save_steps = n_steps // 8

    from transformers import AutoModelForSequenceClassification
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        # model_init=model_init,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=optimizers,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
        train_tracker.start()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_emissions = train_tracker.stop()
        trainer.save_model()  # Saves the tokenizer too for easy upload

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

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        eval_metrics = predict_class(trainer, eval_dataset, max_eval_samples, training_args, "eval")
        wandb.log(eval_metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        predict_metrics = predict_class(trainer, predict_dataset, max_predict_samples, training_args, "predict")
        wandb.log(predict_metrics)

if __name__ == "__main__":
    main()
