import sys
sys.path.append('./')

import os
import yaml
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from colorama import Fore
from preprocessing.dataset import get_dataloader
from utils import get_optimizer_and_scheduler, evaluate_model

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from memes_graph_enhanced.models.simple_classifier import SimpleClassifier
from models.xlm_roberta_classifier import XLMRobertaForSequenceClassification


def train(config):
    model.train()
    optimizer.zero_grad()
    train_stats = {'loss': 0.0} 

    step = 0
    for batch in tqdm(train_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

        # -- forward pass
        model_output = model(batch)

        # -- optimization
        model_output['loss'].backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss = model_output['loss'].item()

        train_stats['loss'] += loss
        step += 1

    train_stats['loss'] = train_stats['loss'] / len(train_loader)

    return train_stats

def evaluate(args, config, eval_loader, test_for_submission=False):
    model.eval()
    eval_output = {'id': [], 'logits': [], 'probs': [], 'preds': [], 'labels': [], 'loss': 0.0}

    with torch.no_grad():
        for batch in tqdm(eval_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

            # -- forward pass
            model_output = model(batch)

            for eval_key in eval_output.keys():
                if eval_key == "id":
                    eval_output[eval_key] += batch[eval_key]
                else:
                    eval_output[eval_key] += model_output[eval_key].detach().cpu().numpy().tolist()

        report = evaluate_model(eval_output)

    return report

def main(config):

    import numpy
    import random
    numpy.random.seed(seed=42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set seed before initializing model.
    set_seed(42)

    label_list = list(set(raw_datasets["train"][data_args.target_column]))
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        "FacebookAI/xlm-roberta-large",
        num_labels=num_labels,
        cache_dir="../llms",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    # Custom config hyperparameters
    config.num_gcn_layers = 0
    config.custom_gcn = "None"
    config.save_affinity = False
    config.apply_ffw = False
    config.output_dir = None

    model_kwargs = dict(
            torch_dtype="auto",
            # use_cache=False, # set to False as we're going to use gradient checkpointing
            device_map=device_map,
            quantization_config= None,
            config=config,
            use_flash_attention_2=False,
            cache_dir="../llms",
        )
    

    # -- building model architecture
    global model, optimizer, scheduler, train_loader, val_loader, test_loader
    # model = SimpleClassifier(config).to(config.device)
    model = XLMRobertaForSequenceClassification("FacebookAI/xlm-roberta-large", **model_kwargs)

    # -- training process
    if args.mode == "train":

        # -- creating data loaders
        train_loader = get_dataloader(config, "train")
        val_loader = get_dataloader(config, "validation")
        test_loader = get_dataloader(config, "test")

        # -- defining the optimizer and its scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(config, model, train_loader)

        for epoch in range(1, config.training_settings['epochs']+1):
            train_stats = train(config)
            print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],4)}")
            report = evaluate(args, config, val_loader)
            print(report)

            #print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],4)} || VAL LOSS={round(val_output['loss'],4)} | VAL ICM-NORM={round(val_output['icm-norm'],2)}%")

    if args.mode in ['evaluation', 'both']:
        val_loader = get_dataloader(config, args.validation_dataset, is_training=False)
        val_output = evaluate(args, config, val_loader)

        test_loader = get_dataloader(config, args.test_dataset, is_training=False)
        test_output = evaluate(args, config, test_loader, test_for_submission=True)

        # -- saving model output
        if args.save:
            save_model_output(val_output, args.output_dir, 'validation')
            save_model_output(test_output, args.output_dir, 'test')

        # -- displaying final report
        print(f'\nVALIDATION REPORT:')
        print(val_output['pyevall-report'].print_report())

        # eval_report = classification_report(
        #     val_output['preds'],
        #     val_output['labels'],
        # )
        #     target_names=config.class_names,
        # print(eval_report)

    return val_output

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', required=True, type=str, help='Configuration file name')
    parser.add_argument('--mode', default="train", type=str, choices=['train', "test"], help='Mode of the script')
    args = parser.parse_args()

    # -- loading configuration file
    config_path = os.path.join('configs', args.config_file)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config = argparse.Namespace(**config)

    main(config)