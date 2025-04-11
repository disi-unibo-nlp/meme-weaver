import os
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset


def get_dataloader(config, split):

    dataset = MemesDataset(split, task_id=config.task)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=config.training_settings['batch_size'],
        num_workers=config.training_settings['num_workers'],
        pin_memory=True,
    )

    return dataloader


class EXISTDataset(Dataset):
    def __init__(self, split, task_id='hard_label_task4'):

        emebddings_path = os.path.join("data", "embeddings", "xlm_roberta_large", f"{split}_embeddings.npz")
        embeddings_data = np.load(emebddings_path)
        embeddings_array = embeddings_data["embeddings"]
        embedding_ids = embeddings_data["ids"]
        self.embeddings = {id_val: emb for id_val, emb in zip(embedding_ids, embeddings_array)}

        self.dataset = load_dataset("paoloitaliani/memes_exist2024", split=split)
        self.ids = self.dataset['id_EXIST']
        self.texts = self.dataset['text']
        self.labels = self.dataset[task_id]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        
        sample_id = self.ids[index]
        batch_sample = {'id': self.ids[index],
                        'text': self.texts[index],
                        'label': self.labels[index],
                        'embedding': self.embeddings[sample_id],
                    }

        return batch_sample
    

class MAMIDataset(Dataset):
    def __init__(self, split, label_key='label',
                 tokenizer=None, data_args=None, label_to_id=None,
                 padding="max_length", max_seq_length=128):
        """
        Parameters:
            split (str): The dataset split (e.g., "train", "validation", "test").
            label_key (str): The key for the target labels.
            tokenizer: A tokenizer instance to convert text to tokens.
            data_args: An object (or dictionary) containing:
                - input_column: The column name for input text (e.g., "text").
                - target_column: The column name for labels.
                - add_caption: Boolean flag to decide if a caption should be appended.
                - prefix: A string prefix to add before the input.
            label_to_id (dict): A mapping from the label values to their IDs.
            padding (str): Padding strategy to use during tokenization.
            max_seq_length (int): Maximum sequence length for tokenization.
        """

        # Load the dataset.
        self.dataset = load_dataset("paoloitaliani/mami", split=split)
        self.ids = self.dataset['file_name']
        self.texts = self.dataset['text']
        self.labels = self.dataset[label_key]

        # Save the provided parameters.
        self.tokenizer = tokenizer
        self.data_args = data_args  # Should include input_column, target_column, add_caption, prefix.
        self.label_to_id = label_to_id
        self.padding = padding
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Retrieve the raw input text and target label.
        raw_input = self.texts[index]
        target = self.labels[index]

        # Tokenize the processed input.
        result = self.tokenizer(
            raw_input,
            padding=self.padding,
            max_length=self.max_seq_length,
            truncation=True
        )

        # Map labels to IDs (handling the special case where target == -1).
        result["label"] = self.label_to_id[target] if target != -1 else -1

        return result   