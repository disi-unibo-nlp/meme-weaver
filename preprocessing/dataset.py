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


class MemesDataset(Dataset):
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

   