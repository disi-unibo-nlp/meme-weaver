import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Image


def get_image_path(meme):
    path = os.path.join("exist2024_memes_dataset", "training", "memes", meme)

    return path

def majority_vote(labels):
    # Count the number of "YES" responses
    yes_count = labels.count('YES')
    # If the number of YES is greater than or equal to half the total, return 1 (covers tie as well)
    return 1 if yes_count >= len(labels) / 2 else 0


features = Features({
    "id_EXIST": Value("string"),
    "lang": Value("string"),
    "text": Value("string"),
    "hard_label_task4": Value("int8"),
    "image_path": Image()  # This will ensure that image data is handled correctly.
})

data_path = os.path.join("exist2024_memes_dataset", "training", "EXIST2024_training.json")
with open(data_path, 'r', encoding='utf-8') as file:
    data_labelled = json.load(file)

df_labelled  = pd.DataFrame.from_dict(data_labelled, orient='index')
df_labelled['hard_label_task4'] = df_labelled['labels_task4'].apply(majority_vote)
df_labelled['image_path'] = df_labelled['meme'].apply(get_image_path)
df_labelled = df_labelled[["id_EXIST", "lang", "text", "hard_label_task4", "image_path"]]

df_shuffled = df_labelled.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df_shuffled)
n_train = int(0.8 * n)
n_val = int(0.1 * n)  # for validation

# Split the DataFrame
train_df = df_shuffled.iloc[:n_train]
val_df = df_shuffled.iloc[n_train:n_train+n_val]
test_df = df_shuffled.iloc[n_train+n_val:]

dataset_train = Dataset.from_pandas(train_df, features=features)
dataset_val = Dataset.from_pandas(val_df, features=features)
dataset_test = Dataset.from_pandas(test_df, features=features)

# Create a DatasetDict with your splits.
dataset = DatasetDict({
    "train": dataset_train,
    "validation": dataset_val,
    "test": dataset_test,
})


# Now, push the dataset to your Hugging Face Hub repository.
# Replace "your_dataset_name" with your desired repository name.
# You can also pass your Hugging Face token here or have it set in your environment.
dataset.push_to_hub("paoloitaliani/memes_exist2024")

