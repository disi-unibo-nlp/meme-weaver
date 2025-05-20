import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from googletrans import Translator
from datasets import Dataset, DatasetDict, Features, Value, Image, concatenate_datasets





def get_captions(meme, prompt, caption_csv):
    caption = caption_csv[caption_csv['meme_id'] == meme][prompt].values[0]

    return caption

def add_captions(df, csv, caption_csv_id, prompts=('promptA','promptB')):
    for p in prompts:
        df[f'caption_{p}'] = (
            df[caption_csv_id]
              .apply(lambda id_val: get_captions(id_val, p, csv))
        )
    return df

def add_image_path(df, root):
    df['image'] = df["id"].apply(lambda x: os.path.join(root, x))
    return df


def prepare_df(df, csv, root, caption_csv_id, prompts):
    df = add_captions(df, csv, caption_csv_id, prompts)
    df = add_image_path(df, root)
    return df


def get_exist_data(data_path, translator, split):
    def translate_text(text):

        return translator.translate(text, dest="en", src='es').text

    def majority_vote(labels):
        # Count the number of "YES" responses
        yes_count = labels.count('YES')
        # If the number of YES is greater than or equal to half the total, return 1 (covers tie as well)
        return 1 if yes_count >= len(labels) / 2 else 0

    with open(data_path, 'r', encoding='utf-8') as file:
        data_labelled = json.load(file)

    df_labelled  = pd.DataFrame.from_dict(data_labelled, orient='index')
    if split == "training":
        df_labelled['hard_label_task4'] = df_labelled['labels_task4'].apply(majority_vote)
    df_labelled = df_labelled.rename(columns={"meme": "id"})

    if not "text_en" in df_labelled.columns:
        print("Translating texts...")
        df_labelled['text_en'] = df_labelled.progress_apply(
            lambda row: translate_text(row['text']) if row['lang'] == 'es' else row['text'],
            axis=1
        )

    for _, row in df_labelled.iterrows():
        id_ = row["id_EXIST"]
        data_labelled[id_]["text_en"] = row["text_en"]

    with open(data_path, 'w', encoding='utf-8') as file:
        json.dump(data_labelled, file, ensure_ascii=False)

    # Load the CSV file containing the captions
    caption_csv = pd.read_csv(os.path.join("data", args.caption_model,  f'exist.csv'))
    caption_csv['meme_id'] = caption_csv['meme_id'].astype(str)

    path_to_meme = os.path.join("exist2024_memes_dataset", split, "memes")
    prepare_df(df_labelled, caption_csv, path_to_meme, "id_EXIST", ('promptA', 'promptB'))
    if split == "training":
        df_labelled = df_labelled[["id", "lang", "text", 'text_en', "hard_label_task4", "image", "caption_promptA", "caption_promptB"]]
    else:
        df_labelled = df_labelled[["id", "lang", "text", 'text_en', "image", "caption_promptA", "caption_promptB"]]

    df_shuffled = df_labelled.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_shuffled



def process_exist_2024():

    translator = Translator()
    tqdm.pandas()

    features = Features({
        "id": Value("string"),
        "lang": Value("string"),
        "text": Value("string"),
        "text_en": Value("string"),
        f'caption_promptA': Value("string"),
        f'caption_promptB': Value("string"),
        "hard_label_task4": Value("int8"),
        "image": Image()  # This will ensure that image data is handled correctly.
    })

    data_path_experiments = os.path.join("exist2024_memes_dataset", "training", "EXIST2024_training.json")
    df_experiments = get_exist_data(data_path_experiments, translator, "training")

    n = len(df_experiments)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)  # for validation

    # Split the DataFrame
    train_df = df_experiments.iloc[:n_train]
    val_df = df_experiments.iloc[n_train:n_train + n_val]
    test_df = df_experiments.iloc[n_train + n_val:]

    dataset_train = Dataset.from_pandas(train_df, features=features)
    dataset_val = Dataset.from_pandas(val_df, features=features)
    dataset_test = Dataset.from_pandas(test_df, features=features)

    # add challenge split
    data_path_challenge = os.path.join("exist2024_memes_dataset", "test", "EXIST2024_test_clean.json")
    df_challenge = get_exist_data(data_path_challenge, translator, "test")

    test_challenge_df = Dataset.from_pandas(df_challenge, features=features)

    # Create a DatasetDict with your splits.
    dataset = DatasetDict({
        "train": dataset_train,
        "validation": dataset_val,
        "test": dataset_test,
        "test_challenge": test_challenge_df,
    })

    # Now, push the dataset to your Hugging Face Hub repository.
    # Replace "your_dataset_name" with your desired repository name.
    # You can also pass your Hugging Face token here or have it set in your environment.
    dataset.push_to_hub("paoloitaliani/memes_exist2024")


def process_mami():
    features = Features({
        "file_name": Value("string"),
        "label": Value("int8"),
        "shaming": Value("int8"),
        "stereotype": Value("int8"),
        f'caption_promptA': Value("string"),
        f'caption_promptB': Value("string"),
        "objectification": Value("int8"),
        "violence": Value("int8"),
        "text": Value("string"),
        "image": Image()  # This will ensure that image data is handled correctly.
    })

    test_df = pd.read_csv(os.path.join("MAMI_Dataset", "test.tsv"), sep="\t")
    train_df = pd.read_csv(os.path.join("MAMI_Dataset", "train.tsv"), sep="\t")
    val_df = pd.read_csv(os.path.join("MAMI_Dataset", "validation.tsv"), sep="\t")

    test_df = test_df.rename(columns={"id": "meme"})
    train_df = train_df.rename(columns={"id": "meme"})
    val_df = val_df.rename(columns={"id": "meme"})


    path_to_meme = os.path.join("MAMI_Dataset", "MAMI_2022_images")
    caption_csv = pd.read_csv(os.path.join("data", args.caption_model, f'mami.csv'))

    test_df = prepare_df(test_df, caption_csv, path_to_meme, "file_name", ('promptA', 'promptB'))
    train_df = prepare_df(train_df, caption_csv, path_to_meme, "file_name", ('promptA', 'promptB'))
    val_df = prepare_df(val_df, caption_csv, path_to_meme, "file_name", ('promptA', 'promptB'))

    dataset_train = Dataset.from_pandas(train_df, features=features)
    dataset_val = Dataset.from_pandas(val_df, features=features)
    dataset_test = Dataset.from_pandas(test_df, features=features)

    dataset = DatasetDict({
        "train": dataset_train,
        "validation": dataset_val,
        "test": dataset_test,
    })

    dataset.push_to_hub("paoloitaliani/mami")


def main():

    if args.dataset == "exist":
        process_exist_2024()
    elif args.dataset == "mami":
        process_mami()
    else:
        raise ValueError("Dataset not supported")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["exist", "mami"], required=True)
    parser.add_argument("--caption_model", type=str, default="qwen25vl_prompting")

    args = parser.parse_args()

    main()