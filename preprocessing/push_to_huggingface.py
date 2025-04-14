import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from googletrans import Translator
from datasets import Dataset, DatasetDict, Features, Value, Image, concatenate_datasets

CAPTION_MODEL = "qwen25vl"


def get_image_path(path_to_meme, meme):
    path = os.path.join(path_to_meme, meme)

    return path


def process_exist_2024():

    def translate_text(text):

        return translator.translate(text, dest="en", src='es').text

    def get_captions(meme):
        caption = caption_csv[caption_csv['meme_id'] == meme]['caption'].values[0]

        return caption


    def majority_vote(labels):
        # Count the number of "YES" responses
        yes_count = labels.count('YES')
        # If the number of YES is greater than or equal to half the total, return 1 (covers tie as well)
        return 1 if yes_count >= len(labels) / 2 else 0

    translator = Translator()
    tqdm.pandas()

    features = Features({
        "id_EXIST": Value("string"),
        "lang": Value("string"),
        "text": Value("string"),
        "text_en": Value("string"),
        f'{CAPTION_MODEL}_caption': Value("string"),
        "hard_label_task4": Value("int8"),
        "image_path": Image()  # This will ensure that image data is handled correctly.
    })

    data_path = os.path.join("exist2024_memes_dataset", "training", "EXIST2024_training.json")
    with open(data_path, 'r', encoding='utf-8') as file:
        data_labelled = json.load(file)

    df_labelled  = pd.DataFrame.from_dict(data_labelled, orient='index')
    df_labelled['hard_label_task4'] = df_labelled['labels_task4'].apply(majority_vote)
    path_to_meme = os.path.join("exist2024_memes_dataset", "training", "memes")
    df_labelled['image_path'] = df_labelled['meme'].apply(get_image_path, path_to_meme)

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
    caption_csv = pd.read_csv(os.path.join("data", f'{CAPTION_MODEL}_captions_training.csv'))

    # meme_id to string
    caption_csv['meme_id'] = caption_csv['meme_id'].astype(str)
    df_labelled[f'{CAPTION_MODEL}_caption'] = df_labelled['id_EXIST'].apply(get_captions)

    df_labelled = df_labelled[["id_EXIST", "lang", "text", 'text_en', "hard_label_task4", "image_path", f'{CAPTION_MODEL}_caption']]

    df_shuffled = df_labelled.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df_shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)  # for validation

    # Split the DataFrame
    train_df = df_shuffled.iloc[:n_train]
    val_df = df_shuffled.iloc[n_train:n_train + n_val]
    test_df = df_shuffled.iloc[n_train + n_val:]

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


def process_mami():
    features = Features({
        "file_name": Value("string"),
        "label": Value("int8"),
        "shaming": Value("int8"),
        "stereotype": Value("int8"),
        "objectification": Value("int8"),
        "violence": Value("int8"),
        "text": Value("string"),
        "image_path": Image()  # This will ensure that image data is handled correctly.
    })

    test_df = pd.read_csv(os.path.join("MAMI_Dataset", "test.tsv"), sep="\t")
    train_df = pd.read_csv(os.path.join("MAMI_Dataset", "train.tsv"), sep="\t")
    val_df = pd.read_csv(os.path.join("MAMI_Dataset", "validation.tsv"), sep="\t")

    path_to_meme = os.path.join("MAMI_Dataset", "MAMI_2022_images")
    test_df["image_path"] = test_df["file_name"].apply(lambda x: os.path.join(path_to_meme, x))
    train_df["image_path"] = train_df["file_name"].apply(lambda x: os.path.join(path_to_meme, x))
    val_df["image_path"] = val_df["file_name"].apply(lambda x: os.path.join(path_to_meme, x))

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

    if args.dataset == "memes_exist2024":
        process_exist_2024()
    elif args.dataset == "mami":
        process_mami()
    else:
        raise ValueError("Dataset not supported")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["exist2024", "mami"], required=True)

    args = parser.parse_args()

    main()