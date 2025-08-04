import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from collections import Counter
from googletrans import Translator
from datasets import Dataset, DatasetDict, Features, Sequence, Value, Image

LABELS_TASK6 = [
    "IDEOLOGICAL-INEQUALITY",
    "STEREOTYPING-DOMINANCE",
    "OBJECTIFICATION",
    "SEXUAL-VIOLENCE",
    "MISOGYNY-NON-SEXUAL-VIOLENCE"
]

# 2. Build a lookup dict
LABELS_TOIDX_TASK6 = {label: idx for idx, label in enumerate(LABELS_TASK6)}
LABELS_TOIDX_TASK6['UNKNOWN'] = None



LABELS_MAMI = ["non misogynist", "shaming", "stereotype", "objectification", "violence"]
LABELS_TOIDX_MAMI = {lbl: i for i, lbl in enumerate(LABELS_MAMI)}


def add_image_path(df, root, inst_id):
    df['image'] = df[inst_id].apply(lambda x: os.path.join(root, f"{x}"))


def add_captions_via_map(df, caption_csv, caption_csv_id, prompts=('promptA','promptB')):
    # Build one dict per prompt: { meme_id → caption_string }
    mappings = {
        p: caption_csv.set_index('meme_id')[p].to_dict()
        for p in prompts
    }
    # Then vectorized map
    for p in prompts:
        df[f'caption_{p}'] = df[caption_csv_id].map(mappings[p])



def prepare_df(df, csv, root, caption_csv_id, prompts, inst_id='id'):
    add_captions_via_map(df, csv, caption_csv_id, prompts)
    add_image_path(df, root, inst_id)


def get_exist_data(data_path, translator, split):
    def translate_text(text):

        return translator.translate(text, dest="en", src='es').text

    def majority_vote(labels):
        # Count the number of "YES" responses
        yes_count = labels.count('YES')
        # If the number of YES is greater than or equal to half the total, return 1 (covers tie as well)
        return 1 if yes_count >= len(labels) / 2 else 0

    def majority_vote_multilabel(labels):

        # Count occurrences of each class (ignoring "-")
        counter = Counter(
            label
            for annotator_labels in labels
            for label in annotator_labels
            if label != "-"
        )

        # Select classes annotated by more than 1 annotator
        hard_labels = [label for label, count in counter.items() if count > 1]
        # start with all zeros
        hard_label_onehot = [0] * len(LABELS_TASK6)
        # set 1 for each present label
        for label in hard_labels:
            idx = LABELS_TOIDX_TASK6.get(label)
            if idx is not None:
                hard_label_onehot[idx] = 1

        # If no class meets the threshold, the item would be removed
        if hard_labels:
            return hard_label_onehot
        else:
            return None


    def soft_labels(labels):
        # Count the number of "YES" responses
        yes_count = labels.count('YES')
        return yes_count / len(labels)

    with open(data_path, 'r', encoding='utf-8') as file:
        data_labelled = json.load(file)

    df_labelled  = pd.DataFrame.from_dict(data_labelled, orient='index')
    if split == "training":
        df_labelled['hard_label_task4'] = df_labelled['labels_task4'].apply(majority_vote)
        df_labelled['hard_label_task6'] = df_labelled['labels_task6'].apply(majority_vote_multilabel)
        df_labelled["soft_label_task4"] = df_labelled['labels_task4'].apply(soft_labels)
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
        df_labelled = df_labelled[["id", "lang", "text", 'text_en', "soft_label_task4", "hard_label_task4", "hard_label_task6", "image", "caption_promptA", "caption_promptB"]]
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
        "hard_label_task6": Sequence(feature=Value("int8")),  # Multilabel task
        "soft_label_task4": Value("float32"),
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
    dataset.push_to_hub("exist_hf_repo")




def process_mmhs150k():

    features = Features({
        "tweet_id": Value("string"),
        "label": Value("int8"),
        "text": Value("string"),
        "all_labels": Sequence(feature=Value("int8")),
        "image": Image(),
        f'caption_promptA': Value("string"),
        f'caption_promptB': Value("string"),
    })

    data_path = os.path.join("MMHS150K", "MMHS150K_GT.json")

    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    train_ids_path = os.path.join("MMHS150K", "splits", "train_ids.txt")
    val_ids_path = os.path.join("MMHS150K", "splits", "val_ids.txt")
    test_ids_path = os.path.join("MMHS150K", "splits", "test_ids.txt")
    with open(train_ids_path, 'r') as file:
        train_ids = [line.strip() for line in file.readlines()]
    with open(val_ids_path, 'r') as file:
        val_ids = [line.strip() for line in file.readlines()]
    with open(test_ids_path, 'r') as file:
        test_ids = [line.strip() for line in file.readlines()]

    inst_ids = []
    labels = []
    all_labels = []
    texts = []
    for inst_id, inst_data in data.items():
        ann_labels = inst_data['labels']
        binary_ann_labels = [1 if label >= 1 else 0 for label in ann_labels]

        yes_count = binary_ann_labels.count(1)
        majority_label = 1 if yes_count >= len(binary_ann_labels) / 2 else 0

        complete_text = inst_data['tweet_text']

        text_path = os.path.join("MMHS150K", "img_txt", f"{inst_id}.json")
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as file:
                text_data = json.load(file)

            complete_text += " [SEP] " + text_data["img_text"]
            complete_text = complete_text.strip()

        texts.append(complete_text)
        labels.append(majority_label)
        all_labels.append(ann_labels)
        inst_ids.append(inst_id)

    data_df = pd.DataFrame({
        "tweet_id": inst_ids,
        "label": labels,
        "text": texts,
        "all_labels": all_labels,
    }).astype({"label": "int8"})

    path_to_meme = os.path.join("MMHS150K", "img_resized")
    caption_csv = pd.read_csv(os.path.join("data", args.caption_model, f'mmhs150k.csv'))
    caption_csv["meme_id"] = caption_csv["meme_id"].astype(str)

    prepare_df(data_df, caption_csv, path_to_meme, "tweet_id", ('promptA', 'promptB'), "tweet_id")

    # Split out each set
    train_df = data_df[data_df['tweet_id'].isin(train_ids)].copy().reset_index(drop=True)
    val_df = data_df[data_df['tweet_id'].isin(val_ids)].copy().reset_index(drop=True)
    test_df = data_df[data_df['tweet_id'].isin(test_ids)].copy().reset_index(drop=True)

    dataset_train = Dataset.from_pandas(train_df, features=features)
    dataset_val = Dataset.from_pandas(val_df, features=features)
    dataset_test = Dataset.from_pandas(test_df, features=features)

    dataset = DatasetDict({
        "train": dataset_train,
        "validation": dataset_val,
        "test": dataset_test,
    })

    dataset.push_to_hub("mmhs150k_hf_repo")

def get_multilabel_idx(row):
    if row["label"] == 0:
        # non-misogynist → index 0
        return [label_to_idx["non misogynist"]]
    # otherwise collect any of the 4 flags that are 1
    idxs = [label_to_idx[c] for c in cats if row[c] == 1]
    # fallback to “non misogynist” if none are flagged
    return idxs or [label_to_idx["non misogynist"]]


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

    dataset.push_to_hub("mami_hf_repo")


def main():

    if args.dataset == "exist":
        process_exist_2024()
    elif args.dataset == "mami":
        process_mami()
    elif args.dataset == "mmhs150k":
        process_mmhs150k()
    else:
        raise ValueError("Dataset not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--caption_model", type=str, default="qwen25vl_prompting")

    args = parser.parse_args()

    main()