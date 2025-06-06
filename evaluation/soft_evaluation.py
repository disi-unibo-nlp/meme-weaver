import os
import json
import argparse

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from tqdm import tqdm
from datasets import load_dataset

def main():
    path_to_json = os.path.join(args.output_dir, args.run_name, "generated_predict_set.json")
    with open(path_to_json, "r") as f:
        predictions = json.load(f)
    
    predict_dataset = load_dataset(args.dataset_name, split="test")
    predict_df = predict_dataset.to_pandas()
    predict_df['id'] = predict_df['id'].str.split(".", n=1).str[0]

    hard_predictions = []
    hard_targets = []
    for pred in tqdm(predictions):

        hard_target = predict_df[predict_df["id"] == pred['id']]["hard_label_task4"]
        hard_pred = 1 if pred["value"] >= 0.5 else 0

        hard_predictions.append(hard_pred)
        hard_targets.append(hard_target.iloc[0])

    result = {
        "precision_macro": round(100 * precision_score(hard_targets, hard_predictions, average='macro'), 2),
        "recall_macro": round(100 * recall_score(hard_targets, hard_predictions, average='macro'), 2),
        "F1_macro": round(100 * f1_score(hard_targets, hard_predictions, average='macro'), 2),
        "accuracy": round(100 * accuracy_score(hard_targets, hard_predictions), 2),
    }

    print(result)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_dir", default="output_hard_label_task4")
    parser.add_argument("--dataset_name", default="paoloitaliani/memes_exist2024")

    args = parser.parse_args()
    main()