import sys
sys.path.append('./')

import os
import pickle
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import plotly.express as px
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer, util

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import numpy as np

from src.utils import get_model

def get_most_similar_elements(affinity_matrix, top_k):
    # Flatten the matrix
    flat_affinity = affinity_matrix.flatten()

    # Use torch.topk to find the top 10 values and their indices
    topk_values, topk_indices = torch.topk(flat_affinity, top_k)

    # Calculate row and column indices from the flattened index
    num_cols = affinity_matrix.shape[1]
    couples = []
    for flat_idx, value in zip(topk_indices, topk_values):
        row = flat_idx // num_cols
        col = flat_idx % num_cols
        couples.append((row.item(), col.item(), value.item()))


def get_cls_embeddings(texts, tokenizer, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.roberta(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of layers

        # Usually the last hidden state of [CLS] token is used
        cls_embeddings = hidden_states[-1][:, 0, :]  # shape: (batch_size, hidden_dim)
    return cls_embeddings.cpu().numpy()

def normalize_scores(scores):
    scores = np.array(scores)
    min_val = scores.min()
    max_val = scores.max()
    if max_val == min_val:
        return np.zeros_like(scores)  # avoid division by zero
    return (scores - min_val) / (max_val - min_val)

def main():

    dataset = load_dataset("paoloitaliani/mami", dataset_subset=None)

    affinity_files = os.listdir(os.path.join(args.output_dir, args.run_name, "affinity_matrices"))
    all_affinity_data = []
    for affinity_file in affinity_files:
        affinity_path = os.path.join(args.output_dir, args.run_name, "affinity_matrices", affinity_file)
        with open(affinity_path, "rb") as f:
            all_affinity_data.append(pickle.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

    output_path = os.path.join(args.output_dir, args.run_name)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        cache_dir="../llms",
        trust_remote_code=True,
    )

    # Custom config hyperparameters
    config.num_gcn_layers = 1
    config.save_affinity = False
    config.output_dir = output_path 

    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

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
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_affinity_scores = []
    all_sim_scores = []
    all_label_pairs = []
    text_to_metadata = {}
    # Loop through all affinity data
    for entry in tqdm(all_affinity_data):
        affinity_matrix = entry["R_norm"].numpy()
        input_ids = entry["input_ids"]

        # Decode input_ids to texts
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

        # Lookup metadata
        for split in dataset:
            for example in dataset[split]:
                if example["text"] in texts:
                    text_to_metadata[example["text"]] = {
                        "file_name": example["file_name"],
                        "label": example["label"]
                    }       

        # Get embeddings and cosine similarity matrix
        embeddings = get_cls_embeddings(texts, tokenizer, model, device)
        # embeddings = model.encode(texts, convert_to_tensor=True).cpu()
        sim_matrix = cosine_similarity(embeddings)

        # Flatten entire matrices (include diagonal + lower triangle)
        all_affinity_scores.extend(affinity_matrix.flatten())
        all_sim_scores.extend(sim_matrix.flatten())

        
        # Now collect the label pairs for each (i, j)
        for i in range(len(texts)):
            for j in range(len(texts)):
                label_i = text_to_metadata.get(texts[i], {}).get("label", None)
                label_j = text_to_metadata.get(texts[j], {}).get("label", None)

                if (label_i, label_j) == (0, 0):
                    pair_label = "Both Non-Misogynistic"
                elif (label_i, label_j) == (1, 1):
                    pair_label = "Both Misogynistic"
                elif (label_i, label_j) == (0, 1):
                    pair_label = "Mixed Pair"
                elif (label_i, label_j) == (1, 0):
                    pair_label = "Mixed Pair"
                all_label_pairs.append(pair_label)

    
    # Convert to numpy arrays
    all_affinity_scores = normalize_scores(np.array(all_affinity_scores))
    all_sim_scores = normalize_scores(np.array(all_sim_scores))
    all_label_pairs = np.array(all_label_pairs)
    

    # Get top k% threshold
    threshold = np.percentile(all_affinity_scores, args.percentile)

    if args.fig_type == "top":
        # Filter by top k% affinity
        mask = all_affinity_scores >= threshold
    else:
        # Filter by bottom k% affinity
        mask = all_affinity_scores <= threshold

    # mask = all_affinity_scores <= threshold
    aff_topk = all_affinity_scores[mask]
    sim_topk = all_sim_scores[mask]
    label_pairs_topk = all_label_pairs[mask]

    label_pair_counts = Counter(label_pairs_topk)
    total_pairs = sum(label_pair_counts.values())

    # Convert to percentage
    label_pair_percentages = {
        label: round((count / total_pairs) * 100, 2)
        for label, count in label_pair_counts.items()
    }

    # Print nicely
    print("\nLabel Pair Distribution (Percentages):")
    for label, percent in label_pair_percentages.items():
        print(f"{label}: {percent}%")

    pearson_corr, pearson_p = pearsonr(aff_topk, sim_topk)
    spearman_corr, spearman_p = spearmanr(aff_topk, sim_topk)

    print("\n")
    print(f"Pearson correlation: {pearson_corr:.4f} (p = {pearson_p:.4e})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p = {spearman_p:.4e})")

    # Make a DataFrame from the scores
    df = pd.DataFrame({
        "Affinity Score": aff_topk,
        "Cosine Similarity": sim_topk,
        "Label Pair": label_pairs_topk
    })

    # Create scatter plot
    fig = px.scatter(
        df,
        x="Affinity Score",
        y="Cosine Similarity",
        title="Affinity Score vs Cosine Similarity",
        labels={"Affinity Score": "Affinity Matrix Value", "Cosine Similarity": "Text Similarity"},
        opacity=0.6,
        color="Label Pair", 
        trendline="ols"  # Adds a linear regression line
    )
    fig.update_traces(marker=dict(size=4))
    folder_path = os.path.join(args.output_dir, args.run_name, f"affinity_matrix_corr_{args.fig_type}.pdf")
    fig.write_image(folder_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output_mami")
    parser.add_argument("--run_name")
    parser.add_argument("--model_name_or_path", default="FacebookAI/xlm-roberta-large")
    parser.add_argument("--fig_type", help="top or bottom")
    parser.add_argument("--percentile", type=int, help="Percentile for top/bottom k%")

    args = parser.parse_args()
    main()