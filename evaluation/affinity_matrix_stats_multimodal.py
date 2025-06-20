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


def compute_distribution(elements):
    label_pair_counts = Counter(elements)
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
        

def create_plot(df, sim_measure, sim_type, output_path, color_map, color):

    # Create scatter plot
    fig = px.scatter(
        df,
        x="Affinity Score",
        y=sim_measure,
        title=f"Affinity Score vs {sim_measure}",
        opacity=0.4,
        color_discrete_map=color_map,
        color=color, 
    )
    fig.update_traces(marker=dict(size=4))
    folder_path = os.path.join(output_path, f"affinity_matrix_corr_{sim_type}_{args.fig_type}_{args.batch_size}_bs.pdf")
    fig.write_image(folder_path)


def get_label_pair(labels, preds):
    # Now collect the label pairs for each (i, j)
    misogniny_pairs = []
    pred_pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            label_i = labels[i]
            label_j = labels[j]

            if (label_i, label_j) == (0, 0):
                pair_label = "Both Non-Misogynistic"
            elif (label_i, label_j) == (1, 1):
                pair_label = "Both Misogynistic"
            elif (label_i, label_j) == (0, 1):
                pair_label = "Mixed Pair"
            elif (label_i, label_j) == (1, 0):
                pair_label = "Mixed Pair"
            misogniny_pairs.append(pair_label)

            pred_i = preds[i]
            pred_j = preds[j]

            if pred_i == label_i:
                outcome_i = "correct"
            if pred_i != label_i:
                outcome_i = "incorrect"
            if pred_j == label_j:
                outcome_j = "correct"
            if pred_j != label_j:
                outcome_j = "incorrect"
            
            if (outcome_i, outcome_j) == ("correct", "correct"):
                outcome_label = "Both Correct"
            elif (outcome_i, outcome_j) == ("incorrect", "incorrect"):
                outcome_label = "Both Incorrect"
            elif (outcome_i, outcome_j) == ("correct", "incorrect"):
                outcome_label = "Mixed Outcome"
            elif (outcome_i, outcome_j) == ("incorrect", "correct"):
                outcome_label = "Mixed Outcome"
            
            pred_pairs.append(outcome_label)


            
    return misogniny_pairs, pred_pairs


def get_color_map(df, color):
    unique_pairs = df[color].unique()
    # pick whichever palette you like; here we use Plotly’s built-in qualitative palette
    palette = px.colors.qualitative.Plotly  
    # map each Pred Pair -> one colour (cycling if you have more pairs than colours)
    color_map = {
        pair: palette[i % len(palette)]
        for i, pair in enumerate(unique_pairs)
    }
    return color_map
    


def main():


    affinity_files = os.listdir(os.path.join(args.output_dir, args.run_name, f"affinity_matrices_{args.batch_size}_bs"))
    all_affinity_data = []
    for affinity_file in affinity_files:
        affinity_path = os.path.join(args.output_dir, args.run_name, f"affinity_matrices_{args.batch_size}_bs", affinity_file)
        with open(affinity_path, "rb") as f:
            all_affinity_data.append(pickle.load(f))
    

    output_path = os.path.join(args.output_dir, args.run_name, "images")
    os.makedirs(output_path, exist_ok=True)

    
    all_affinity_scores = []
    all_sim_features_scores = []
    all_sim_features_upd_scores = []
    all_sim_image_scores = []
    all_sim_text_scores = []
    all_misoginy_pairs = []
    all_pred_pairs = []
    # Loop through all affinity data
    for entry in tqdm(all_affinity_data):
        affinity_matrix = entry["R_norm"].numpy()

        features = entry["features"]
        features_upd = entry["features_upd"]
        image_embeds = entry["image_embeds"]
        text_embeds = entry["text_embeds"]
        labels = entry["labels"]
        logits = entry["logits"]
        preds = np.argmax(logits, axis=1)

        sim_matrix_features_upd = cosine_similarity(features_upd)
        sim_matrix_features = cosine_similarity(features)
        sim_matrix_image = cosine_similarity(image_embeds)
        sim_matrix_text = cosine_similarity(text_embeds)

        misoginy_pairs, pred_pairs = get_label_pair(labels, preds)
        
        all_misoginy_pairs.extend(misoginy_pairs)
        all_pred_pairs.extend(pred_pairs)


        # Flatten entire matrices (include diagonal + lower triangle)
        all_affinity_scores.extend(affinity_matrix.flatten())
        all_sim_features_scores.extend(sim_matrix_features.flatten())
        all_sim_features_upd_scores.extend(sim_matrix_features_upd.flatten())
        all_sim_image_scores.extend(sim_matrix_image.flatten())
        all_sim_text_scores.extend(sim_matrix_text.flatten())

    # Convert to numpy arrays
    all_affinity_scores = np.array(all_affinity_scores)
    all_sim_features_scores = np.array(all_sim_features_scores)
    all_sim_features_upd_scores = np.array(all_sim_features_upd_scores)
    all_sim_image_scores = np.array(all_sim_image_scores)
    all_sim_text_scores = np.array(all_sim_text_scores)
    all_misoginy_pairs = np.array(all_misoginy_pairs)
    all_pred_pairs = np.array(all_pred_pairs)
    
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
    sim_features_topk = all_sim_features_scores[mask]
    sim_features_upd_topk = all_sim_features_upd_scores[mask]
    sim_image_topk = all_sim_image_scores[mask]
    sim_text_topk = all_sim_text_scores[mask]
    misogniny_pairs_topk = all_misoginy_pairs[mask]
    pred_pairs_topk = all_pred_pairs[mask]

    pearson_corr, pearson_p = pearsonr(aff_topk, sim_features_topk)
    spearman_corr, spearman_p = spearmanr(aff_topk, sim_features_topk)

    print("\n")
    print(f"Pearson correlation: {pearson_corr:.4f} (p = {pearson_p:.4e})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p = {spearman_p:.4e})")

    # Make a DataFrame from the scores
    df = pd.DataFrame({
        "Affinity Score": aff_topk,
        "Fused Features Similarity": sim_features_topk,
        "Fused Features Updated Similarity": sim_features_upd_topk,
        "Image Similarity": sim_image_topk,
        "Text Similarity": sim_text_topk,
        "Misogniny Pair": misogniny_pairs_topk,
        "Pred Pair": pred_pairs_topk
    })

    # df = df[df["Misogniny Pair"] == "Both Misogynistic"]
    # df = df[df["Misogniny Pair"] == "Both Non-Misogynistic"]
    # df = df[df["Misogniny Pair"] == "Mixed Pair"]

    df_mixed_mis = df[df["Misogniny Pair"] == "Mixed Pair"]
    df_both_mis = df[df["Misogniny Pair"] == "Both Misogynistic"]
    df_both_non_mis = df[df["Misogniny Pair"] == "Both Non-Misogynistic"]

    df_both_correct = df[df["Pred Pair"] == "Both Correct"]
    df_both_incorrect = df[df["Pred Pair"] == "Both Incorrect"]
    df_mixed_outcome = df[df["Pred Pair"] == "Mixed Outcome"]


    color_mis_pair = "Misogniny Pair"
    color_map_mis = get_color_map(df, color_mis_pair)

    color_pred_pair = "Pred Pair"
    color_map_pred = get_color_map(df, color_pred_pair)

    
    output_path_features = os.path.join(output_path, "correlation", "fused_features")
    os.makedirs(output_path_features, exist_ok=True)

    create_plot(df, "Fused Features Similarity", "features", output_path_features, color_map_mis, color_pred_pair)
    create_plot(df_mixed_mis, "Fused Features Similarity", "features_mix_mis", output_path_features, color_map_pred, color_pred_pair)
    create_plot(df_both_mis, "Fused Features Similarity", "features_both_mis", output_path_features, color_map_pred, color_pred_pair)
    create_plot(df_both_non_mis, "Fused Features Similarity", "features_both_non_mis", output_path_features, color_map_pred, color_pred_pair)
    create_plot(df_both_correct, "Fused Features Similarity", "features_both_correct", output_path_features, color_map_mis, color_mis_pair)
    create_plot(df_both_incorrect, "Fused Features Similarity", "features_both_incorrect", output_path_features, color_map_mis, color_mis_pair)
    create_plot(df_mixed_outcome, "Fused Features Similarity", "features_mixed_outcome", output_path_features, color_map_mis, color_mis_pair)

    output_path_features_upd = os.path.join(output_path, "correlation", "fused_features_upd")
    os.makedirs(output_path_features_upd, exist_ok=True)
    create_plot(df, "Fused Features Updated Similarity", "features", output_path_features_upd, color_map_mis, color_pred_pair)
    create_plot(df_mixed_mis, "Fused Features Updated Similarity", "features_mix_mis", output_path_features_upd, color_map_pred, color_pred_pair)
    create_plot(df_both_mis, "Fused Features Updated Similarity", "features_both_mis", output_path_features_upd, color_map_pred, color_pred_pair)
    create_plot(df_both_non_mis, "Fused Features Updated Similarity", "features_both_non_mis", output_path_features_upd, color_map_pred, color_pred_pair)
    create_plot(df_both_correct, "Fused Features Updated Similarity", "features_both_correct", output_path_features_upd, color_map_mis, color_mis_pair)
    create_plot(df_both_incorrect, "Fused Features Updated Similarity", "features_both_incorrect", output_path_features_upd, color_map_mis, color_mis_pair)
    create_plot(df_mixed_outcome, "Fused Features Updated Similarity", "features_mixed_outcome", output_path_features_upd, color_map_mis, color_mis_pair)

    # create_plot(df, "Image Similarity", "image", output_path)
    # create_plot(df, "Text Similarity", "text", output_path)

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output_mami")
    parser.add_argument("--run_name")
    parser.add_argument("--fig_type", help="top or bottom")
    parser.add_argument("--percentile", type=int, help="Percentile for top/bottom k%")
    parser.add_argument("--batch_size", type=int, help="Batch size for evaluation")

    args = parser.parse_args()
    main()