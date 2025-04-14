import sys
sys.path.append('./')

import os
import torch
import pickle
import argparse
import pandas as pd
import plotly.express as px
import torch.nn.functional as F
from sklearn.manifold import MDS



def main():
    affinity_files = os.listdir(os.path.join(args.output_dir, args.run_name, f"affinity_matrices_{args.batch_size}_bs"))

    all_affinity_data = []
    for affinity_file in affinity_files:
        affinity_path = os.path.join(args.output_dir, args.run_name, f"affinity_matrices_{args.batch_size}_bs", affinity_file)
        with open(affinity_path, "rb") as f:
            all_affinity_data.append(pickle.load(f))

    R_norm = all_affinity_data[0]['R_norm']

    R_softmax = F.softmax(R_norm, dim=-1)
    R_softmax_sym = (R_softmax + R_softmax.T) / 2

    D = 1.0 - R_softmax_sym

    D_numpy = D.cpu().numpy()

    # Run MDS to get 2D coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    points_2d = mds.fit_transform(D_numpy)


    df = pd.DataFrame(points_2d, columns=["x", "y"])
    df["instance"] = df.index.astype(str)  # optional labels

    # Create Plotly scatter plot
    fig = px.scatter(
        df, x="x", y="y",
        text="instance",
        title="2D Embedding of Instances from Affinity Matrix (via MDS)",
        labels={"x": "MDS Dimension 1", "y": "MDS Dimension 2"},
        width=800, height=600
    )

    # Optional: Show point labels
    fig.update_traces(marker=dict(size=10), textposition='top center')

    fig.write_image("test.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output_mami")
    parser.add_argument("--run_name")
    parser.add_argument("--batch_size", type=int, default=100)

    args = parser.parse_args()
    main()