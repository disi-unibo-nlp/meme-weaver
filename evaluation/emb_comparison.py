import sys
sys.path.append('./')

import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


columns_dict = {"mami": ["shaming", "stereotype", "objectification", "violence"],}


def clean_image(image_path):
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(image_path, format="pdf")
    time.sleep(1)


def plot_all_embeddings(features, instance_ids, output_path, updated=False):

    # 1) prepare metadata
    ds = load_dataset(args.dataset, split="test")
    ds_df = ds.to_pandas()
    if args.id_column not in ds_df:
        ds_df = ds_df.reset_index().rename(columns={"index": args.id_column})
    mami = [args.target_column, "shaming", "stereotype", "objectification", "violence"]
    ds_df = ds_df[[args.id_column] + [c for c in mami if c in ds_df.columns]]
    # 1a) build merged base
    df = pd.DataFrame({args.id_column: instance_ids})
    merged = df.merge(ds_df, on=args.id_column, how="inner")
    conds = [
        merged[args.target_column] == 0,
        merged["shaming"] == 1,
        merged["stereotype"] == 1,
        merged["objectification"] == 1,
        merged["violence"] == 1,
    ]
    labels = ["non-misogynistic","shaming","stereotype","objectification","violence"]
    merged["type"] = np.select(conds, labels, default="other")
    merged["binary_type"] = merged[args.target_column].map({0:"non-misogynistic",1:"misogynistic"})

    # 2) embedding methods
    methods = {
        "tsne": lambda X,n: TSNE(n_components=n, perplexity=30, random_state=42).fit_transform(X),
        "pca":  lambda X,n: PCA(n_components=n, random_state=42).fit_transform(X),
    }

    for name, embed_fn in methods.items():
        for dim in (2,3):
            X = embed_fn(features, dim)
            # add coords
            for i, axis in enumerate(("x","y","z")[:dim]):
                merged[axis] = X[:, i]
            # prepare folder
            sub = os.path.join(output_path, name, f"{dim}d")
            os.makedirs(sub, exist_ok=True)

            # two plots: multiclass and binary
            for col, suffix in (("type",""), ("binary_type","_binary")):

                fname = f"{name}{'_upd' if updated else ''}_{args.batch_size}bs_{dim}d{suffix}.pdf"
                image_path = os.path.join(sub, fname)
                clean_image(image_path)

                fig = (px.scatter if dim==2 else px.scatter_3d)(
                    merged, x="x", y="y", **({"z":"z"} if dim==3 else {}),
                    color=col,
                    hover_data=[args.id_column],
                    title=f"{name.upper()} ({dim}D) colored by {col}",
                    width=800, height=600
                )
                # smaller, translucent markers
                fig.update_traces(marker=dict(size=6 if dim==2 else 2))
                
                fig.write_image(image_path)


def evaluate_cluster_separation(all_feats_stacked, all_instance_ids, output_path, updated=False):
    
    output_path = os.path.join(output_path, "separation")
    os.makedirs(output_path, exist_ok=True)

    image_path = os.path.join(output_path, f"tsne{'_upd' if updated else ''}_{args.batch_size}bs_2d_binary.pdf")
    clean_image(image_path)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X2 = tsne.fit_transform(all_feats_stacked)

    ds = load_dataset(args.dataset, split="test")
    ds_df = ds.to_pandas()
    if args.id_column not in ds_df:
        ds_df = ds_df.reset_index().rename(columns={"index": args.id_column})
    mami = [args.target_column, "shaming", "stereotype", "objectification", "violence"]
    ds_df = ds_df[[args.id_column] + [c for c in mami if c in ds_df.columns]]
    # 1a) build merged base
    df = pd.DataFrame({args.id_column: all_instance_ids})
    merged = df.merge(ds_df, on=args.id_column, how="inner")

    conds = [
        merged[args.target_column] == 0,
        merged["shaming"] == 1,
        merged["stereotype"] == 1,
        merged["objectification"] == 1,
        merged["violence"] == 1,
    ]
    labels = ["non-misogynistic","shaming","stereotype","objectification","violence"]
    merged["type"] = np.select(conds, labels, default="other")
    merged["binary_type"] = merged[args.target_column].map({0:"non-misogynistic",1:"misogynistic"})

    merged["x"], merged["y"] = X2[:,0], X2[:,1]
    coords = merged[["x","y"]].values
    binary_labels = (merged["binary_type"] == "misogynistic").astype(int).values

    clf = LogisticRegression(solver="liblinear", random_state=42)
    scores = cross_val_score(clf, coords, binary_labels, cv=5, scoring="accuracy")
    mean_acc, std_acc = scores.mean(), scores.std()
    print(f"features{'_upd' if updated else ''} 2D linear‐sep accuracy: "
        f"{mean_acc:.3f} ± {std_acc:.3f}")

    clf.fit(coords, binary_labels)
    

    # after clf.fit(...)
    xx = np.linspace(merged.x.min()-1, merged.x.max()+1, 200)
    yy = np.linspace(merged.y.min()-1, merged.y.max()+1, 200)
    xxg, yyg = np.meshgrid(xx, yy)
    Z = clf.predict(np.c_[xxg.ravel(), yyg.ravel()]).reshape(xxg.shape)

    # 4) create contour for decision region
    contour = go.Contour(
        x=xx, y=yy, z=Z,
        showscale=False,
        opacity=0.6,
        colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
        hoverinfo='skip',
        contours=dict(start=0, end=1, size=1)
    )

    # 5) build your scatter
    fig = px.scatter(
        merged, x="x", y="y",
        color="binary_type",
        hover_data=[args.id_column],
        width=800, height=600
    )
    # make the markers small in the plot but larger in the legend
    fig.update_traces(
        marker=dict(size=7),
        selector=dict(mode='markers')
    )

    # 6) add the contour *before* the scatter so points draw on top
    fig.add_trace(contour)

    # 7) layout tweaks:
    fig.update_layout(
        # remove axis titles
        xaxis_title='',
        yaxis_title='',
        legend_title_text='',
        
        font=dict(
            family="Arial, sans-serif",
            size=18,          # global font size
            color="black"
        ),

        # legend on top, horizontal, centered
        legend=dict(
            orientation='h',
            y=1.05,
            x=0.5,
            xanchor='center',
            yanchor='bottom',
            # force the legend symbols to reflect marker.size
            itemsizing='constant',
            # enlarge the font (optional)
            font=dict(size=20)
        ),

        # tighten margins so legend doesn’t get cut off
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    # 7) save or show
    fig.write_image(image_path)


def main():

    output_path = os.path.join(args.output_dir, args.run_name, "images", "embeddings")
    os.makedirs(output_path, exist_ok=True)

    affinity_files = os.listdir(os.path.join(args.output_dir, args.run_name, f"affinity_matrices_{args.batch_size}_bs"))

    all_affinity_data = []
    for affinity_file in affinity_files:
        affinity_path = os.path.join(args.output_dir, args.run_name, f"affinity_matrices_{args.batch_size}_bs", affinity_file)
        with open(affinity_path, "rb") as f:
            all_affinity_data.append(pickle.load(f))

    all_feats = []
    all_feats_upd = []
    all_instance_ids = []
    for entry in all_affinity_data:
        all_feats.append(entry["features"].numpy())
        all_feats_upd.append(entry["features_upd"].numpy())
        all_instance_ids.extend(entry["instance_ids"])

    all_feats_stacked = np.vstack(all_feats)
    all_feats_upd_stacked = np.vstack(all_feats_upd)
    # plot_all_embeddings(all_feats_stacked, all_instance_ids, output_path, updated=False)
    # plot_all_embeddings(all_feats_upd_stacked, all_instance_ids, output_path, updated=True)

    evaluate_cluster_separation(all_feats_stacked, all_instance_ids, output_path, updated=False)
    evaluate_cluster_separation(all_feats_upd_stacked, all_instance_ids, output_path, updated=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output_mami")
    parser.add_argument("--dataset", default="paoloitaliani/mami")
    parser.add_argument("--run_name")
    parser.add_argument("--id_column", default="file_name")
    parser.add_argument("--target_column", default="label")
    parser.add_argument("--batch_size", type=int)

    args = parser.parse_args()
    main()