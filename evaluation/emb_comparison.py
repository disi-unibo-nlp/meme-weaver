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
from sklearn.pipeline import Pipeline

columns_dict = {
    "mami": ["shaming", "stereotype", "objectification", "violence"],
}

def clean_image(image_path):
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(image_path, format="pdf")
    time.sleep(1)


def plot_all_embeddings(features, instance_ids, output_path, updated=False, use_pca_tsne=False):
    # Load metadata
    ds = load_dataset(args.dataset, split="test")
    ds_df = ds.to_pandas()
    if args.id_column not in ds_df:
        ds_df = ds_df.reset_index().rename(columns={"index": args.id_column})

    # Determine target and extra label cols
    target = args.target_column
    extra_labels = columns_dict.get(args.dataset, [])

    # Keep only id, target, and any extra label columns present
    keep_cols = [args.id_column, target] + [c for c in extra_labels if c in ds_df.columns]
    ds_df = ds_df[keep_cols]

    df_ids = pd.DataFrame({args.id_column: instance_ids})
    merged = df_ids.merge(ds_df, on=args.id_column, how="inner")

    # Build multiclass type if extra_labels exist
    if extra_labels:
        conds = [merged[target] == 0] + [merged[col] == 1 for col in extra_labels]
        labels = [f"non-{target}".replace('-', '_')] + extra_labels
        merged['type'] = np.select(conds, labels, default='other')
    else:
        merged['type'] = merged[target].map({0: f"Non-Misogynistic", 1: "Misogynistic"})

    merged['binary_type'] = merged[target].map({0: f"Non-Misogynistic", 1: "Misogynistic"})

    # Choose embedding
    if use_pca_tsne:
        methods = {'pca_tsne': lambda X, n: Pipeline([
            ('pca50', PCA(n_components=50, random_state=42)),
            ('tsne', TSNE(n_components=n, perplexity=30, random_state=42))
        ]).fit_transform(X)}
    else:
        methods = {'tsne': lambda X, n: TSNE(n_components=n, perplexity=30, random_state=42).fit_transform(X)}

    for name, embed_fn in methods.items():
        for dim in (2, 3):
            X = embed_fn(features, dim)
            for i, axis in enumerate(("x", "y", "z")[:dim]):
                merged[axis] = X[:, i]
            subdir = os.path.join(output_path, name, f"{dim}d")
            os.makedirs(subdir, exist_ok=True)

            # Multiclass plot
            fname = f"{name}{'_upd' if updated else ''}_{args.batch_size}bs_{dim}d.pdf"
            path = os.path.join(subdir, fname)
            clean_image(path)
            fig = (px.scatter if dim==2 else px.scatter_3d)(
                merged, x="x", y="y", **({"z": "z"} if dim==3 else {}),
                color="type" if extra_labels else None,
                hover_data=[args.id_column],
                title=f"{name.upper()} ({dim}D) colored by type",
                width=800, height=600
            )
            fig.update_traces(marker=dict(size=6 if dim==2 else 2))
            fig.write_image(path)

            # Binary plot
            fname_b = f"{name}{'_upd' if updated else ''}_{args.batch_size}bs_{dim}d_binary.pdf"
            path_b = os.path.join(subdir, fname_b)
            clean_image(path_b)
            fig_b = (px.scatter if dim==2 else px.scatter_3d)(
                merged, x="x", y="y", **({"z": "z"} if dim==3 else {}),
                color="binary_type",
                hover_data=[args.id_column],
                title=f"{name.upper()} ({dim}D) colored by binary_type",
                width=800, height=600
            )
            fig_b.update_traces(marker=dict(size=6 if dim==2 else 2))
            fig_b.write_image(path_b)


def evaluate_cluster_separation(all_feats_stacked, all_instance_ids, output_path,
                                updated=False, use_pca_tsne=False):
    out_sep = os.path.join(output_path, "separation")
    os.makedirs(out_sep, exist_ok=True)
    tag = 'pca_tsne' if use_pca_tsne else 'tsne'
    image_path = os.path.join(out_sep, f"{tag}{'_upd' if updated else ''}_{args.batch_size}bs_2d_binary.pdf")
    clean_image(image_path)

    if use_pca_tsne:
        pipe = Pipeline([
            ('pca50', PCA(n_components=50, random_state=42)),
            ('tsne2', TSNE(n_components=2, perplexity=30, random_state=42))
        ])
        X2 = pipe.fit_transform(all_feats_stacked)
    else:
        X2 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_feats_stacked)

    # Load and merge metadata
    ds = load_dataset(args.dataset, split="test")
    ds_df = ds.to_pandas()
    if args.id_column not in ds_df:
        ds_df = ds_df.reset_index().rename(columns={"index": args.id_column})
    target = args.target_column
    ds_df = ds_df[[args.id_column, target]]
    df_ids = pd.DataFrame({args.id_column: all_instance_ids})
    merged = df_ids.merge(ds_df, on=args.id_column, how="inner")

    merged['x'], merged['y'] = X2[:,0], X2[:,1]
    coords = merged[['x', 'y']].values
    binary_labels = (merged[target] == 1).astype(int).values

    clf = LogisticRegression(solver="liblinear", random_state=42)
    scores = cross_val_score(clf, coords, binary_labels, cv=5, scoring="accuracy")
    print(f"{tag}{'_upd' if updated else ''} 2D linear‐sep accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    clf.fit(coords, binary_labels)

    xx = np.linspace(merged.x.min()-1, merged.x.max()+1, 200)
    yy = np.linspace(merged.y.min()-1, merged.y.max()+1, 200)
    xxg, yyg = np.meshgrid(xx, yy)
    Z = clf.predict(np.c_[xxg.ravel(), yyg.ravel()]).reshape(xxg.shape)

    contour = go.Contour(
        x=xx, y=yy, z=Z,
        showscale=False, opacity=0.6,
        colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
        hoverinfo='skip', contours=dict(start=0, end=1, size=1)
    )

    label_map = {0: f"Non-Misogynistic", 1: "Misogynistic"} if "output_mami" in output_path else {0: "Non-Sexist", 1: "Sexist"}

    # 2) Map the numeric target to strings, then cast to Categorical with a fixed order:
    merged[args.target_column] = merged[target].map(label_map)
    fixed_order = list(label_map.values())  # e.g. ["Non-Misogynistic", "Misogynistic"]
    merged[args.target_column] = pd.Categorical(merged[args.target_column], categories=fixed_order, ordered=True)

    fig = px.scatter(
        merged, x="x", y="y", color=args.target_column,
        hover_data=[args.id_column], width=800, height=600, category_orders={args.target_column: fixed_order},
    )
    fig.update_traces(marker=dict(size=7), selector=dict(mode='markers'))
    fig.add_trace(contour)

    fig_model = "CLIP(MemeWeaver)" if "gcn" in output_path else "CLIP"
    fig_dataset = "MAMI" if "output_mami" in output_path else "EXIST"
    fig_title = f"{fig_model} {fig_dataset}"
    fig.update_layout(
        title=dict(
            text=fig_title,
            x=0.5,
            xanchor='center'
        ),
        font_family="Latin Modern Roman, serif",
        # template="plotly_white",
        xaxis_title="", yaxis_title='', legend_title_text='',
        font=dict(family="Arial, sans-serif", size=18, color="black"),
        #legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center', yanchor='bottom', itemsizing='constant', font=dict(size=20)),
        showlegend=False,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    fig.write_image(image_path)


def main():
    output_path = os.path.join(args.output_dir, args.run_name, "images", "embeddings")
    os.makedirs(output_path, exist_ok=True)

    affinity_dir = os.path.join(args.output_dir, args.run_name, f"affinity_matrices_{args.batch_size}_bs")
    affinity_files = os.listdir(affinity_dir)
    all_affinity_data = [pickle.load(open(os.path.join(affinity_dir, f), 'rb')) for f in affinity_files]

    feats = np.vstack([e["features"].numpy() for e in all_affinity_data])
    feats_upd = np.vstack([e["features_upd"].numpy() for e in all_affinity_data])
    all_ids = [i for e in all_affinity_data for i in e["instance_ids"]]

    plot_all_embeddings(feats, all_ids, output_path, updated=False, use_pca_tsne=False)
    evaluate_cluster_separation(feats, all_ids, output_path, updated=False, use_pca_tsne=False)

    # PCA->TSNE
    plot_all_embeddings(feats_upd, all_ids, output_path, updated=True, use_pca_tsne=True)
    evaluate_cluster_separation(feats_upd, all_ids, output_path, updated=True, use_pca_tsne=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output_mami")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_name")
    parser.add_argument("--id_column", default="file_name")
    parser.add_argument("--target_column", default="label")
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    main()
