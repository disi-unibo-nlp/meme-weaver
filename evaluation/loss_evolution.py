import sys 
sys.path.append('./')


import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from evaluation.emb_comparison import clean_image


output_path = "loss_evolution.pdf"
clean_image(output_path)

# 1) Load CSVs (adjust paths as needed)
bases = {
    ("EXIST", "CLIP"):        "output_hard_label_task4/clip-vit-large_batch64_5e-6lr_mami20bs_captionPA_concat",
    ("EXIST", "MemeWeaver"):  "output_hard_label_task4/clip-vit-large_batch64_5e-6lr_1gcn_xuinit_mami20bs_captionPA_concat",
    ("MAMI",  "CLIP"):        "output_mami/clip-vit-large_batch20_5e-6lr_mfb",
    ("MAMI",  "MemeWeaver"):  "output_mami/clip-vit-large_batch20_5e-6lr_1gcn_xuinit_mfb",
}

dfs = []
for (dataset, tech), base in bases.items():
    df = pd.read_csv(os.path.join(base, "training_loss.csv"))
    df["dataset"]   = dataset
    df["technique"] = tech
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# 2) Compute normalized step per run
#    (so each run’s global_step goes from 0.0 → 1.0)
df["step_norm"] = (
    df
    .groupby(["dataset", "technique"])["train/global_step"]
    .transform(lambda x: x / x.max())
)

# 1) Base line‐plot, but hide its legend
fig = px.line(
    df,
    x="step_norm", y="train/loss",
    color="dataset",
    line_dash="technique",
    color_discrete_map={"MAMI":"blue","EXIST":"red"},
    line_dash_map={"CLIP":"dot","MemeWeaver":"solid"}
)
fig.update_layout(showlegend=False)

# 2) Define your custom‐legend items and positions (in paper‐coords)
#    x,y are fractional (0→1) positions on the figure
legend_defs = [
    {"label":"MAMI",        "color":"blue", "dash":"solid", "x":0.35, "y":1.10},
    {"label":"EXIST",       "color":"red",  "dash":"solid", "x":0.65, "y":1.10},
    {"label":"CLIP",        "color":"black","dash":"dot",   "x":0.35, "y":1.05},
    {"label":"MemeWeaver",  "color":"black","dash":"solid", "x":0.65, "y":1.05},
]

for item in legend_defs:
    fig.add_shape(
        type="line",
        xref="paper", x0=item["x"]-0.05, x1=item["x"]-0.01,
        yref="paper", y0=item["y"],    y1=item["y"],
        line=dict(color=item["color"], dash=item["dash"], width=4)
    )
    fig.add_annotation(
        xref="paper", x=item["x"]+0.02,
        yref="paper", y=item["y"],
        text=item["label"],
        showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(size=16)
    )

# 3) Tidy up layout
fig.update_layout(
    title_x=0.5,
    xaxis_title="Normalized Global Step",
    yaxis_title="Training Loss",
)
fig.write_image(output_path)

