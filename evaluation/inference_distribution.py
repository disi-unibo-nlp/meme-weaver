import os
import time
import json
import argparse
import pandas as pd
import plotly.express as px


def main():

    output_path = os.path.join(args.output_dir, args.run_name, "inferences")

    inferences_jsons = os.listdir(output_path)

    metric_values = []
    batch_sizes = []
    for inf_json in inferences_jsons:
        
        inf_json_path = os.path.join(output_path, inf_json)
        with open(inf_json_path, "r") as f:
            metrics = json.load(f)
            batch_sizes.append(int(inf_json.split(".")[0].split("batch")[1]))
        
        metric_values.append(metrics[args.metric])
    
    df = pd.DataFrame({
        'batch_size': batch_sizes,
        'metric_value': metric_values
    })
    
    
    max_row = df.loc[df['metric_value'].idxmax()]
    max_value = max_row['metric_value']
    corresponding_batch_size = max_row['batch_size']

    print(f"Max {args.metric} value: {max_value}")
    print(f"Corresponding batch size: {corresponding_batch_size}")

    folder_path = os.path.join(args.output_dir, args.run_name, f"{args.metric}_batch_distribution.pdf")
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(folder_path, format="pdf")
    time.sleep(1)

    df = df.sort_values('batch_size')

    # Calculate a rolling average (adjust window size as needed)
    df['rolling_metric'] = df['metric_value'].rolling(window=20, min_periods=1).mean()

    # Create a line chart for the rolling average
    fig = px.line(df, x='batch_size', y='rolling_metric',
                title=f'Smoothed {args.metric} Value vs Batch Size (Rolling Average)',
                labels={'batch_size': 'Batch Size', 'rolling_metric': f'Smoothed {args.metric} Value'})

    # Update the trace to use a spline for an extra smooth appearance
    fig.update_traces(line_shape='spline', marker=dict(size=6))


    # Adjust y-axis range to zoom into the small differences
    fig.update_yaxes(range=[min(metric_values), max(metric_values)])
    fig.write_image(folder_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output_mami")
    parser.add_argument("--run_name")
    parser.add_argument("--metric", default="predict_accuracy")

    args = parser.parse_args()
    main()