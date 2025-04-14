import os
import time
import json
import argparse
import pandas as pd
import plotly.express as px

def print_max_stat(df, metric_name, output_path):
        """
        Computes the maximum value of a given metric and the corresponding batch size.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            metric_name (str): Column name of the metric.
        """
        max_row = df.loc[df[metric_name].idxmax()]
        max_value = max_row[metric_name]
        batch_size_at_max = int(max_row['batch_size'])
        print(f"Max {metric_name} value: {max_value}")
        print(f"Corresponding batch size: {batch_size_at_max}")

        if metric_name == "acc_value":
            try:
                old_filename = os.path.join(output_path, f"test_results_batch{batch_size_at_max}.json")
                new_filename = os.path.join(output_path, f"test_results_batch{batch_size_at_max}_best1.json")
                os.rename(old_filename, new_filename)
            except:
                pass
                
                            
def main():

    output_path = os.path.join(args.output_dir, args.run_name, "inferences")

    inferences_jsons = os.listdir(output_path)

    f1_values = []
    acc_values = []
    auc_values = []
    batch_sizes = []
    for inf_json in inferences_jsons:
        
        inf_json_path = os.path.join(output_path, inf_json)
        with open(inf_json_path, "r") as f:
            metrics = json.load(f)
            batch_str = inf_json.split(".")[0].split("batch")[1]
            if "best1" in batch_str:
                batch_str = batch_str.replace("_best1", "")
            batch_sizes.append(int(batch_str))
        
        f1_values.append(metrics["predict_F1_macro"])
        acc_values.append(metrics["predict_accuracy"])
        auc_values.append(metrics["predict_roc_auc"])
    
    df = pd.DataFrame({
        'batch_size': batch_sizes,   # List/array of batch sizes
        'f1_value': f1_values,       # F1 metric values
        'acc_value': acc_values,     # Accuracy metric values
        'auc_value': auc_values      # AUC metric values
    })

    # --- Compute and print statistics for each metric ---
    for metric in ['f1_value', 'acc_value', 'auc_value']:
        print_max_stat(df, metric, output_path)
        print('-' * 40)

    # --- Sort DataFrame by batch size ---
    df = df.sort_values('batch_size')

    # --- Calculate a rolling average (smoothed curve) for each metric ---
    window_size = 10  # You can adjust the window size as needed
    for metric in ['f1_value', 'acc_value', 'auc_value']:
        df[f'rolling_{metric}'] = df[metric].rolling(window=window_size, min_periods=1).mean()

    # --- Plot all three metrics ---
    # Prepare list of rolling metric columns to be plotted
    rolling_metrics = [f'rolling_f1_value', f'rolling_acc_value', f'rolling_auc_value']

    fig = px.line(
        df,
        x='batch_size',
        y=rolling_metrics,
        title='Smoothed Metrics vs. Batch Size (Rolling Average)',
        labels={
            'batch_size': 'Batch Size',
            'value': 'Smoothed Metric Value',
            'variable': 'Metric'
        }
    )

    # Update the plot to use spline curves and adjust the marker size
    # fig.update_traces(line_shape='spline', marker=dict(size=6))

    # --- Save the plot to a PDF file ---
    output_file = os.path.join(args.output_dir, args.run_name, "metrics_batch_distribution.pdf")
    fig.write_image(output_file, format="pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output_mami")
    parser.add_argument("--run_name")

    args = parser.parse_args()
    main()