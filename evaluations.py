import os
import json
import pandas as pd
import matplotlib.pyplot as plt

results_dir = "results"
models = ["Mamba", "SSM", "Transformer", "LSTM"]
colors = {
    "Mamba": "orange",
    "SSM": "blue",
    "Transformer": "green",
    "LSTM": "red"
}

runtime_data = []
summary_data = []

for model in models:
    csv_path = os.path.join(results_dir, f"{model.lower()}_inference_log.csv")
    json_path = os.path.join(results_dir, f"{model.lower()}_summary.json")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["model"] = model
        runtime_data.append(df)

    if os.path.exists(json_path):
        with open(json_path) as f:
            summary = json.load(f)
            summary_data.append(summary)

# Runtime vs. Sequence Length Plot
if runtime_data:
    runtime_df = pd.concat(runtime_data, ignore_index=True)
    plt.figure(figsize=(10, 6))
    for model in models:
        sub_df = runtime_df[runtime_df["model"] == model]
        grouped = sub_df.groupby("seq_len")["runtime_s"].mean().reset_index()
        plt.plot(grouped["seq_len"], grouped["runtime_s"], marker='o', label=model, color=colors[model])

    plt.title("Inference Runtime vs. Sequence Length")
    plt.xlabel("Sequence Length (non-padding tokens)")
    plt.ylabel("Average Runtime (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "runtime_vs_seq_length.png"), dpi=300)
    plt.close()

# Accuracy and Inference Time Summary Plots 
if summary_data:
    summary_df = pd.DataFrame(summary_data)

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["model"], summary_df["accuracy"], color=[colors[m] for m in summary_df["model"]])
    plt.title("Average Accuracy per Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()

    # Inference time plot
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["model"], summary_df["avg_inference_time"], color=[colors[m] for m in summary_df["model"]])
    plt.title("Average Inference Time per Model")
    plt.ylabel("Time (s)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "inference_time_comparison.png"), dpi=300)
    plt.close()