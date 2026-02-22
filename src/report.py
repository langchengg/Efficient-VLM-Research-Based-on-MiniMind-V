"""
Result aggregation: tables, plots, Markdown report.
"""

import os
import json
import logging
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def to_dataframe(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Method": r.get("method"),
                "Perplexity_down": r.get("perplexity"),
                "Tok_s_up": r.get("tokens_per_sec"),
                "Latency_s_down": r.get("latency_s"),
                "GPU_Mem_MB_down": r.get("gpu_mem_mb"),
                "Peak_GPU_MB": r.get("peak_gpu_mb"),
            }
        )
    df = pd.DataFrame(rows)
    df.columns = ["Method", "Perplexity↓", "Tok/s↑", "Latency(s)↓", "GPU Mem(MB)↓", "Peak GPU(MB)"]
    return df


def print_table(df: pd.DataFrame):
    from tabulate import tabulate

    fmt = df.copy()
    for c in fmt.select_dtypes("number").columns:
        fmt[c] = fmt[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    print()
    print(tabulate(fmt, headers="keys", tablefmt="github", showindex=False))
    print()


def relative_table(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics relative to the first row (FP16 baseline)."""
    base = df.iloc[0]
    rel = df[["Method"]].copy()
    if pd.notna(base["Perplexity↓"]):
        rel["PPL Delta%"] = (
            (df["Perplexity↓"] - base["Perplexity↓"]) / base["Perplexity↓"] * 100
        ).round(2)
    if pd.notna(base["Tok/s↑"]):
        rel["Speed x"] = (df["Tok/s↑"] / base["Tok/s↑"]).round(2)
    if pd.notna(base["GPU Mem(MB)↓"]):
        rel["Mem Save%"] = (
            (1 - df["GPU Mem(MB)↓"] / base["GPU Mem(MB)↓"]) * 100
        ).round(1)
    return rel


def plot_results(df: pd.DataFrame, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    methods = df["Method"].tolist()
    colors = plt.cm.Set2.colors

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("LLM Quantization Benchmark", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    vals = df["Perplexity↓"].dropna()
    if len(vals):
        ax.bar(vals.index, vals, color=[colors[i % len(colors)] for i in vals.index])
        ax.set_xticks(vals.index)
        ax.set_xticklabels([methods[i] for i in vals.index], rotation=30, ha="right")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity (lower = better)")
        for i, v in zip(vals.index, vals):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)

    ax = axes[0, 1]
    vals = df["Tok/s↑"].dropna()
    if len(vals):
        ax.bar(vals.index, vals, color=[colors[i % len(colors)] for i in vals.index])
        ax.set_xticks(vals.index)
        ax.set_xticklabels([methods[i] for i in vals.index], rotation=30, ha="right")
        ax.set_ylabel("Tokens / sec")
        ax.set_title("Inference Speed (higher = better)")
        for i, v in zip(vals.index, vals):
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)

    ax = axes[1, 0]
    vals = df["GPU Mem(MB)↓"].dropna()
    if len(vals):
        ax.bar(vals.index, vals, color=[colors[i % len(colors)] for i in vals.index])
        ax.set_xticks(vals.index)
        ax.set_xticklabels([methods[i] for i in vals.index], rotation=30, ha="right")
        ax.set_ylabel("MB")
        ax.set_title("GPU Memory (lower = better)")
        for i, v in zip(vals.index, vals):
            ax.text(i, v + 10, f"{v:.0f}", ha="center", fontsize=8)

    ax = axes[1, 1]
    valid = df.dropna(subset=["Perplexity↓", "GPU Mem(MB)↓"])
    if len(valid):
        for idx, row in valid.iterrows():
            ax.scatter(row["GPU Mem(MB)↓"], row["Perplexity↓"],
                       color=colors[idx % len(colors)], s=90, zorder=5)
            ax.annotate(row["Method"], (row["GPU Mem(MB)↓"], row["Perplexity↓"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("GPU Memory (MB)")
        ax.set_ylabel("Perplexity")
        ax.set_title("Quality vs. Efficiency Trade-off")

    plt.tight_layout()
    path = os.path.join(save_dir, "benchmark_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Plot -> {path}")


def save_results(results: list, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)

    clean = [
        {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool, type(None)))}
        for r in results
    ]
    json_path = os.path.join(save_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)

    df = to_dataframe(results)
    csv_path = os.path.join(save_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved -> {json_path}, {csv_path}")
    return df


def generate_full_report(results: list, save_dir: str = "results"):
    """Print tables, plot charts, save files."""
    df = save_results(results, save_dir)

    print("\n" + "=" * 70)
    print("  QUANTIZATION BENCHMARK -- ABSOLUTE RESULTS")
    print("=" * 70)
    print_table(df)

    rel = relative_table(df)
    print("  RELATIVE TO FP16 BASELINE")
    print("-" * 70)
    print_table(rel)

    plot_results(df, save_dir)
    return df
