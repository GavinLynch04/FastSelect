import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_single_plot(df, scenario, x_col, y_col, y_label, title, output_path, log_scale=True):
    """Generates and saves a high-resolution single metric benchmark line plot."""
    scenario_df = df[df["scenario"] == scenario].copy()
    if scenario_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Distinct palette for fast-select CPU, GPU, and scikit-rebate baselines
    unique_algos = sorted(scenario_df["algorithm"].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_algos))

    sns.lineplot(
        data=scenario_df,
        x=x_col,
        y=y_col,
        hue="algorithm",
        style="algorithm",
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8,
        palette=palette,
        ax=ax,
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")

    if log_scale and y_col == "runtime":
        ax.set_yscale("log")

    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(title="Algorithm Backend", fontsize=10, title_fontsize=11, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated plot: '{output_path}'")


def generate_combined_plot(df, scenario, x_col, title, output_path):
    """Generates a 2-panel figure showing Runtime (log scale) and Memory Overhead side-by-side."""
    scenario_df = df[df["scenario"] == scenario].copy()
    if scenario_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    unique_algos = sorted(scenario_df["algorithm"].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_algos))

    # Panel 1: Runtime
    sns.lineplot(
        data=scenario_df,
        x=x_col,
        y="runtime",
        hue="algorithm",
        style="algorithm",
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8,
        palette=palette,
        ax=ax1,
    )
    ax1.set_title("Runtime (Seconds)", fontsize=13, fontweight="bold")
    ax1.set_xlabel(x_col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax1.set_ylabel("Execution Time (Seconds, Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.7)
    ax1.legend().remove()

    # Panel 2: Memory
    sns.lineplot(
        data=scenario_df,
        x=x_col,
        y="peak_ram_mb",
        hue="algorithm",
        style="algorithm",
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8,
        palette=palette,
        ax=ax2,
    )
    ax2.set_title("Peak RAM Overhead (MB)", fontsize=13, fontweight="bold")
    ax2.set_xlabel(x_col.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax2.set_ylabel("Peak RAM Overhead (MB)", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.7)
    ax2.legend(title="Algorithm Backend", fontsize=10, title_fontsize=11, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated combined plot: '{output_path}'")


def main():
    csv_file = "benchmarking/benchmark_results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found. Run benchmarking/benchmarking.py first.")
        return

    df = pd.read_csv(csv_file)

    # 1. Runtime vs Samples (n >> p)
    generate_single_plot(
        df,
        scenario="n >> p",
        x_col="n_samples",
        y_col="runtime",
        y_label="Execution Time (Seconds, Log Scale)",
        title="Fast-Select Benchmark: Runtime vs. Number of Samples (n >> p)\n(n_features fixed at 100)",
        output_path="benchmarking/benchmark_n_dominant_runtime.png",
    )

    # 2. Runtime vs Features (p >> n)
    generate_single_plot(
        df,
        scenario="p >> n",
        x_col="n_features",
        y_col="runtime",
        y_label="Execution Time (Seconds, Log Scale)",
        title="Fast-Select Benchmark: Runtime vs. Number of Features (p >> n)\n(n_samples fixed at 100)",
        output_path="benchmarking/benchmark_p_dominant_runtime.png",
    )

    # 3. Peak RAM vs Samples (n >> p)
    generate_single_plot(
        df,
        scenario="n >> p",
        x_col="n_samples",
        y_col="peak_ram_mb",
        y_label="Peak RAM Overhead (MB)",
        title="Fast-Select Benchmark: Peak RAM vs. Number of Samples (n >> p)\n(n_features fixed at 100)",
        output_path="benchmarking/benchmark_n_dominant_memory.png",
        log_scale=False,
    )

    # 4. Peak RAM vs Features (p >> n)
    generate_single_plot(
        df,
        scenario="p >> n",
        x_col="n_features",
        y_col="peak_ram_mb",
        y_label="Peak RAM Overhead (MB)",
        title="Fast-Select Benchmark: Peak RAM vs. Number of Features (p >> n)\n(n_samples fixed at 100)",
        output_path="benchmarking/benchmark_p_dominant_memory.png",
        log_scale=False,
    )

    # 5. Combined N-Dominant Figure
    generate_combined_plot(
        df,
        scenario="n >> p",
        x_col="n_samples",
        title="Fast-Select Performance Scaling: N-Dominant Scenario (n >> p)",
        output_path="benchmarking/benchmark_n_dominant.png",
    )

    # 6. Combined P-Dominant Figure
    generate_combined_plot(
        df,
        scenario="p >> n",
        x_col="n_features",
        title="Fast-Select Performance Scaling: P-Dominant Scenario (p >> n)",
        output_path="benchmarking/benchmark_p_dominant.png",
    )


if __name__ == "__main__":
    main()
