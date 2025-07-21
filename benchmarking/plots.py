# plot_benchmarks.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_scenario(df, scenario_name, x_axis, title, filename):
    """Helper function to generate and save a plot for a given scenario."""

    # Filter the DataFrame for the specific scenario
    scenario_df = df[df["scenario"] == scenario_name].copy()

    # Create a new figure and axes for the plot
    plt.figure(figsize=(12, 8))

    # Use seaborn for a clean, publication-quality line plot
    # It automatically groups by 'algorithm' and calculates mean/confidence intervals
    sns.lineplot(
        data=scenario_df,
        x=x_axis,
        y="runtime",
        hue="algorithm",
        marker="o",
        linewidth=2.5,
        ci="sd",  # Show standard deviation as a shaded area
    )

    # Set plot properties
    plt.title(title, fontsize=18, fontweight="bold", pad=20)
    plt.xlabel(x_axis.replace("_", " ").title(), fontsize=14)
    plt.ylabel("Runtime (seconds, log scale)", fontsize=14)
    plt.yscale("log")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", ls="--", c="0.7")
    plt.legend(title="Algorithm", fontsize=11, title_fontsize=13)
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to '{filename}'")


def main():
    """Main function to load results and generate plots."""
    try:
        df = pd.read_csv("benchmark_results.csv")
    except FileNotFoundError:
        print("Error: 'benchmark_results.csv' not found.")
        print("Please run 'run_benchmarks.py' first.")
        return

    # Use a professional plot style
    sns.set_theme(style="whitegrid")

    # --- Plot for p >> n scenario ---
    plot_scenario(
        df=df,
        scenario_name="p >> n",
        x_axis="n_features",
        title="Benchmark: Runtime vs. Number of Features (p >> n)\n(n_samples fixed at 500)",
        filename="benchmark_p_dominant.png",
    )

    # --- Plot for n >> p scenario ---
    plot_scenario(
        df=df,
        scenario_name="n >> p",
        x_axis="n_samples",
        title="Benchmark: Runtime vs. Number of Samples (n >> p)\n(n_features fixed at 100)",
        filename="benchmark_n_dominant.png",
    )


if __name__ == "__main__":
    main()
