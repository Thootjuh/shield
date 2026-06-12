import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import glob
import os
import sys
import seaborn as sns


# CONFIGURATION
sns.set_theme(
    style="whitegrid",
    context="paper",
    font="sans-serif",
    rc={
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    },
)

ALLOWED_METHODS = [
    # MRL
    "SPIBB_mrl",
    "shield-SPIBB_mrl",
    "baseline_mrl",
    "shielded_baseline_mrl",

    # GreedyCut
    "SPIBB_GreedyCut",
    "shield-SPIBB_GreedyCut",
    "baseline_GreedyCut",
    "shielded_baseline_GreedyCut",

    # Grid
    "SPIBB_grid",
    "shield-SPIBB_grid",
    "baseline_grid",
    "shielded_baseline_grid",

    # DQN
    "spibb_dqn",
    "cql_dqn"
]


# Load Data
def read_data(filepath):
    data = pd.read_csv(filepath)

    if "nb_trajectories" in data.columns:
        data = data.rename(
            columns={"nb_trajectories": "length_trajectory"}
        )

    return data


def read_data_from_directory(directory_path):
    csv_files = glob.glob(
        os.path.join(directory_path, "*.csv")
    )

    combined_data = pd.DataFrame()

    for file in csv_files:
        print(f"Reading file: {file}")

        data = pd.read_csv(file)

        if "nb_trajectories" in data.columns:
            data = data.rename(
                columns={"nb_trajectories": "length_trajectory"}
            )

        combined_data = pd.concat(
            [combined_data, data],
            ignore_index=True
        )

    return combined_data


# Color Plots
def darken_color(color, factor=0.6):
    r, g, b = mcolors.to_rgb(color)

    return (
        r * factor,
        g * factor,
        b * factor
    )


def get_method_style_map():

    cmap = plt.get_cmap("Paired")

    style_map = {}

    # SPIBB color, baseline color
    variant_colors = {
        "mrl": (cmap(1), cmap(0)),
        "GreedyCut": (cmap(3), cmap(2)),
        "grid": (cmap(5), cmap(4)),
    }

    for variant, (spibb_color, baseline_color) in variant_colors.items():

        style_map[f"SPIBB_{variant}"] = {
            "color": spibb_color,
            "linestyle": "-"
        }

        style_map[f"shield-SPIBB_{variant}"] = {
            "color": spibb_color,
            "linestyle": "--"
        }

        style_map[f"baseline_{variant}"] = {
            "color": baseline_color,
            "linestyle": "-"
        }

        style_map[f"shielded_baseline_{variant}"] = {
            "color": baseline_color,
            "linestyle": "--"
        }

    style_map["spibb_dqn"] = {
        "color": cmap(7),
        "linestyle": "-"
    }

    style_map["cql_dqn"] = {
        "color": cmap(9),
        "linestyle": "-"
    }

    return style_map


# Legend
def build_custom_legend(style_map, data):

    handles = []

    present_methods = set(data["method"].unique())

    # SPIBB SECTION
    spibb_base = [
        ("SPIBB_mrl", "mrl SPIBB"),
        ("baseline_mrl", "mrl baseline"),

        ("SPIBB_GreedyCut", "GreedyCut SPIBB"),
        ("baseline_GreedyCut", "GreedyCut baseline"),

        ("SPIBB_grid", "grid SPIBB"),
        ("baseline_grid", "grid baseline"),
    ]

    spibb_handles = []

    for method, label in spibb_base:
        if method in present_methods:
            spibb_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=style_map[method]["color"],
                    linestyle="-",
                    linewidth=3,
                    label=label
                )
            )

    if spibb_handles:
        handles.append(
            mlines.Line2D([], [], linestyle="None", label="SPIBB")
        )
        handles.extend(spibb_handles)

        handles.append(
            mlines.Line2D([], [], linestyle="None", label="")
        )

    # SHIELDING SECTION
    shielding_present = any(
        m in present_methods
        for m in [
            "SPIBB_mrl",
            "baseline_mrl",
            "SPIBB_GreedyCut",
            "baseline_GreedyCut",
            "SPIBB_grid",
            "baseline_grid",
        ]
    )

    if shielding_present:

        handles.append(
            mlines.Line2D([], [], linestyle="None", label="SHIELDING")
        )

        handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="-",
                linewidth=3,
                label="standard"
            )
        )

        handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=3,
                label="shielded"
            )
        )

        handles.append(
            mlines.Line2D([], [], linestyle="None", label="")
        )

    # DQN SECTION
    dqn_present = any(
        m in present_methods for m in ["spibb_dqn", "cql_dqn"]
    )

    if dqn_present:

        handles.append(
            mlines.Line2D([], [], linestyle="None", label="DQN")
        )

        if "spibb_dqn" in present_methods:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=style_map["spibb_dqn"]["color"],
                    linestyle="-",
                    linewidth=3,
                    label="SPIBB-DQN"
                )
            )

        if "cql_dqn" in present_methods:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=style_map["cql_dqn"]["color"],
                    linestyle="-",
                    linewidth=3,
                    label="CQL-DQN"
                )
            )

    return handles


# Plotting
def create_combined_plot(data, output_file, env_name):

    # Keep only supported methods
    data = data[
        data["method"].isin(ALLOWED_METHODS)
    ].copy()

    style_map = get_method_style_map()

    grouped = (
        data.groupby(
            ["method", "length_trajectory"]
        )
        .mean(numeric_only=True)
        .reset_index()
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
        sharex=True
    )

    ax_perf = axes[0, 0]
    ax_legend = axes[0, 1]
    ax_success = axes[1, 0]
    ax_avoid = axes[1, 1]
    fig.suptitle(
        env_name,
        fontsize=22,
        fontweight="bold"
    )
    plot_specs = [
        (
            ax_perf,
            "method_perf",
            "Average Method Performance",
            "Performance"
        ),
        (
            ax_success,
            "success_rate",
            "Success Rate",
            "Success Rate"
        ),
        (
            ax_avoid,
            "avoid_rate",
            "Avoid Rate",
            "Avoid Rate"
        )
    ]

    dqn_methods = [
        "spibb_dqn",
        "cql_dqn"
    ]

    spibb_methods = [
        m for m in ALLOWED_METHODS
        if m not in dqn_methods
    ]

    for ax, metric, ylabel, title in plot_specs:

        # Plot DQN first
        for method in dqn_methods:

            method_data = grouped[
                grouped["method"] == method
            ]

            if method_data.empty:
                continue

            style = style_map[method]

            ax.plot(
                method_data["length_trajectory"],
                method_data[metric],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.5,
                zorder=1
            )

        # Plot SPIBB variants
        for method in spibb_methods:

            method_data = grouped[
                grouped["method"] == method
            ]

            if method_data.empty:
                continue

            style = style_map[method]

            ax.plot(
                method_data["length_trajectory"],
                method_data[metric],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.5,
                zorder=3
            )

        ax.set_title(title)
        ax.set_xlabel("Number of Trajectories")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    handles = build_custom_legend(style_map, data)

    ax_legend.axis("off")

    ax_legend.legend(
        handles=handles,
        loc="center",
        frameon=True,
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()



# Plot Discounted Performance
def create_discounted_performance_plot(data, output_file, env_name):

    if "discounted_method_perf" not in data.columns:
        print(
            "Column 'discounted_method_perf' not found. "
            "Skipping discounted plot."
        )
        return

    # Keep only supported methods
    data = data[
        data["method"].isin(ALLOWED_METHODS)
    ].copy()

    style_map = get_method_style_map()

    grouped = (
        data.groupby(
            ["method", "length_trajectory"]
        )
        .mean(numeric_only=True)
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    fig.suptitle(
        f"{env_name} - Discounted Performance",
        fontsize=22,
        fontweight="bold"
    )

    dqn_methods = [
        "spibb_dqn",
        "cql_dqn"
    ]

    spibb_methods = [
        m for m in ALLOWED_METHODS
        if m not in dqn_methods
    ]

    # Plot DQN
    for method in dqn_methods:

        method_data = grouped[
            grouped["method"] == method
        ]

        if method_data.empty:
            continue

        style = style_map[method]

        ax.plot(
            method_data["length_trajectory"],
            method_data["discounted_method_perf"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.5,
            zorder=1
        )

    # Plot SPIBB variants
    for method in spibb_methods:

        method_data = grouped[
            grouped["method"] == method
        ]

        if method_data.empty:
            continue

        style = style_map[method]

        ax.plot(
            method_data["length_trajectory"],
            method_data["discounted_method_perf"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.5,
            zorder=3
        )

    ax.set_title("Discounted Method Performance")
    ax.set_xlabel("Number of Trajectories")
    ax.set_ylabel("Discounted Performance")
    ax.grid(True, alpha=0.3)

    handles = build_custom_legend(style_map, data)

    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python plot.py <csv_file_or_directory> <environment_name>"
        )
        sys.exit(1)

    path = sys.argv[1]
    env_name = sys.argv[2]

    if os.path.isdir(path):
        data = read_data_from_directory(path)
    else:
        data = read_data(path)

    create_combined_plot(
        data,
        f"plot_combined_results_{env_name}.png",
        env_name
    )

    create_discounted_performance_plot(
        data,
        f"plot_discounted_results_{env_name}.png",
        env_name
    )