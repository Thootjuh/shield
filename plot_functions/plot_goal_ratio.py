import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Load data
with open("goal_counts.txt", "r") as f:
    text = f.read()

# Methods to plot
METHODS = [
    "spibb_mrl",
    "shielded_spibb_mrl",
    "spibb_grid",
    "shielded_spibb_grid",
    "spibb_GC",
    "shielded_spibb_GC",
]

# Parse file
data = defaultdict(dict)

current_samples = None

for line in text.splitlines():
    line = line.strip()

    if not line:
        continue

    # Sample count line (e.g. 250, 500, ...)
    if re.fullmatch(r"\d+", line):
        current_samples = int(line)
        continue

    # Method line
    match = re.match(
        r"(.+?)\.gif\s*:\s*\[(\d+)\s*,\s*(\d+)\]",
        line
    )

    if match and current_samples is not None:
        method = match.group(1)
        state1 = int(match.group(2))
        state2 = int(match.group(3))

        if method in METHODS:
            total = state1 + state2

            percentage_state2 = (
                100.0 * state2 / total
                if total > 0
                else 0.0
            )

            data[method][current_samples] = percentage_state2

# Plot
plt.figure(figsize=(10, 6))

for method in METHODS:
    if method not in data:
        continue

    xs = sorted(data[method].keys())
    ys = [data[method][x] for x in xs]

    plt.plot(
        xs,
        ys,
        marker="o",
        linewidth=2,
        label=method,
    )

plt.xlabel("Training Samples")
plt.ylabel("% Episodes Ending in State 2")
plt.title("State 2 Termination Rate vs Training Samples")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("plot_goal_ratios.png", dpi=300, bbox_inches="tight")
plt.close()