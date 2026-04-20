import re
import argparse
from collections import defaultdict

def parse_file(file_path):
    data = defaultdict(lambda: {"perf": [], "succ": [], "avoid": []})

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("----------"):
            i += 1
            continue

        method = line

        try:
            perf_line = lines[i + 1]
            succ_line = lines[i + 2]
            avoid_line = lines[i + 3]
        except IndexError:
            break

        perf = float(re.search(r"[-+]?\d*\.\d+|\d+", perf_line).group())
        succ = float(re.search(r"[-+]?\d*\.\d+|\d+", succ_line).group())
        avoid = float(re.search(r"[-+]?\d*\.\d+|\d+", avoid_line).group())

        data[method]["perf"].append(perf)
        data[method]["succ"].append(succ)
        data[method]["avoid"].append(avoid)

        i += 4

    return data


def compute_averages(data):
    for method, values in data.items():
        avg_perf = sum(values["perf"]) / len(values["perf"])
        avg_succ = sum(values["succ"]) / len(values["succ"])
        avg_avoid = sum(values["avoid"]) / len(values["avoid"])

        print(f"Method: {method}")
        print(f"  Avg perf: {avg_perf}")
        print(f"  Avg succ rate: {avg_succ}")
        print(f"  Avg avoid rate: {avg_avoid}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute averages from experiment log file")
    parser.add_argument("file_path", help="Path to the .txt file")

    args = parser.parse_args()

    data = parse_file(args.file_path)
    compute_averages(data)