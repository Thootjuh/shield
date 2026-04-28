import sys
import re
from collections import defaultdict

def parse_line(line):
    """
    Parse a line like:
    method_name.gif : [31, 32]
    """
    pattern = r"(.+?)\.gif\s*:\s*\[(\d+),\s*(\d+)\]"
    match = re.match(pattern, line.strip())
    
    if match:
        method = match.group(1)
        goal1 = int(match.group(2))
        goal2 = int(match.group(3))
        return method, goal1, goal2
    return None


def main(filepath):
    # Store sums per method
    totals = defaultdict(lambda: {"g1": 0, "g2": 0})

    with open(filepath, 'r') as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            
            method, goal1, goal2 = parsed
            totals[method]["g1"] += goal1
            totals[method]["g2"] += goal2

    # Compute and print percentages
    for method, vals in totals.items():
        g1 = vals["g1"]
        g2 = vals["g2"]
        total = g1 + g2

        if total == 0:
            percentage = 0.0
        else:
            percentage = (g1 / total) * 100

        print(f"{method} {percentage:.2f}% first goal")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_txt_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    main(filepath)