#!/usr/bin/env python3
import argparse
import csv
import math

# Exact Table II TriC 32-PE Cori runtimes (seconds) and undirected edge counts |E|
PAPER_TABLE_II = {
    "ca-GrQc": {"E": 14484, "time_s": 0.002},
    "facebook_combined": {"E": 88234, "time_s": 0.067},
    "p2p-Gnutella30": {"E": 88328, "time_s": 0.004},
    "email-Enron": {"E": 183831, "time_s": 0.131},
    "ca-AstroPh": {"E": 198050, "time_s": 0.054},
    "soc-Slashdot0811": {"E": 469180, "time_s": 0.390},
    "amazon0601": {"E": 2443408, "time_s": 0.235},
    "roadNet-CA": {"E": 2766607, "time_s": 0.007},
}


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x, y):
    rx = average_ranks(x)
    ry = average_ranks(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Descriptive Spearman rank analysis vs Table II of TriC.")
    parser.add_argument("--csv", required=True, help="Path to results/benchmark_*.csv")
    parser.add_argument("--ranks", type=int, default=4, help="Rank count P to evaluate")
    parser.add_argument("--algo", default="tric", help="Algorithm name in CSV")
    args = parser.parse_args()

    obs = {}
    with open(args.csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["algo"] == args.algo and int(row["ranks"]) == args.ranks:
                obs[row["dataset"]] = {
                    "time_s": float(row["median_time_s"]),
                    "E": int(row["E"]),
                }

    common = [d for d in PAPER_TABLE_II if d in obs]
    n = len(common)
    if n < 3:
        raise SystemExit("Need at least 3 overlapping datasets in CSV; found {}".format(n))

    paper_t = [PAPER_TABLE_II[d]["time_s"] for d in common]
    our_t = [obs[d]["time_s"] for d in common]

    paper_tpe = [PAPER_TABLE_II[d]["time_s"] / PAPER_TABLE_II[d]["E"] for d in common]
    our_tpe = [obs[d]["time_s"] / obs[d]["E"] for d in common]

    paper_teps = [PAPER_TABLE_II[d]["E"] / PAPER_TABLE_II[d]["time_s"] for d in common]
    our_teps = [obs[d]["E"] / obs[d]["time_s"] for d in common]

    rho_time = spearman_rho(paper_t, our_t)
    rho_tpe = spearman_rho(paper_tpe, our_tpe)
    rho_teps = spearman_rho(paper_teps, our_teps)

    print("Sample size: N={} SNAP datasets (note: small-sample descriptive indicator, not a formal proof)".format(n))
    print("Datasets: {}".format(", ".join(common)))
    print("-" * 78)
    print("{:<20} {:>12} {:>12} {:>15} {:>15}".format(
        "Dataset", "TriC_t(s)", "Our_t(s)", "TriC_ns/edge", "Our_ns/edge"
    ))
    for d in common:
        p_t = PAPER_TABLE_II[d]["time_s"]
        o_t = obs[d]["time_s"]
        e = PAPER_TABLE_II[d]["E"]
        print("{:<20} {:>12.4f} {:>12.4f} {:>15.3f} {:>15.3f}".format(
            d, p_t, o_t, (p_t * 1e9) / e, (o_t * 1e9) / e
        ))
    print("-" * 78)
    print("Spearman rho (Wall Time):     {:.4f}  [Target >= 0.80: {}]".format(
        rho_time, "MET" if rho_time >= 0.80 else "BELOW TARGET"
    ))
    print("Spearman rho (Time per Edge): {:.4f}  [Target >= 0.80: {}]".format(
        rho_tpe, "MET" if rho_tpe >= 0.80 else "BELOW TARGET"
    ))
    print("Spearman rho (TEPS = 1/TPE):  {:.4f}  [Target >= 0.80: {}]".format(
        rho_teps, "MET" if rho_teps >= 0.80 else "BELOW TARGET"
    ))


if __name__ == "__main__":
    main()
