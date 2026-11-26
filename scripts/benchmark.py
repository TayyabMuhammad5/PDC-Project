#!/usr/bin/env python3
import argparse
import csv
import os
import statistics
import subprocess

DATASETS = [
    ("ca-GrQc", "datasets/ca-GrQc.txt"),
    ("facebook_combined", "datasets/facebook_combined.txt"),
    ("p2p-Gnutella30", "datasets/p2p-Gnutella30.txt"),
    ("email-Enron", "datasets/email-Enron.txt"),
    ("ca-AstroPh", "datasets/ca-AstroPh.txt"),
    ("soc-Slashdot0811", "datasets/soc-Slashdot0811.txt"),
    ("amazon0601", "datasets/amazon0601.txt"),
    ("roadNet-CA", "datasets/roadNet-CA.txt"),
]

ALGOS = ["tric", "step1", "step1_id", "step2", "step3"]
RANKS = [1, 2, 4]


def run_once(algo, path, ranks, max_buf=0):
    cmd = ["mpirun", "-np", str(ranks), "./tc_mpi", "--algo", algo, "--graph", path]
    if max_buf > 0:
        cmd.extend(["--max-buf", str(max_buf)])
    out = subprocess.check_output(cmd, universal_newlines=True).strip()
    return dict(item.split("=") for item in out.split())


def main():
    parser = argparse.ArgumentParser(description="Run distributed triangle counting benchmarks.")
    parser.add_argument("--mode", default="cluster", choices=["cluster", "single_container"],
                        help="Placement mode label for CSV output")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed runs per config")
    parser.add_argument("--algos", nargs="+", default=ALGOS, help="Algorithms to benchmark")
    parser.add_argument("--ranks", nargs="+", type=int, default=RANKS, help="Rank counts to benchmark")
    parser.add_argument("--max-buf", type=int, default=0,
                        help="Buffer cap in bytes passed via --max-buf to tc_mpi (0 = tc_mpi default)")
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    out_csv = args.out or "results/benchmark_{}.csv".format(args.mode)

    fieldnames = [
        "mode", "dataset", "algo", "ranks", "max_buf", "V", "E", "triangles",
        "median_time_s", "min_time_s", "max_time_s", "time_per_edge_ns", "teps",
        "bytes_sent", "messages_sent", "wedges_examined",
        "peak_rss_kb", "imbalance_ratio", "mpi_wait_pct"
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, path in DATASETS:
            if not os.path.exists(path):
                print("[SKIP] Missing {}".format(path))
                continue

            for algo in args.algos:
                for p in args.ranks:
                    runs = []
                    for _ in range(args.repeats):
                        kv = run_once(algo, path, p, max_buf=args.max_buf)
                        runs.append(kv)

                    times = [float(r["TIME_S"]) for r in runs]
                    med_t = statistics.median(times)
                    min_t = min(times)
                    max_t = max(times)

                    mid_idx = times.index(sorted(times)[len(times) // 2])
                    ref = runs[mid_idx]
                    e_cnt = int(ref["E"])
                    teps = (e_cnt / med_t) if med_t > 0 else 0.0
                    ns_per_edge = (med_t * 1e9 / e_cnt) if e_cnt > 0 else 0.0

                    row = {
                        "mode": args.mode,
                        "dataset": name,
                        "algo": algo,
                        "ranks": p,
                        "max_buf": args.max_buf,
                        "V": ref.get("V", 0),
                        "E": e_cnt,
                        "triangles": ref.get("TRIANGLES", 0),
                        "median_time_s": "{:.6f}".format(med_t),
                        "min_time_s": "{:.6f}".format(min_t),
                        "max_time_s": "{:.6f}".format(max_t),
                        "time_per_edge_ns": "{:.3f}".format(ns_per_edge),
                        "teps": "{:.2f}".format(teps),
                        "bytes_sent": ref.get("BYTES_SENT", 0),
                        "messages_sent": ref.get("MSGS", 0),
                        "wedges_examined": ref.get("WEDGES", 0),
                        "peak_rss_kb": ref.get("PEAK_RSS_KB", 0),
                        "imbalance_ratio": ref.get("IMBALANCE", 0),
                        "mpi_wait_pct": ref.get("MPI_WAIT_PCT", 0),
                    }
                    writer.writerow(row)
                    f.flush()
                    print("[DONE] mode={} {} algo={} P={} max_buf={} median={:.4f}s ({:.1f} ns/edge) bytes={}".format(
                        args.mode, name, algo, p, args.max_buf, med_t, ns_per_edge, row["bytes_sent"]
                    ))


if __name__ == "__main__":
    main()
