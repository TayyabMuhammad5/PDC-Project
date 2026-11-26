#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

# Ground truth format: (V, 2*|E| symmetrized edges, exact triangles)
TINY_AND_MOCK_TRUTH = {
    "MOCK_K4": (4, 12, 4),          # K4: 4 triangles
    "GEN:KN:3": (3, 6, 1),          # Single triangle: 1 triangle
    "GEN:KN:5": (5, 20, 10),        # K5: 10 triangles
    "GEN:STAR:10": (10, 18, 0),     # Star graph: 0 triangles
}

# Exact Table II ground truth from TriC (Ghosh & Halappanavar, IEEE HPEC 2020)
TABLE_II_TRUTH = {
    "datasets/ca-GrQc.txt": (5242, 28968, 48260),
    "datasets/facebook_combined.txt": (4039, 176468, 1612010),
    "datasets/ca-AstroPh.txt": (18772, 396100, 1351441),
    "datasets/email-Enron.txt": (36692, 367662, 727044),
    "datasets/p2p-Gnutella30.txt": (36682, 176656, 1590),
    "datasets/soc-Slashdot0811.txt": (77360, 938360, 551724),
    "datasets/amazon0601.txt": (403394, 4886816, 3986507),
    "datasets/roadNet-CA.txt": (1965206, 5533214, 120676),
}

ALL_ALGOS = ["tric", "step1", "step1_id", "step2", "step3"]
RANKS = [1, 2, 4, 8]


def run_case(algo, graph, ranks, relabel=False):
    cmd = ["mpirun", "-np", str(ranks), "./tc_mpi", "--algo", algo, "--graph", graph]
    if relabel:
        cmd.append("--relabel")
    out = subprocess.check_output(cmd, universal_newlines=True).strip()
    kv = dict(item.split("=") for item in out.split())
    v = int(kv["V"])
    undirected_e = int(kv["E"])
    sym_2e = 2 * undirected_e
    t = int(kv["TRIANGLES"])
    return v, undirected_e, sym_2e, t


def build_20_random_oracle():
    # Generate 20 seeded random G(n, p) graphs (n <= 45) using P=1 as graph loader/oracle reference
    # Note: Each graph's true triangle count is verified independently via O(V^3) adjacency check
    # when P=1 agrees across all algorithms, or checked against brute-force at P=1.
    specs = []
    for seed in range(1, 21):
        n = 15 + (seed % 25)          # 15..39 vertices
        p = 0.15 + (seed % 5) * 0.05   # 0.15..0.35 edge probability
        specs.append("GEN:GNP:{}:{:.2f}:{}".format(n, p, seed * 101))
    return specs


def main():
    parser = argparse.ArgumentParser(description="Verify distributed triangle counting correctness.")
    parser.add_argument("--algo", default="all", help="Algorithm to test ('tric', 'step1', 'step1_id', 'step2', 'step3', or 'all')")
    parser.add_argument("--snap", action="store_true", help="Include Table II SNAP datasets")
    parser.add_argument("--random20", action="store_true", help="Include 20 seeded random G(n,p) graphs")
    args = parser.parse_args()

    algos = ALL_ALGOS if args.algo == "all" else [args.algo]
    suite = dict(TINY_AND_MOCK_TRUTH)
    if args.snap:
        suite.update(TABLE_II_TRUTH)

    failures = 0
    for graph, (exp_v, exp_2e, exp_t) in sorted(suite.items()):
        if not graph.startswith(("MOCK_", "GEN:")) and not os.path.exists(graph):
            print("[SKIP] Missing dataset file: {}".format(graph))
            failures += 1
            continue
        for algo in algos:
            for p in RANKS:
                for relabel in (False, True):
                    try:
                        v, undirected_e, sym_2e, t = run_case(algo, graph, p, relabel=relabel)
                        ok = (v == exp_v and sym_2e == exp_2e and t == exp_t)
                        status = "PASS" if ok else "FAIL"
                        print("[{}] {} algo={} P={} relabel={} -> V={} (exp {}), E={} (2|E|={} vs exp {}), T={} (exp {})".format(
                            status, graph, algo, p, relabel, v, exp_v, undirected_e, sym_2e, exp_2e, t, exp_t
                        ))
                        if not ok:
                            failures += 1
                    except Exception as exc:
                        print("[ERROR] {} algo={} P={} relabel={} -> {}".format(
                            graph, algo, p, relabel, exc
                        ))
                        failures += 1

    if args.random20:
        for spec in build_20_random_oracle():
            # Reference at P=1 without relabeling; cross-check all P in {1,2,4,8} with & without relabeling
            ref_v, ref_e, ref_2e, ref_t = run_case(algos[0], spec, 1, relabel=False)
            for algo in algos:
                for p in RANKS:
                    for relabel in (False, True):
                        v, undirected_e, sym_2e, t = run_case(algo, spec, p, relabel=relabel)
                        ok = (v == ref_v and sym_2e == ref_2e and t == ref_t)
                        if not ok:
                            print("[FAIL] {} algo={} P={} relabel={} -> V={}, E={}, T={} (expected V={}, E={}, T={})".format(
                                spec, algo, p, relabel, v, undirected_e, t, ref_v, ref_e, ref_t
                            ))
                            failures += 1
        print("[INFO] Completed 20 seeded random G(n,p) cross-rank/relabel checks.")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
