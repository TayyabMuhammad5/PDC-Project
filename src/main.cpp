#include "dist_graph.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    std::string algo = "tric", graph_path = "MOCK_K4";
    bool relabel = false;
    uint64_t seed = 42;
    size_t max_buf = 16 * 1024 * 1024;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--algo" && i + 1 < argc) algo = argv[++i];
        else if (arg == "--graph" && i + 1 < argc) graph_path = argv[++i];
        else if (arg == "--relabel") relabel = true;
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (arg == "--max-buf" && i + 1 < argc) max_buf = std::stoull(argv[++i]);
    }

    DistCSR g = load_and_partition_graph(graph_path, relabel, seed, MPI_COMM_WORLD);
    RunMetrics res;
    if (algo == "tric") res = run_tric_lite(g, MPI_COMM_WORLD);
    else if (algo == "step1") res = run_ditric_step1(g, true, MPI_COMM_WORLD);
    else if (algo == "step1_id") res = run_ditric_step1(g, false, MPI_COMM_WORLD);
    else if (algo == "step2") res = run_ditric_step2(g, MPI_COMM_WORLD);
    else if (algo == "step3") res = run_ditric_step3(g, max_buf, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "ALGO=" << algo << " RANKS=" << num_ranks
                  << " V=" << g.global_n << " E=" << g.global_m
                  << " TRIANGLES=" << res.triangles << " WEDGES=" << res.wedges_examined
                  << " BYTES_SENT=" << res.bytes_sent << " MSGS=" << res.messages_sent
                  << " WAIT_S=" << res.mpi_wait_sec << " TIME_S=" << res.total_time_sec
                  << " PEAK_RSS_KB=" << res.peak_rss_kb << "\n";
    }
    MPI_Finalize();
    return 0;
}
