#include "dist_graph.hpp"
#include "test_utils.hpp"
#include <iostream>
#include <string>
#include <exception>

static bool is_known_algo(const std::string& a) {
    return a == "tric" || a == "step1" || a == "step1_id" || a == "step2" ||
           a == "step3" || a == "brute";
}

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
        else {
            if (rank == 0) std::cerr << "ERROR: unknown or incomplete argument: " << arg << "\n";
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    if (!is_known_algo(algo)) {
        if (rank == 0) std::cerr << "ERROR: unknown --algo '" << algo
                                 << "' (tric, step1, step1_id, step2, step3, brute)\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    try {
        DistCSR g = load_and_partition_graph(graph_path, relabel, seed, MPI_COMM_WORLD);
        RunMetrics res;
        if (algo == "tric") res = run_tric_lite(g, MPI_COMM_WORLD);
        else if (algo == "step1") res = run_ditric_step1(g, true, MPI_COMM_WORLD);
        else if (algo == "step1_id") res = run_ditric_step1(g, false, MPI_COMM_WORLD);
        else if (algo == "step2") res = run_ditric_step2(g, MPI_COMM_WORLD);
        else if (algo == "step3") res = run_ditric_step3(g, max_buf, MPI_COMM_WORLD);
        else if (algo == "brute") {
            // Independent oracle: O(V^3) count on rank 0 for tiny graphs (MOCK_K4 or GEN: specs, |V| <= 200).
            if (rank == 0) {
                std::vector<std::vector<int64_t>> adj;
                if (graph_path == "MOCK_K4") adj = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
                else if (test_utils::is_generator_spec(graph_path)) adj = test_utils::generate_from_spec(graph_path);
                else throw std::runtime_error("--algo brute only supports MOCK_K4 and GEN: specs");
                res.triangles = static_cast<uint64_t>(test_utils::brute_force_triangles(adj));
            }
        }

        if (rank == 0) {
            std::cout << "ALGO=" << algo << " RANKS=" << num_ranks
                      << " V=" << g.global_n << " E=" << g.global_m
                      << " TRIANGLES=" << res.triangles << " WEDGES=" << res.wedges_examined
                      << " BYTES_SENT=" << res.bytes_sent << " MSGS=" << res.messages_sent
                      << " WAIT_S=" << res.mpi_wait_sec << " TIME_S=" << res.total_time_sec
                      << " PEAK_RSS_KB=" << res.peak_rss_kb << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] ERROR: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    MPI_Finalize();
    return 0;
}
